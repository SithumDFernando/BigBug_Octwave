import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('data/raw/train.csv')

def engineer_features_v1(data):
    """Original feature set (F1=0.987)"""
    d = data.copy()
    d['is_night'] = d['transaction_hour'].isin([0,1,2,3]).astype(int)
    d['low_device_trust'] = (d['device_trust_score'] < 35).astype(int)
    d['high_velocity'] = (d['velocity_last_24h'] >= 5).astype(int)
    d['risk_flags'] = (d['foreign_transaction'] + d['location_mismatch'] + 
                       d['is_night'] + d['low_device_trust'] + d['high_velocity'])
    d['foreign_loc_mismatch'] = d['foreign_transaction'] * d['location_mismatch']
    d['foreign_high_vel'] = d['foreign_transaction'] * d['high_velocity']
    d['night_foreign'] = d['is_night'] * d['foreign_transaction']
    d['night_low_trust'] = d['is_night'] * d['low_device_trust']
    d['amount_log'] = np.log1p(d['amount'])
    d['high_amount'] = (d['amount'] > 500).astype(int)
    le = LabelEncoder()
    d['merchant_cat_encoded'] = le.fit_transform(d['merchant_category'])
    return d

def engineer_features_v2(data):
    """Enhanced features - trying to catch the edge cases"""
    d = data.copy()
    # Broaden thresholds slightly
    d['is_night'] = d['transaction_hour'].isin([0,1,2,3]).astype(int)
    d['is_night_broad'] = d['transaction_hour'].isin([0,1,2,3,4,5]).astype(int)
    d['low_device_trust'] = (d['device_trust_score'] < 35).astype(int)
    d['low_device_trust_v2'] = (d['device_trust_score'] < 40).astype(int)  # catches edge cases at 36-37
    d['high_velocity'] = (d['velocity_last_24h'] >= 5).astype(int)
    
    # Multiple risk score variants
    d['risk_flags'] = (d['foreign_transaction'] + d['location_mismatch'] + 
                       d['is_night'] + d['low_device_trust'] + d['high_velocity'])
    d['risk_flags_v2'] = (d['foreign_transaction'] + d['location_mismatch'] + 
                          d['is_night_broad'] + d['low_device_trust_v2'] + d['high_velocity'])
    
    # Interactions
    d['foreign_loc_mismatch'] = d['foreign_transaction'] * d['location_mismatch']
    d['foreign_high_vel'] = d['foreign_transaction'] * d['high_velocity']
    d['night_foreign'] = d['is_night'] * d['foreign_transaction']
    d['night_low_trust'] = d['is_night'] * d['low_device_trust']
    d['night_loc_mismatch'] = d['is_night'] * d['location_mismatch']
    d['foreign_low_trust'] = d['foreign_transaction'] * d['low_device_trust_v2']
    
    # Amount features - key for edge cases (those 2 missed frauds had amount ~$1000)
    d['amount_log'] = np.log1p(d['amount'])
    d['high_amount'] = (d['amount'] > 500).astype(int)
    d['very_high_amount'] = (d['amount'] > 800).astype(int)
    d['amount_squared'] = d['amount'] ** 2
    
    # Night + high amount (catches the 2 edge cases: night + amount ~$1000 + trust 36-37)
    d['night_high_amount'] = d['is_night'] * d['high_amount']
    d['night_low_trust_v2'] = d['is_night'] * d['low_device_trust_v2']
    
    # Composite ratios
    d['trust_velocity_ratio'] = d['device_trust_score'] / (d['velocity_last_24h'] + 1)
    d['amount_per_velocity'] = d['amount'] / (d['velocity_last_24h'] + 1)
    
    le = LabelEncoder()
    d['merchant_cat_encoded'] = le.fit_transform(d['merchant_category'])
    return d

# ============ FIRST: Diagnose what's being missed ============
print("=" * 70)
print("DIAGNOSING MISSED CASES (v1 features)")
print("=" * 70)

df_v1 = engineer_features_v1(df)
feature_cols_v1 = ['amount', 'amount_log', 'transaction_hour', 'foreign_transaction', 
                   'location_mismatch', 'device_trust_score', 'velocity_last_24h', 
                   'cardholder_age', 'merchant_cat_encoded', 'is_night', 'low_device_trust', 
                   'high_velocity', 'risk_flags', 'foreign_loc_mismatch', 'foreign_high_vel', 
                   'night_foreign', 'night_low_trust', 'high_amount']

X_v1 = df_v1[feature_cols_v1]
y = df_v1['is_fraud']

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_missed = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_v1, y)):
    X_train, X_val = X_v1.iloc[train_idx], X_v1.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    
    f1 = f1_score(y_val, y_pred)
    
    # Find missed fraud cases (FN)
    fn_mask = (y_val == 1) & (y_pred == 0)
    fn_indices = val_idx[fn_mask.values]
    
    print(f"\nFold {fold}: F1={f1:.4f}, Fraud in fold={y_val.sum()}, Missed={fn_mask.sum()}")
    if fn_mask.sum() > 0:
        missed = df.iloc[fn_indices][['transaction_id','amount','transaction_hour','merchant_category',
                                      'foreign_transaction','location_mismatch','device_trust_score',
                                      'velocity_last_24h','cardholder_age','is_fraud']]
        missed['predicted_prob'] = y_prob[fn_mask.values]
        print(missed.to_string())
        all_missed.extend(fn_indices.tolist())

print(f"\n\nTotal unique missed fraud cases across all folds: {len(set(all_missed))}")
print(f"Indices: {sorted(set(all_missed))}")

# Show these cases in detail
if all_missed:
    print("\n=== ALL MISSED FRAUD CASES (detail) ===")
    for idx in sorted(set(all_missed)):
        row = df.iloc[idx]
        print(f"  idx={idx}: amount={row['amount']:.2f}, hour={row['transaction_hour']}, "
              f"foreign={row['foreign_transaction']}, loc_mis={row['location_mismatch']}, "
              f"trust={row['device_trust_score']}, vel={row['velocity_last_24h']}, "
              f"age={row['cardholder_age']}, cat={row['merchant_category']}")

# ============ NOW: Try v2 features ============
print("\n" + "=" * 70)
print("V2 FEATURES - ENHANCED")
print("=" * 70)

df_v2 = engineer_features_v2(df)
feature_cols_v2 = ['amount', 'amount_log', 'amount_squared', 'transaction_hour', 
                   'foreign_transaction', 'location_mismatch', 'device_trust_score', 
                   'velocity_last_24h', 'cardholder_age', 'merchant_cat_encoded',
                   'is_night', 'is_night_broad', 'low_device_trust', 'low_device_trust_v2',
                   'high_velocity', 'risk_flags', 'risk_flags_v2',
                   'foreign_loc_mismatch', 'foreign_high_vel', 'night_foreign', 
                   'night_low_trust', 'night_loc_mismatch', 'foreign_low_trust',
                   'high_amount', 'very_high_amount', 'night_high_amount', 
                   'night_low_trust_v2', 'trust_velocity_ratio', 'amount_per_velocity']

X_v2 = df_v2[feature_cols_v2]

for fold, (train_idx, val_idx) in enumerate(skf.split(X_v2, y)):
    X_train, X_val = X_v2.iloc[train_idx], X_v2.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_pred)
    fn = ((y_val==1) & (y_pred==0)).sum()
    fp = ((y_val==0) & (y_pred==1)).sum()
    print(f"  Fold {fold}: F1={f1:.4f}, FN={fn}, FP={fp}")

# ============ Try different GB configs with v2 features ============
print("\n" + "=" * 70)
print("HYPERPARAMETER SWEEP (v2 features)")
print("=" * 70)

configs = [
    ('GB_200_d5_lr01', dict(n_estimators=200, max_depth=5, learning_rate=0.1)),
    ('GB_300_d5_lr01', dict(n_estimators=300, max_depth=5, learning_rate=0.1)),
    ('GB_500_d5_lr005', dict(n_estimators=500, max_depth=5, learning_rate=0.05)),
    ('GB_200_d6_lr01', dict(n_estimators=200, max_depth=6, learning_rate=0.1)),
    ('GB_300_d6_lr005', dict(n_estimators=300, max_depth=6, learning_rate=0.05)),
    ('GB_200_d7_lr01', dict(n_estimators=200, max_depth=7, learning_rate=0.1)),
    ('GB_500_d7_lr005', dict(n_estimators=500, max_depth=7, learning_rate=0.05)),
    ('GB_300_d4_lr01', dict(n_estimators=300, max_depth=4, learning_rate=0.1)),
    ('GB_1000_d3_lr005', dict(n_estimators=1000, max_depth=3, learning_rate=0.05)),
    ('GB_200_d5_lr02', dict(n_estimators=200, max_depth=5, learning_rate=0.2)),
]

results = []
for name, params in configs:
    f1_scores = []
    fn_total = 0
    fp_total = 0
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_v2, y)):
        X_train, X_val = X_v2.iloc[train_idx], X_v2.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = GradientBoostingClassifier(random_state=42, **params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        f1_scores.append(f1_score(y_val, y_pred))
        fn_total += ((y_val==1) & (y_pred==0)).sum()
        fp_total += ((y_val==0) & (y_pred==1)).sum()
    mean_f1 = np.mean(f1_scores)
    results.append((name, mean_f1, np.std(f1_scores), fn_total, fp_total, f1_scores))
    
results.sort(key=lambda x: -x[1])
for name, mean_f1, std_f1, fn, fp, folds in results:
    perfect = sum(1 for f in folds if f >= 0.999)
    print(f"  {name:25s} F1={mean_f1:.4f}±{std_f1:.4f}  FN={fn} FP={fp}  Perfect folds: {perfect}/5  {[f'{x:.4f}' for x in folds]}")

# ============ Threshold tuning on best config ============
print("\n" + "=" * 70)
print("PROBABILITY THRESHOLD TUNING (best config)")  
print("=" * 70)

best_params = results[0]
print(f"Best config: {results[0][0]}")

# Use the best config params - parse from name
best_name = results[0][0]
# Just use what worked best, re-run with threshold tuning
for threshold in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
    f1_scores = []
    fn_total = 0
    fp_total = 0
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_v2, y)):
        X_train, X_val = X_v2.iloc[train_idx], X_v2.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        # Use the top config
        best_cfg = configs[[c[0] for c in configs].index(best_name)][1]
        model = GradientBoostingClassifier(random_state=42, **best_cfg)
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_val)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
        f1_scores.append(f1_score(y_val, y_pred))
        fn_total += ((y_val==1) & (y_pred==0)).sum()
        fp_total += ((y_val==0) & (y_pred==1)).sum()
    mean_f1 = np.mean(f1_scores)
    perfect = sum(1 for f in f1_scores if f >= 0.999)
    print(f"  threshold={threshold:.2f}  F1={mean_f1:.4f}±{np.std(f1_scores):.4f}  FN={fn_total} FP={fp_total}  Perfect: {perfect}/5  {[f'{x:.4f}' for x in f1_scores]}")
