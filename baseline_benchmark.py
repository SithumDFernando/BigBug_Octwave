import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('data/raw/train.csv')

# ============ FEATURE ENGINEERING ============
def engineer_features(data):
    d = data.copy()
    
    # 1. Night hours flag (hours 0-3 are the key fraud window)
    d['is_night'] = d['transaction_hour'].isin([0,1,2,3]).astype(int)
    
    # 2. Low device trust (< 35 is the critical threshold from EDA)
    d['low_device_trust'] = (d['device_trust_score'] < 35).astype(int)
    
    # 3. High velocity (>= 5)
    d['high_velocity'] = (d['velocity_last_24h'] >= 5).astype(int)
    
    # 4. Risk flag count
    d['risk_flags'] = (d['foreign_transaction'] + d['location_mismatch'] + 
                       d['is_night'] + d['low_device_trust'] + d['high_velocity'])
    
    # 5. Foreign + location mismatch interaction
    d['foreign_loc_mismatch'] = d['foreign_transaction'] * d['location_mismatch']
    
    # 6. Foreign + high velocity
    d['foreign_high_vel'] = d['foreign_transaction'] * d['high_velocity']
    
    # 7. Night + foreign
    d['night_foreign'] = d['is_night'] * d['foreign_transaction']
    
    # 8. Night + low trust
    d['night_low_trust'] = d['is_night'] * d['low_device_trust']
    
    # 9. Amount log transform
    d['amount_log'] = np.log1p(d['amount'])
    
    # 10. High amount flag
    d['high_amount'] = (d['amount'] > 500).astype(int)
    
    # 11. Encode merchant_category
    le = LabelEncoder()
    d['merchant_cat_encoded'] = le.fit_transform(d['merchant_category'])
    
    return d, le

df_eng, le = engineer_features(df)

# Features to use
feature_cols = ['amount', 'amount_log', 'transaction_hour', 'foreign_transaction', 
                'location_mismatch', 'device_trust_score', 'velocity_last_24h', 
                'cardholder_age', 'merchant_cat_encoded',
                'is_night', 'low_device_trust', 'high_velocity', 'risk_flags',
                'foreign_loc_mismatch', 'foreign_high_vel', 'night_foreign', 
                'night_low_trust', 'high_amount']

X = df_eng[feature_cols]
y = df_eng['is_fraud']

# ============ CROSS-VALIDATION ============
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
}

print("=" * 70)
print("5-FOLD STRATIFIED CROSS-VALIDATION RESULTS")
print("=" * 70)

for name, model in models.items():
    f1_scores = []
    prec_scores = []
    rec_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        f1 = f1_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred)
        
        f1_scores.append(f1)
        prec_scores.append(prec)
        rec_scores.append(rec)
    
    print(f"\n{name}:")
    print(f"  F1:        {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}  (folds: {[f'{x:.4f}' for x in f1_scores]})")
    print(f"  Precision: {np.mean(prec_scores):.4f} ± {np.std(prec_scores):.4f}")
    print(f"  Recall:    {np.mean(rec_scores):.4f} ± {np.std(rec_scores):.4f}")

# Feature importance from best model
print("\n" + "=" * 70)
print("FEATURE IMPORTANCES (GradientBoosting, full training)")
print("=" * 70)
gb = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
gb.fit(X, y)
importances = pd.Series(gb.feature_importances_, index=feature_cols).sort_values(ascending=False)
for feat, imp in importances.items():
    print(f"  {feat:30s} {imp:.4f}")

# Also try with SMOTE-like oversampling via class_weight
print("\n" + "=" * 70)
print("GRADIENT BOOSTING WITH DIFFERENT CONFIGS")
print("=" * 70)

configs = [
    ('GB_default', GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)),
    ('GB_deep', GradientBoostingClassifier(n_estimators=300, max_depth=7, learning_rate=0.05, random_state=42)),
    ('GB_shallow', GradientBoostingClassifier(n_estimators=500, max_depth=3, learning_rate=0.05, random_state=42)),
    ('RF_deep', RandomForestClassifier(n_estimators=500, max_depth=15, class_weight='balanced_subsample', random_state=42)),
]

for name, model in configs:
    f1_scores = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        f1_scores.append(f1_score(y_val, y_pred))
    print(f"  {name:20s} F1: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")

# Try XGBoost if available
try:
    from xgboost import XGBClassifier
    print("\n" + "=" * 70)
    print("XGBOOST MODELS")
    print("=" * 70)
    
    xgb_configs = [
        ('XGB_default', XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, 
                                       scale_pos_weight=len(y[y==0])/len(y[y==1]),
                                       eval_metric='logloss', random_state=42, verbosity=0)),
        ('XGB_tuned', XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                     scale_pos_weight=len(y[y==0])/len(y[y==1]),
                                     min_child_weight=3, gamma=0.1, subsample=0.8,
                                     colsample_bytree=0.8, eval_metric='logloss',
                                     random_state=42, verbosity=0)),
    ]
    
    for name, model in xgb_configs:
        f1_scores = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            f1_scores.append(f1_score(y_val, y_pred))
        print(f"  {name:20s} F1: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
except ImportError:
    print("\nXGBoost not installed, skipping.")

# Try LightGBM if available
try:
    from lightgbm import LGBMClassifier
    print("\n" + "=" * 70)
    print("LIGHTGBM MODELS")
    print("=" * 70)
    
    lgbm_configs = [
        ('LGBM_default', LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                         is_unbalance=True, random_state=42, verbose=-1)),
        ('LGBM_tuned', LGBMClassifier(n_estimators=300, max_depth=7, learning_rate=0.05,
                                       is_unbalance=True, min_child_samples=5,
                                       num_leaves=31, random_state=42, verbose=-1)),
    ]
    
    for name, model in lgbm_configs:
        f1_scores = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            f1_scores.append(f1_score(y_val, y_pred))
        print(f"  {name:20s} F1: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
except ImportError:
    print("\nLightGBM not installed, skipping.")
