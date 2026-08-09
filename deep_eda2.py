import pandas as pd
import numpy as np

df = pd.read_csv('data/raw/train.csv')

# More granular night hours
print("=== HOUR-LEVEL FRAUD DETAIL (hours 0-5) ===")
night_df = df[df['transaction_hour'].isin([0,1,2,3,4,5])]
print(f"Night txns: {len(night_df)}, Fraud: {night_df['is_fraud'].sum()}, Rate: {night_df['is_fraud'].mean():.4f}")

# Triple interaction: night + foreign + low device trust
print("\n=== TRIPLE: night(0-3) + foreign + low_trust(<50) ===")
triple = df[(df['transaction_hour'].isin([0,1,2,3])) & 
            (df['foreign_transaction']==1) & 
            (df['device_trust_score']<50)]
print(f"Count: {len(triple)}, Fraud: {triple['is_fraud'].sum()}, Rate: {triple['is_fraud'].mean():.4f}" if len(triple) > 0 else "No records")

# Triple interaction: night + location_mismatch + low device trust
print("\n=== TRIPLE: night(0-3) + loc_mismatch + low_trust(<50) ===")
triple2 = df[(df['transaction_hour'].isin([0,1,2,3])) & 
             (df['location_mismatch']==1) & 
             (df['device_trust_score']<50)]
print(f"Count: {len(triple2)}, Fraud: {triple2['is_fraud'].sum()}, Rate: {triple2['is_fraud'].mean():.4f}" if len(triple2) > 0 else "No records")

# velocity 5+ analysis
print("\n=== VELOCITY >= 5 DETAIL ===")
high_vel = df[df['velocity_last_24h'] >= 5]
print(f"Count: {len(high_vel)}, Fraud: {high_vel['is_fraud'].sum()}, Rate: {high_vel['is_fraud'].mean():.4f}")
print("With foreign=1:", len(high_vel[high_vel['foreign_transaction']==1]), 
      "fraud:", high_vel[high_vel['foreign_transaction']==1]['is_fraud'].sum())
print("With loc_mismatch=1:", len(high_vel[high_vel['location_mismatch']==1]),
      "fraud:", high_vel[high_vel['location_mismatch']==1]['is_fraud'].sum())

# Device trust score < 35 is clearly the key threshold
print("\n=== DEVICE TRUST SCORE FINE-GRAINED (25-45) ===")
for lower in range(25, 46, 5):
    upper = lower + 4
    subset = df[(df['device_trust_score'] >= lower) & (df['device_trust_score'] <= upper)]
    if len(subset) > 0:
        print(f"  {lower}-{upper}: count={len(subset)}, fraud={subset['is_fraud'].sum()}, rate={subset['is_fraud'].mean():.4f}")

# Risk score simulation
print("\n=== SIMULATED RISK SCORING ===")
df['risk_score'] = 0
df.loc[df['foreign_transaction']==1, 'risk_score'] += 1
df.loc[df['location_mismatch']==1, 'risk_score'] += 1
df.loc[df['device_trust_score'] < 35, 'risk_score'] += 1
df.loc[df['velocity_last_24h'] >= 5, 'risk_score'] += 1
df.loc[df['transaction_hour'].isin([0,1,2,3]), 'risk_score'] += 1

rs = df.groupby('risk_score')['is_fraud'].agg(['sum','count','mean'])
rs.columns = ['fraud_count', 'total', 'fraud_rate']
print(rs.to_string())

# What do fraud cases look like?
print("\n=== ALL 121 FRAUD CASES - risk_score distribution ===")
fraud_rs = df[df['is_fraud']==1]['risk_score'].value_counts().sort_index()
print(fraud_rs.to_string())

# Check how many fraud cases have risk_score >= 2
high_risk_fraud = df[(df['is_fraud']==1) & (df['risk_score'] >= 2)]
print(f"\nFraud with risk_score >= 2: {len(high_risk_fraud)} out of 121 ({len(high_risk_fraud)/121*100:.1f}%)")
print(f"Legit with risk_score >= 2: {len(df[(df['is_fraud']==0) & (df['risk_score'] >= 2)])} out of 7879")

# Amount for fraud cases in each risk score bin
print("\n=== FRAUD AMOUNT by RISK SCORE ===")
fraud_df = df[df['is_fraud']==1]
for rs_val in sorted(fraud_df['risk_score'].unique()):
    subset = fraud_df[fraud_df['risk_score']==rs_val]
    print(f"  risk_score={rs_val}: n={len(subset)}, mean_amt={subset['amount'].mean():.2f}, median_amt={subset['amount'].median():.2f}")

# Test data distribution check
print("\n=== TEST DATA DISTRIBUTION CHECK ===")
test_df = pd.read_csv('data/raw/test.csv')
print("Train vs Test feature distributions:")
for col in ['amount','transaction_hour','foreign_transaction','location_mismatch','device_trust_score','velocity_last_24h','cardholder_age']:
    print(f"  {col}: train_mean={df[col].mean():.3f}, test_mean={test_df[col].mean():.3f}, diff={abs(df[col].mean()-test_df[col].mean()):.4f}")

print("\nMerchant category distribution (test):")
print(test_df['merchant_category'].value_counts().to_string())
