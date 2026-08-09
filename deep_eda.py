import pandas as pd
import numpy as np

df = pd.read_csv('data/raw/train.csv')
fraud = df[df['is_fraud']==1]
legit = df[df['is_fraud']==0]

print("=== FRAUD vs LEGIT COMPARISON ===")
print(f"Total: {len(df)}, Fraud: {len(fraud)} ({len(fraud)/len(df)*100:.2f}%), Legit: {len(legit)}")

print("\n=== AMOUNT ANALYSIS ===")
print(f"Fraud  - mean: {fraud['amount'].mean():.2f}, median: {fraud['amount'].median():.2f}, min: {fraud['amount'].min():.2f}, max: {fraud['amount'].max():.2f}")
print(f"Legit  - mean: {legit['amount'].mean():.2f}, median: {legit['amount'].median():.2f}, min: {legit['amount'].min():.2f}, max: {legit['amount'].max():.2f}")

# Amount quantiles for fraud
print("\nFraud amount percentiles:")
for p in [10, 25, 50, 75, 90, 95]:
    print(f"  {p}th: {fraud['amount'].quantile(p/100):.2f}")

print("\n=== TRANSACTION HOUR (fraud rate by hour) ===")
hr = df.groupby('transaction_hour')['is_fraud'].agg(['sum','count','mean']).sort_values('mean', ascending=False)
hr.columns = ['fraud_count','total','fraud_rate']
print(hr.to_string())

print("\n=== MERCHANT CATEGORY (fraud rate) ===")
mc = df.groupby('merchant_category')['is_fraud'].agg(['sum','count','mean'])
mc.columns = ['fraud_count','total','fraud_rate']
print(mc.sort_values('fraud_rate', ascending=False).to_string())

print("\n=== FOREIGN TRANSACTION (fraud rate) ===")
ft = df.groupby('foreign_transaction')['is_fraud'].agg(['sum','count','mean'])
ft.columns = ['fraud_count','total','fraud_rate']
print(ft.to_string())

print("\n=== LOCATION MISMATCH (fraud rate) ===")
lm = df.groupby('location_mismatch')['is_fraud'].agg(['sum','count','mean'])
lm.columns = ['fraud_count','total','fraud_rate']
print(lm.to_string())

print("\n=== DEVICE TRUST SCORE ===")
print(f"Fraud  - mean: {fraud['device_trust_score'].mean():.2f}, median: {fraud['device_trust_score'].median():.0f}")
print(f"Legit  - mean: {legit['device_trust_score'].mean():.2f}, median: {legit['device_trust_score'].median():.0f}")
df['dts_bin'] = pd.cut(df['device_trust_score'], bins=[24,35,50,65,80,100], labels=['25-35','36-50','51-65','66-80','81-99'])
dts = df.groupby('dts_bin')['is_fraud'].agg(['sum','count','mean'])
dts.columns = ['fraud_count','total','fraud_rate']
print(dts.to_string())

print("\n=== VELOCITY LAST 24H ===")
print(f"Fraud  - mean: {fraud['velocity_last_24h'].mean():.2f}, median: {fraud['velocity_last_24h'].median():.0f}")
print(f"Legit  - mean: {legit['velocity_last_24h'].mean():.2f}, median: {legit['velocity_last_24h'].median():.0f}")
vel = df.groupby('velocity_last_24h')['is_fraud'].agg(['sum','count','mean'])
vel.columns = ['fraud_count','total','fraud_rate']
print(vel.to_string())

print("\n=== CARDHOLDER AGE ===")
print(f"Fraud  - mean: {fraud['cardholder_age'].mean():.2f}, median: {fraud['cardholder_age'].median():.0f}")
print(f"Legit  - mean: {legit['cardholder_age'].mean():.2f}, median: {legit['cardholder_age'].median():.0f}")
df['age_bin'] = pd.cut(df['cardholder_age'], bins=[17,25,35,45,55,70], labels=['18-25','26-35','36-45','46-55','56-69'])
age = df.groupby('age_bin')['is_fraud'].agg(['sum','count','mean'])
age.columns = ['fraud_count','total','fraud_rate']
print(age.to_string())

print("\n=== INTERACTION: foreign + location_mismatch ===")
df['foreign_loc'] = df['foreign_transaction'].astype(str) + '_' + df['location_mismatch'].astype(str)
fl = df.groupby('foreign_loc')['is_fraud'].agg(['sum','count','mean'])
fl.columns = ['fraud_count','total','fraud_rate']
print(fl.to_string())

print("\n=== INTERACTION: foreign + high velocity (>=4) ===")
df['foreign_highvel'] = df['foreign_transaction'].astype(str) + '_' + (df['velocity_last_24h']>=4).astype(int).astype(str)
fhv = df.groupby('foreign_highvel')['is_fraud'].agg(['sum','count','mean'])
fhv.columns = ['fraud_count','total','fraud_rate']
print(fhv.to_string())

print("\n=== CORRELATION MATRIX (numeric features vs is_fraud) ===")
numeric_cols = ['amount','transaction_hour','foreign_transaction','location_mismatch',
                'device_trust_score','velocity_last_24h','cardholder_age','is_fraud']
corr = df[numeric_cols].corr()['is_fraud'].drop('is_fraud').sort_values(ascending=False)
print(corr.to_string())

print("\n=== AMOUNT BINS vs FRAUD RATE ===")
df['amount_bin'] = pd.cut(df['amount'], bins=[0,50,100,200,400,600,1500], labels=['0-50','50-100','100-200','200-400','400-600','600+'])
ab = df.groupby('amount_bin')['is_fraud'].agg(['sum','count','mean'])
ab.columns = ['fraud_count','total','fraud_rate']
print(ab.to_string())

print("\n=== HIGH-RISK COMBO: foreign=1 AND location_mismatch=1 AND device_trust_score<50 ===")
high_risk = df[(df['foreign_transaction']==1) & (df['location_mismatch']==1) & (df['device_trust_score']<50)]
print(f"Count: {len(high_risk)}, Fraud: {high_risk['is_fraud'].sum()}, Rate: {high_risk['is_fraud'].mean():.4f}")

print("\n=== HIGH-RISK COMBO: foreign=1 AND velocity>=4 AND amount>300 ===")
high_risk2 = df[(df['foreign_transaction']==1) & (df['velocity_last_24h']>=4) & (df['amount']>300)]
print(f"Count: {len(high_risk2)}, Fraud: {high_risk2['is_fraud'].sum()}, Rate: {high_risk2['is_fraud'].mean():.4f}")

print("\n=== NIGHT HOURS (22-5) vs DAY ===")
df['is_night'] = ((df['transaction_hour'] >= 22) | (df['transaction_hour'] <= 5)).astype(int)
night = df.groupby('is_night')['is_fraud'].agg(['sum','count','mean'])
night.columns = ['fraud_count','total','fraud_rate']
print(night.to_string())
