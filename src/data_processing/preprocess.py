import pandas as pd
import numpy as np
import os

def load_data(base_dir):
    raw_dir = os.path.join(base_dir, "data", "raw")
    train = pd.read_csv(os.path.join(raw_dir, "train.csv"))
    test = pd.read_csv(os.path.join(raw_dir, "test.csv"))
    return train, test

def engineer_features(df):
    """Applies feature engineering logic to a dataframe."""
    
    # 1. Time-Based Risk Features
    df['is_night_transaction'] = df['transaction_hour'].apply(lambda x: 1 if 0 <= x <= 5 else 0)
    
    # Binning time of day
    def get_time_category(hour):
        if 0 <= hour <= 5: return 'Night'
        elif 6 <= hour <= 11: return 'Morning'
        elif 12 <= hour <= 17: return 'Afternoon'
        else: return 'Evening'
    
    df['time_of_day_category'] = df['transaction_hour'].apply(get_time_category)
    
    # 2. Trust and Anomaly Ratios
    # Add small epsilon to denominator to prevent division by zero (trust_score min is 25, so no div 0 anyway)
    df['amount_to_trust_ratio'] = df['amount'] / (df['device_trust_score'] + 1e-6)
    
    # +1 to avoid div by zero (velocity can be 0)
    df['amount_velocity_ratio'] = df['amount'] / (df['velocity_last_24h'] + 1)
    
    # 3. High-Risk Location Flags
    df['is_high_risk_location'] = ((df['foreign_transaction'] == 1) | (df['location_mismatch'] == 1)).astype(int)
    df['location_anomaly_score'] = df['foreign_transaction'] + df['location_mismatch']
    
    # 4. Strategic Binning
    df['is_high_amount'] = (df['amount'] > 500).astype(int)
    df['is_low_trust'] = (df['device_trust_score'] < 40).astype(int)
    
    return df

def preprocess_and_save():
    print("Starting Feature Engineering Pipeline...")
    
    base_dir = r"c:\Users\ADMIN\Desktop\BigBug_Octwave"
    processed_dir = os.path.join(base_dir, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    train, test = load_data(base_dir)
    print(f"Loaded raw data. Train shape: {train.shape}, Test shape: {test.shape}")
    
    # Save test IDs
    test_ids = test['transaction_id']
    test_ids.to_csv(os.path.join(processed_dir, "test_transaction_ids.csv"), index=False)
    
    # Drop IDs from feature sets
    train.drop(columns=['transaction_id'], inplace=True)
    test.drop(columns=['transaction_id'], inplace=True)
    
    # Separate target
    y_train = train['is_fraud']
    train.drop(columns=['is_fraud'], inplace=True)
    
    # Combine for consistent feature engineering
    combined = pd.concat([train, test], axis=0)
    
    print("Engineering features...")
    combined = engineer_features(combined)
    
    # One-Hot Encoding
    print("Applying One-Hot Encoding...")
    cat_cols = ['merchant_category', 'time_of_day_category']
    combined_encoded = pd.get_dummies(combined, columns=cat_cols, drop_first=False)
    
    # Convert booleans to integers
    for col in combined_encoded.columns:
        if combined_encoded[col].dtype == 'bool':
            combined_encoded[col] = combined_encoded[col].astype(int)
            
    # Split back
    train_engineered = combined_encoded.iloc[:len(train)].copy()
    test_engineered = combined_encoded.iloc[len(train):].copy()
    
    # Add target back to train
    train_engineered['is_fraud'] = y_train.values
    
    # Save
    print("Saving engineered datasets...")
    train_out = os.path.join(processed_dir, "train_engineered.csv")
    test_out = os.path.join(processed_dir, "test_engineered.csv")
    
    train_engineered.to_csv(train_out, index=False)
    test_engineered.to_csv(test_out, index=False)
    
    print(f"Engineered Train shape: {train_engineered.shape}")
    print(f"Engineered Test shape: {test_engineered.shape}")
    
    # Sanity checks
    assert train_engineered.isnull().sum().sum() == 0, "Null values found in train!"
    assert test_engineered.isnull().sum().sum() == 0, "Null values found in test!"
    assert 'is_fraud' in train_engineered.columns, "is_fraud missing from train!"
    assert 'is_fraud' not in test_engineered.columns, "is_fraud found in test!"
    
    print("Pipeline completed successfully!")

if __name__ == "__main__":
    preprocess_and_save()
