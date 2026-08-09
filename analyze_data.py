import pandas as pd

def analyze():
    print("--- Train Data ---")
    train_df = pd.read_csv('data/raw/train.csv')
    print("Shape:", train_df.shape)
    print("\nMissing values:\n", train_df.isnull().sum())
    print("\nData types:\n", train_df.dtypes)
    print("\nDescribe:\n", train_df.describe().to_string())
    print("\nFraud Class Balance:\n", train_df['is_fraud'].value_counts(normalize=True))
    print("\nMerchant Category counts:\n", train_df['merchant_category'].value_counts())
    
    print("\n--- Test Data ---")
    test_df = pd.read_csv('data/raw/test.csv')
    print("Shape:", test_df.shape)
    print("\nMissing values:\n", test_df.isnull().sum())

if __name__ == '__main__':
    analyze()
