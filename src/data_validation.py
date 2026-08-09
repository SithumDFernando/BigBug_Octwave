"""Data validation checks to run before pipeline execution."""
import pandas as pd
import numpy as np


def validate_data(train_df, test_df, sample_sub_df):
    """Run all validation checks. Raises AssertionError on failure."""
    checks_passed = 0
    total_checks = 0

    def check(condition, message):
        nonlocal checks_passed, total_checks
        total_checks += 1
        if condition:
            checks_passed += 1
            print(f"  [PASS] {message}")
        else:
            print(f"  [FAIL] {message}")
        return condition

    print("=" * 60)
    print("DATA VALIDATION")
    print("=" * 60)

    # Schema checks
    print("\n--- Schema ---")
    expected_train_cols = ['transaction_id', 'amount', 'transaction_hour', 'merchant_category',
                           'foreign_transaction', 'location_mismatch', 'device_trust_score',
                           'velocity_last_24h', 'cardholder_age', 'is_fraud']
    expected_test_cols = [c for c in expected_train_cols if c != 'is_fraud']

    check(list(train_df.columns) == expected_train_cols,
          f"Train columns match expected ({len(train_df.columns)} cols)")
    check(list(test_df.columns) == expected_test_cols,
          f"Test columns match expected ({len(test_df.columns)} cols)")

    # Shape checks
    print("\n--- Shape ---")
    check(train_df.shape[0] == 8000, f"Train has {train_df.shape[0]} rows (expected 8000)")
    check(test_df.shape[0] == 2000, f"Test has {test_df.shape[0]} rows (expected 2000)")
    check(sample_sub_df.shape[0] == 2000, f"Sample submission has {sample_sub_df.shape[0]} rows")

    # Missing values
    print("\n--- Missing Values ---")
    check(train_df.isnull().sum().sum() == 0, "Train has no missing values")
    check(test_df.isnull().sum().sum() == 0, "Test has no missing values")

    # Duplicates
    print("\n--- Duplicates ---")
    check(train_df['transaction_id'].is_unique, "Train transaction_id is unique")
    check(test_df['transaction_id'].is_unique, "Test transaction_id is unique")

    # ID alignment
    print("\n--- ID Alignment ---")
    check(set(test_df['transaction_id']) == set(sample_sub_df['transaction_id']),
          "Test IDs match sample_submission IDs")

    # Target checks
    print("\n--- Target ---")
    check(set(train_df['is_fraud'].unique()) == {0, 1}, "is_fraud is binary (0, 1)")
    fraud_rate = train_df['is_fraud'].mean()
    check(0.005 < fraud_rate < 0.05,
          f"Fraud rate = {fraud_rate:.4f} (expected ~1.5%)")

    # Value range checks
    print("\n--- Value Ranges ---")
    check(train_df['amount'].min() >= 0, f"Train amount min = {train_df['amount'].min():.2f} (>= 0)")
    check((train_df['transaction_hour'].min() >= 0) and (train_df['transaction_hour'].max() <= 23),
          f"Train hour range = [{train_df['transaction_hour'].min()}, {train_df['transaction_hour'].max()}]")
    check(set(train_df['foreign_transaction'].unique()) == {0, 1}, "foreign_transaction is binary")
    check(set(train_df['location_mismatch'].unique()) == {0, 1}, "location_mismatch is binary")
    check(len(train_df['merchant_category'].unique()) == 5,
          f"merchant_category has {len(train_df['merchant_category'].unique())} categories")

    # Distribution alignment (train vs test)
    print("\n--- Distribution Alignment ---")
    for col in ['amount', 'transaction_hour', 'foreign_transaction', 'location_mismatch',
                'device_trust_score', 'velocity_last_24h', 'cardholder_age']:
        diff = abs(train_df[col].mean() - test_df[col].mean())
        train_std = train_df[col].std()
        drift = diff / train_std if train_std > 0 else 0
        check(drift < 0.1,
              f"{col}: drift = {drift:.4f} (< 0.1 threshold)")

    print(f"\n{'=' * 60}")
    print(f"VALIDATION RESULT: {checks_passed}/{total_checks} checks passed")
    print(f"{'=' * 60}")

    if checks_passed < total_checks:
        print("\n[WARNING] Some validation checks failed!")
        return False
    return True
