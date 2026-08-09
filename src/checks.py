"""Post-prediction sanity checks."""
import pandas as pd
import numpy as np


def validate_submission(submission_df, sample_sub_df, train_fraud_rate=0.015125):
    """
    Validate the submission file before Kaggle upload.
    
    Returns True if all checks pass.
    """
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
    print("SUBMISSION VALIDATION")
    print("=" * 60)

    # Format checks
    print("\n--- Format ---")
    check(list(submission_df.columns) == ['transaction_id', 'is_fraud'],
          f"Columns: {list(submission_df.columns)}")
    check(len(submission_df) == 2000,
          f"Row count: {len(submission_df)} (expected 2000)")
    check(set(submission_df['is_fraud'].unique()).issubset({0, 1}),
          f"is_fraud values: {sorted(submission_df['is_fraud'].unique())}")

    # ID checks
    print("\n--- ID Integrity ---")
    check(submission_df['transaction_id'].is_unique,
          "All transaction_ids are unique")
    check(set(submission_df['transaction_id']) == set(sample_sub_df['transaction_id']),
          "IDs match sample_submission exactly")

    # Distribution checks
    print("\n--- Prediction Distribution ---")
    pred_fraud_rate = submission_df['is_fraud'].mean()
    check(0.005 < pred_fraud_rate < 0.03,
          f"Predicted fraud rate: {pred_fraud_rate:.4f} (expected ~{train_fraud_rate:.4f})")
    
    n_fraud = submission_df['is_fraud'].sum()
    n_legit = len(submission_df) - n_fraud
    print(f"  Predicted: {n_fraud} fraud, {n_legit} legitimate")

    print(f"\n{'=' * 60}")
    print(f"SUBMISSION RESULT: {checks_passed}/{total_checks} checks passed")
    print(f"{'=' * 60}")

    return checks_passed == total_checks


def compare_with_heuristic(test_df_engineered, submission_df):
    """
    Compare ML predictions with the simple risk_score >= 3 heuristic.
    Log any disagreements.
    """
    print("\n" + "=" * 60)
    print("HEURISTIC COMPARISON (risk_flags >= 3)")
    print("=" * 60)

    heuristic_pred = (test_df_engineered['risk_flags'] >= 3).astype(int)
    ml_pred = submission_df.set_index('transaction_id')['is_fraud']
    test_ids = test_df_engineered['transaction_id']

    # Align
    heuristic_series = pd.Series(heuristic_pred.values, index=test_ids.values)
    
    agree = (heuristic_series == ml_pred.reindex(heuristic_series.index)).sum()
    disagree = len(heuristic_series) - agree

    print(f"  Agreement: {agree}/{len(heuristic_series)} ({agree/len(heuristic_series)*100:.1f}%)")
    print(f"  Disagreements: {disagree}")

    if disagree > 0:
        # ML says fraud, heuristic says no
        ml_extra = heuristic_series.index[(ml_pred.reindex(heuristic_series.index) == 1) & (heuristic_series == 0)]
        # Heuristic says fraud, ML says no
        heur_extra = heuristic_series.index[(heuristic_series == 1) & (ml_pred.reindex(heuristic_series.index) == 0)]
        
        print(f"\n  ML catches but heuristic misses: {len(ml_extra)} cases")
        if len(ml_extra) > 0:
            for tid in ml_extra[:10]:
                row = test_df_engineered[test_df_engineered['transaction_id'] == tid].iloc[0]
                print(f"    ID={tid}: amount={row['amount']:.2f}, hour={row['transaction_hour']}, "
                      f"risk_flags={row['risk_flags']}")

        print(f"  Heuristic catches but ML misses: {len(heur_extra)} cases")
        if len(heur_extra) > 0:
            for tid in heur_extra[:10]:
                row = test_df_engineered[test_df_engineered['transaction_id'] == tid].iloc[0]
                print(f"    ID={tid}: amount={row['amount']:.2f}, hour={row['transaction_hour']}, "
                      f"risk_flags={row['risk_flags']}")
