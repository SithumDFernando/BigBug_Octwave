"""
Prediction and submission generation pipeline.
- Loads trained model
- Engineers features on test data
- Generates predictions
- Validates submission
- Compares with heuristic
"""
import os
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

import config
from src.features import engineer_features, get_feature_columns
from src.checks import validate_submission, compare_with_heuristic


def predict():
    print("=" * 60)
    print("FRAUD DETECTION PREDICTION PIPELINE")
    print("=" * 60)

    # ── Step 1: Load model ──────────────────────────────────────────────
    print("\n[1/5] Loading trained model...")
    model_path = os.path.join(config.MODEL_DIR, 'gradient_boosting.joblib')
    if not os.path.exists(model_path):
        print(f"  [ERROR] Model not found at {model_path}. Run train.py first.")
        return
    model = joblib.load(model_path)
    print(f"  Loaded model from: {model_path}")

    # ── Step 2: Load and prepare test data ──────────────────────────────
    print("\n[2/5] Loading test data...")
    # Need to fit encoder on train first, then transform test
    train_df = pd.read_csv(config.TRAIN_PATH)
    test_df = pd.read_csv(config.TEST_PATH)
    sample_sub = pd.read_csv(config.SAMPLE_SUB_PATH)
    print(f"  Test shape: {test_df.shape}")

    # ── Step 3: Feature engineering ─────────────────────────────────────
    print("\n[3/5] Engineering features...")
    # Fit encoder on train data first
    _ = engineer_features(train_df, fit_encoder=True)
    # Then transform test data
    test_eng = engineer_features(test_df, fit_encoder=False)
    feature_cols = get_feature_columns()
    X_test = test_eng[feature_cols]
    print(f"  Features: {len(feature_cols)} columns")

    # ── Step 4: Generate predictions ────────────────────────────────────
    print("\n[4/5] Generating predictions...")
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    n_fraud = predictions.sum()
    n_legit = len(predictions) - n_fraud
    print(f"  Predicted: {n_fraud} fraud, {n_legit} legitimate")
    print(f"  Fraud rate: {n_fraud/len(predictions)*100:.2f}%")

    # Probability distribution
    print(f"\n  Probability distribution:")
    print(f"    Min:    {probabilities.min():.6f}")
    print(f"    Max:    {probabilities.max():.6f}")
    print(f"    Mean:   {probabilities.mean():.6f}")
    print(f"    Median: {np.median(probabilities):.6f}")
    print(f"    >0.5:   {(probabilities > 0.5).sum()}")
    print(f"    >0.9:   {(probabilities > 0.9).sum()}")

    # Create submission
    submission = pd.DataFrame({
        'transaction_id': test_df['transaction_id'],
        'is_fraud': predictions.astype(int)
    })

    # ── Step 5: Validate and save ───────────────────────────────────────
    print("\n[5/5] Validating submission...")
    valid = validate_submission(submission, sample_sub)

    # Heuristic comparison
    compare_with_heuristic(test_eng, submission)

    # Save submission
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)
    submission_path = os.path.join(config.SUBMISSION_DIR, 'submission.csv')
    submission.to_csv(submission_path, index=False)
    print(f"\n  Submission saved to: {submission_path}")

    # Also show a preview
    print(f"\n  Preview (first 10 rows):")
    print(submission.head(10).to_string(index=False))

    print(f"\n  Fraud cases in submission:")
    fraud_cases = submission[submission['is_fraud'] == 1]
    if len(fraud_cases) > 0:
        fraud_details = test_df[test_df['transaction_id'].isin(fraud_cases['transaction_id'])]
        print(fraud_details.to_string(index=False))

    print("\n" + "=" * 60)
    print("PREDICTION COMPLETE")
    print(f"  Submission: {submission_path}")
    print(f"  Rows: {len(submission)}, Fraud: {n_fraud}, Legit: {n_legit}")
    print("=" * 60)


if __name__ == '__main__':
    predict()
