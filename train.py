"""
Training pipeline for fraud detection.
- Validates data
- Engineers features
- Runs 5-fold stratified CV with detailed metrics
- Trains final model on full data
- Saves model to disk
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

import config
from src.data_validation import validate_data
from src.features import engineer_features, get_feature_columns


def train():
    print("=" * 60)
    print("FRAUD DETECTION TRAINING PIPELINE")
    print("=" * 60)

    # ── Step 1: Load data ───────────────────────────────────────────────
    print("\n[1/5] Loading data...")
    train_df = pd.read_csv(config.TRAIN_PATH)
    test_df = pd.read_csv(config.TEST_PATH)
    sample_sub = pd.read_csv(config.SAMPLE_SUB_PATH)
    print(f"  Train: {train_df.shape}, Test: {test_df.shape}")

    # ── Step 2: Validate data ──────────────────────────────────────────
    print("\n[2/5] Validating data...")
    valid = validate_data(train_df, test_df, sample_sub)
    if not valid:
        print("\n[ERROR] Data validation failed. Aborting.")
        return
    print("\n  All validation checks passed!")

    # ── Step 3: Feature engineering ─────────────────────────────────────
    print("\n[3/5] Engineering features...")
    train_eng = engineer_features(train_df, fit_encoder=True)
    feature_cols = get_feature_columns()
    X = train_eng[feature_cols]
    y = train_eng[config.TARGET]
    print(f"  Features: {len(feature_cols)} columns")
    print(f"  Target: {y.sum()} fraud / {len(y)} total ({y.mean()*100:.2f}%)")

    # ── Step 4: Cross-validation ────────────────────────────────────────
    print("\n[4/5] Running 5-fold stratified cross-validation...")
    print("-" * 60)

    skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)
    
    f1_scores = []
    prec_scores = []
    rec_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = GradientBoostingClassifier(**config.GB_PARAMS)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        f1 = f1_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred)

        f1_scores.append(f1)
        prec_scores.append(prec)
        rec_scores.append(rec)

        cm = confusion_matrix(y_val, y_pred)
        tn, fp, fn, tp = cm.ravel()

        status = "PERFECT" if f1 >= 0.999 else ("OK" if f1 >= 0.90 else "WARNING")
        print(f"  Fold {fold}: F1={f1:.4f}  P={prec:.4f}  R={rec:.4f}  "
              f"TP={tp} FP={fp} FN={fn} TN={tn}  [{status}]")

    print("-" * 60)
    print(f"  MEAN:   F1={np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}  "
          f"P={np.mean(prec_scores):.4f}  R={np.mean(rec_scores):.4f}")
    print(f"  Perfect folds: {sum(1 for f in f1_scores if f >= 0.999)}/{config.N_FOLDS}")

    # Check guardrails
    if np.mean(f1_scores) < 0.95:
        print("\n  [WARNING] Mean F1 < 0.95 — consider reviewing features/model!")
    if any(f < 0.90 for f in f1_scores):
        print(f"\n  [WARNING] {sum(1 for f in f1_scores if f < 0.90)} fold(s) below F1=0.90!")

    # ── Step 5: Train final model on full data ──────────────────────────
    print("\n[5/5] Training final model on full dataset...")
    final_model = GradientBoostingClassifier(**config.GB_PARAMS)
    final_model.fit(X, y)

    # Feature importances
    importances = pd.Series(final_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\n  Top 10 feature importances:")
    for feat, imp in importances.head(10).items():
        bar = "=" * int(imp * 50)
        print(f"    {feat:30s} {imp:.4f}  {bar}")

    # Save model
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    model_path = os.path.join(config.MODEL_DIR, 'gradient_boosting.joblib')
    joblib.dump(final_model, model_path)
    print(f"\n  Model saved to: {model_path}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"  CV F1: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    print("=" * 60)

    return final_model


if __name__ == '__main__':
    train()
