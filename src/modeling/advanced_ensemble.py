import pandas as pd
import numpy as np
import os
import pickle
import warnings
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score, accuracy_score
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')

def load_data():
    base_dir = r"c:\Users\ADMIN\Desktop\BigBug_Octwave"
    processed_dir = os.path.join(base_dir, "data", "processed")
    train = pd.read_csv(os.path.join(processed_dir, "train_engineered.csv"))
    test = pd.read_csv(os.path.join(processed_dir, "test_engineered.csv"))
    test_ids = pd.read_csv(os.path.join(processed_dir, "test_transaction_ids.csv"))
    return train, test, test_ids, base_dir

def load_models(base_dir):
    models = {}
    top_models = ['adaboost', 'xgboost', 'catboost', 'gradientboosting', 'lightgbm']
    for name in top_models:
        path = os.path.join(base_dir, "models", name, "best_model.pkl")
        with open(path, 'rb') as f:
            models[name] = pickle.load(f)
    return models

def get_oof_meta_features(models, X, y):
    print("Generating OOF Meta-Features (This might take a minute)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    meta_features = np.zeros((X.shape[0], len(models)))
    
    for i, (name, model) in enumerate(models.items()):
        print(f"  Generating OOF for {name}...")
        probs = cross_val_predict(model, X, y, cv=cv, method='predict_proba', n_jobs=-1)[:, 1]
        meta_features[:, i] = probs
        
    return meta_features

def optimize_threshold(y_true, y_probs):
    print("Optimizing Classification Threshold...")
    thresholds = np.arange(0.01, 1.0, 0.01)
    best_thresh = 0.5
    best_f1 = 0.0
    
    for thresh in thresholds:
        preds = (y_probs >= thresh).astype(int)
        score = f1_score(y_true, preds)
        if score > best_f1:
            best_f1 = score
            best_thresh = thresh
            
    print(f"Optimal Threshold Found: {best_thresh:.2f} (F1: {best_f1:.4f})")
    return best_thresh

def log_result(log_path, results):
    row = f"{results['Model_Name']},{results['CV_Accuracy']:.4f},{results['CV_F1_Score']:.4f},{results['CV_PR_AUC']:.4f}," \
          f"{results['CV_Precision']:.4f},{results['CV_Recall']:.4f},\"{str(results['Best_Parameters']).replace('\"','\'')}\"," \
          f"{results['Model_File_Path']},{results['Output_Predictions_Path']}\n"
    with open(log_path, 'a') as f:
        f.write(row)

def main():
    train, test, test_ids, base_dir = load_data()
    X = train.drop(columns=['is_fraud'])
    y = train['is_fraud']
    X_test = test
    
    models = load_models(base_dir)
    
    # Generate Level-0 OOF Features
    X_meta = get_oof_meta_features(models, X, y)
    
    # Train Level-1 Meta-Learner on OOF features (L2 Regularized)
    print("Training Level-1 Meta-Learner (Ridge Logistic Regression)...")
    meta_model = LogisticRegression(penalty='l2', C=1.0, random_state=42)
    
    # Get OOF probabilities from Meta-Learner to evaluate ensemble performance
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    meta_oof_probs = cross_val_predict(meta_model, X_meta, y, cv=cv, method='predict_proba')[:, 1]
    
    # Dynamic Threshold Optimization
    best_thresh = optimize_threshold(y, meta_oof_probs)
    
    # Calculate Final Metrics with Best Threshold
    final_cv_preds = (meta_oof_probs >= best_thresh).astype(int)
    metrics = {
        'F1_Score': f1_score(y, final_cv_preds),
        'Precision': precision_score(y, final_cv_preds),
        'Recall': recall_score(y, final_cv_preds),
        'PR_AUC': average_precision_score(y, meta_oof_probs),
        'Accuracy': accuracy_score(y, final_cv_preds)
    }
    print(f"\nAdvanced Stacking Ensemble CV F1-Score: {metrics['F1_Score']:.4f}")
    
    # Train Final Meta-Model on full Level-0 features
    meta_model.fit(X_meta, y)
    
    # Save Advanced Ensemble Artifacts
    ens_models_dir = os.path.join(base_dir, "models", "advanced_ensemble")
    ens_outputs_dir = os.path.join(base_dir, "outputs", "advanced_ensemble")
    os.makedirs(ens_models_dir, exist_ok=True)
    os.makedirs(ens_outputs_dir, exist_ok=True)
    
    model_path = os.path.join(ens_models_dir, "best_model.pkl")
    preds_path = os.path.join(ens_outputs_dir, "predictions.csv")
    
    # Save the meta-learner and the threshold info
    ensemble_package = {
        'base_models': models,
        'meta_model': meta_model,
        'optimal_threshold': best_thresh
    }
    with open(model_path, 'wb') as f:
        pickle.dump(ensemble_package, f)
        
    print("Generating Final Test Predictions...")
    # Generate test features for Meta-Learner
    X_test_meta = np.zeros((X_test.shape[0], len(models)))
    for i, (name, model) in enumerate(models.items()):
        X_test_meta[:, i] = model.predict_proba(X_test)[:, 1]
        
    # Predict using meta-learner and custom threshold
    test_probs = meta_model.predict_proba(X_test_meta)[:, 1]
    final_test_preds = (test_probs >= best_thresh).astype(int)
    
    ens_sub = pd.DataFrame({'transaction_id': test_ids['transaction_id'], 'is_fraud': final_test_preds})
    ens_sub.to_csv(preds_path, index=False)
    
    # Log Advanced Ensemble
    results = {
        'Model_Name': 'Advanced_Stacking_Ensemble',
        'CV_Accuracy': metrics['Accuracy'],
        'CV_F1_Score': metrics['F1_Score'],
        'CV_PR_AUC': metrics['PR_AUC'],
        'CV_Precision': metrics['Precision'],
        'CV_Recall': metrics['Recall'],
        'Best_Parameters': {'meta_model': 'LogisticRegression_L2', 'optimal_threshold': best_thresh},
        'Model_File_Path': "models/advanced_ensemble/best_model.pkl",
        'Output_Predictions_Path': "outputs/advanced_ensemble/predictions.csv"
    }
    
    log_path = os.path.join(base_dir, "outputs", "model_results_log.csv")
    log_result(log_path, results)
    
    # Final Submission Selection
    best_single_f1 = 0.9959 # Previous Soft-Voting Ensemble
    final_sub_path = os.path.join(base_dir, "outputs", "FINAL_SUBMISSION.csv")
    
    print("\n--- Final Model Selection ---")
    if metrics['F1_Score'] > best_single_f1:
        print("The Advanced Stacking Ensemble outperformed the basic Soft-Voting Ensemble!")
        print("Overwriting FINAL_SUBMISSION.csv with new advanced predictions.")
        ens_sub.to_csv(final_sub_path, index=False)
    else:
        print("The basic Soft-Voting Ensemble matched or outperformed the Stacking Meta-Learner.")
        print("Keeping the previous FINAL_SUBMISSION.csv intact.")
        
    print("Advanced Execution Complete!")

if __name__ == "__main__":
    main()
