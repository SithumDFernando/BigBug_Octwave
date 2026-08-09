import pandas as pd
import numpy as np
import os
import pickle
import optuna
import warnings
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score, accuracy_score
from sklearn.ensemble import VotingClassifier

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

def load_data():
    base_dir = r"c:\Users\ADMIN\Desktop\BigBug_Octwave"
    processed_dir = os.path.join(base_dir, "data", "processed")
    train = pd.read_csv(os.path.join(processed_dir, "train_engineered.csv"))
    test = pd.read_csv(os.path.join(processed_dir, "test_engineered.csv"))
    test_ids = pd.read_csv(os.path.join(processed_dir, "test_transaction_ids.csv"))
    return train, test, test_ids, base_dir

def load_models(base_dir):
    models = {}
    for name in ['adaboost', 'xgboost', 'catboost']:
        path = os.path.join(base_dir, "models", name, "best_model.pkl")
        with open(path, 'rb') as f:
            models[name] = pickle.load(f)
    return models

def get_oof_predictions(models, X, y):
    """Generate Out-Of-Fold probability predictions for each model to quickly tune weights."""
    print("Generating Out-Of-Fold probabilities (This might take a minute)...")
    oof_probs = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        print(f"  Generating OOF for {name}...")
        # Get probability of class 1
        probs = cross_val_predict(model, X, y, cv=cv, method='predict_proba', n_jobs=-1)[:, 1]
        oof_probs[name] = probs
    return oof_probs

def optimize_weights(oof_probs, y):
    print("Optimizing Ensemble weights with Optuna...")
    
    def objective(trial):
        w_ada = trial.suggest_float('w_ada', 0.0, 1.0)
        w_xgb = trial.suggest_float('w_xgb', 0.0, 1.0)
        w_cat = trial.suggest_float('w_cat', 0.0, 1.0)
        
        total = w_ada + w_xgb + w_cat
        if total == 0:
            return 0.0
        
        w_ada /= total
        w_xgb /= total
        w_cat /= total
        
        blended_probs = (w_ada * oof_probs['adaboost'] + 
                         w_xgb * oof_probs['xgboost'] + 
                         w_cat * oof_probs['catboost'])
        
        preds = (blended_probs >= 0.5).astype(int)
        return f1_score(y, preds)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    
    best_weights = study.best_params
    total = sum(best_weights.values())
    final_weights = [best_weights['w_ada']/total, best_weights['w_xgb']/total, best_weights['w_cat']/total]
    print(f"Optimal Weights -> AdaBoost: {final_weights[0]:.3f}, XGBoost: {final_weights[1]:.3f}, CatBoost: {final_weights[2]:.3f}")
    
    # Calculate final OOF metrics
    blended_probs = (final_weights[0] * oof_probs['adaboost'] + 
                     final_weights[1] * oof_probs['xgboost'] + 
                     final_weights[2] * oof_probs['catboost'])
    preds = (blended_probs >= 0.5).astype(int)
    
    metrics = {
        'F1_Score': f1_score(y, preds),
        'Precision': precision_score(y, preds),
        'Recall': recall_score(y, preds),
        'PR_AUC': average_precision_score(y, blended_probs),
        'Accuracy': accuracy_score(y, preds)
    }
    
    return final_weights, metrics

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
    oof_probs = get_oof_predictions(models, X, y)
    
    best_weights, ensemble_metrics = optimize_weights(oof_probs, y)
    print(f"Ensemble CV F1-Score: {ensemble_metrics['F1_Score']:.4f}")
    
    # Build VotingClassifier
    estimators = [('adaboost', models['adaboost']), 
                  ('xgboost', models['xgboost']), 
                  ('catboost', models['catboost'])]
                  
    ensemble_model = VotingClassifier(estimators=estimators, voting='soft', weights=best_weights)
    print("Training Final Ensemble Model on full dataset...")
    ensemble_model.fit(X, y)
    
    # Save Ensemble Artifacts
    ens_models_dir = os.path.join(base_dir, "models", "ensemble")
    ens_outputs_dir = os.path.join(base_dir, "outputs", "ensemble")
    os.makedirs(ens_models_dir, exist_ok=True)
    os.makedirs(ens_outputs_dir, exist_ok=True)
    
    model_path = os.path.join(ens_models_dir, "best_model.pkl")
    preds_path = os.path.join(ens_outputs_dir, "predictions.csv")
    
    with open(model_path, 'wb') as f:
        pickle.dump(ensemble_model, f)
        
    print("Predicting on test set...")
    ensemble_preds = ensemble_model.predict(X_test)
    ens_sub = pd.DataFrame({'transaction_id': test_ids['transaction_id'], 'is_fraud': ensemble_preds})
    ens_sub.to_csv(preds_path, index=False)
    
    # Log Ensemble
    results = {
        'Model_Name': 'Ensemble_SoftVoting',
        'CV_Accuracy': ensemble_metrics['Accuracy'],
        'CV_F1_Score': ensemble_metrics['F1_Score'],
        'CV_PR_AUC': ensemble_metrics['PR_AUC'],
        'CV_Precision': ensemble_metrics['Precision'],
        'CV_Recall': ensemble_metrics['Recall'],
        'Best_Parameters': {'w_ada': best_weights[0], 'w_xgb': best_weights[1], 'w_cat': best_weights[2]},
        'Model_File_Path': "models/ensemble/best_model.pkl",
        'Output_Predictions_Path': "outputs/ensemble/predictions.csv"
    }
    
    log_path = os.path.join(base_dir, "outputs", "model_results_log.csv")
    log_result(log_path, results)
    
    # Select Best Model for Final Submission
    best_single_f1 = 0.9957 # AdaBoost
    best_single_preds_path = os.path.join(base_dir, "outputs", "adaboost", "predictions.csv")
    final_sub_path = os.path.join(base_dir, "outputs", "FINAL_SUBMISSION.csv")
    
    print("\n--- Final Model Selection ---")
    if ensemble_metrics['F1_Score'] > best_single_f1:
        print("The Ensemble outperformed the best single model!")
        print("Using Ensemble predictions for FINAL_SUBMISSION.csv")
        ens_sub.to_csv(final_sub_path, index=False)
    else:
        print("The best single model (AdaBoost) outperformed or matched the Ensemble.")
        print("Using AdaBoost predictions for FINAL_SUBMISSION.csv")
        best_preds = pd.read_csv(best_single_preds_path)
        best_preds.to_csv(final_sub_path, index=False)
        
    print(f"Successfully created final submission file at: {final_sub_path}")

if __name__ == "__main__":
    main()
