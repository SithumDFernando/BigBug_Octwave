import pandas as pd
import numpy as np
import os
import json
import pickle
import warnings
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import optuna

# Models
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

def load_data():
    base_dir = r"c:\Users\ADMIN\Desktop\BigBug_Octwave"
    processed_dir = os.path.join(base_dir, "data", "processed")
    train = pd.read_csv(os.path.join(processed_dir, "train_engineered.csv"))
    test = pd.read_csv(os.path.join(processed_dir, "test_engineered.csv"))
    test_ids = pd.read_csv(os.path.join(processed_dir, "test_transaction_ids.csv"))
    return train, test, test_ids, base_dir

def init_logger(outputs_dir):
    log_path = os.path.join(outputs_dir, "model_results_log.csv")
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write("Model_Name,CV_Accuracy,CV_F1_Score,CV_PR_AUC,CV_Precision,CV_Recall,Best_Parameters,Model_File_Path,Output_Predictions_Path\n")
    return log_path

def log_result(log_path, results):
    row = f"{results['Model_Name']},{results['CV_Accuracy']:.4f},{results['CV_F1_Score']:.4f},{results['CV_PR_AUC']:.4f}," \
          f"{results['CV_Precision']:.4f},{results['CV_Recall']:.4f},\"{str(results['Best_Parameters']).replace('\"','\'')}\"," \
          f"{results['Model_File_Path']},{results['Output_Predictions_Path']}\n"
    with open(log_path, 'a') as f:
        f.write(row)

# Define Model Objectives for Optuna
def get_objectives(X, y):
    pos_weight = (len(y) - sum(y)) / sum(y) # scale_pos_weight
    
    def get_cv_score(model, X, y, needs_scaling=False):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        if needs_scaling:
            model = Pipeline([('scaler', StandardScaler()), ('clf', model)])
        
        scores = cross_validate(model, X, y, cv=cv, scoring=['f1', 'average_precision', 'accuracy', 'precision', 'recall'], n_jobs=-1, error_score=0.0)
        return scores['test_f1'].mean()

    objectives = {
        'LogisticRegression': lambda trial: get_cv_score(
            LogisticRegression(
                C=trial.suggest_float('C', 1e-4, 10.0, log=True),
                class_weight='balanced',
                random_state=42,
                max_iter=1000
            ), X, y, needs_scaling=True),
            
        'RandomForest': lambda trial: get_cv_score(
            RandomForestClassifier(
                n_estimators=trial.suggest_int('n_estimators', 50, 200),
                max_depth=trial.suggest_int('max_depth', 3, 10),
                class_weight='balanced',
                random_state=42,
                n_jobs=1
            ), X, y, needs_scaling=False),
            
        'ExtraTrees': lambda trial: get_cv_score(
            ExtraTreesClassifier(
                n_estimators=trial.suggest_int('n_estimators', 50, 200),
                max_depth=trial.suggest_int('max_depth', 3, 15),
                class_weight='balanced',
                random_state=42,
                n_jobs=1
            ), X, y, needs_scaling=False),
            
        'XGBoost': lambda trial: get_cv_score(
            XGBClassifier(
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                max_depth=trial.suggest_int('max_depth', 3, 9),
                scale_pos_weight=pos_weight,
                eval_metric='logloss',
                random_state=42,
                n_jobs=1
            ), X, y, needs_scaling=False),
            
        'LightGBM': lambda trial: get_cv_score(
            LGBMClassifier(
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                max_depth=trial.suggest_int('max_depth', 3, 9),
                scale_pos_weight=pos_weight,
                random_state=42,
                n_jobs=1,
                verbose=-1
            ), X, y, needs_scaling=False),
            
        'CatBoost': lambda trial: get_cv_score(
            CatBoostClassifier(
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                depth=trial.suggest_int('depth', 4, 8),
                auto_class_weights='Balanced',
                random_state=42,
                verbose=0,
                thread_count=1
            ), X, y, needs_scaling=False),
            
        'AdaBoost': lambda trial: get_cv_score(
            AdaBoostClassifier(
                n_estimators=trial.suggest_int('n_estimators', 50, 200),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 1.0, log=True),
                random_state=42
            ), X, y, needs_scaling=False),
            
        'GradientBoosting': lambda trial: get_cv_score(
            GradientBoostingClassifier(
                n_estimators=trial.suggest_int('n_estimators', 50, 200),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                max_depth=trial.suggest_int('max_depth', 3, 8),
                random_state=42
            ), X, y, needs_scaling=False),
            
        'SVC': lambda trial: get_cv_score(
            SVC(
                C=trial.suggest_float('C', 0.1, 10.0, log=True),
                class_weight='balanced',
                probability=True,
                random_state=42
            ), X, y, needs_scaling=True),
            
        'KNN': lambda trial: get_cv_score(
            KNeighborsClassifier(
                n_neighbors=trial.suggest_int('n_neighbors', 3, 15),
                weights=trial.suggest_categorical('weights', ['uniform', 'distance'])
            ), X, y, needs_scaling=True),
            
        'GaussianNB': lambda trial: get_cv_score(
            GaussianNB(
                var_smoothing=trial.suggest_float('var_smoothing', 1e-9, 1e-2, log=True)
            ), X, y, needs_scaling=True),
            
        'MLPClassifier': lambda trial: get_cv_score(
            MLPClassifier(
                hidden_layer_sizes=trial.suggest_categorical('hidden_layer_sizes', [(50,), (100,), (50, 50)]),
                alpha=trial.suggest_float('alpha', 1e-4, 1e-1, log=True),
                random_state=42,
                max_iter=500
            ), X, y, needs_scaling=True)
    }
    return objectives, pos_weight

def get_best_model_instance(model_name, best_params, pos_weight):
    needs_scaling = model_name in ['LogisticRegression', 'SVC', 'KNN', 'GaussianNB', 'MLPClassifier']
    
    if model_name == 'LogisticRegression': model = LogisticRegression(**best_params, class_weight='balanced', random_state=42, max_iter=1000)
    elif model_name == 'RandomForest': model = RandomForestClassifier(**best_params, class_weight='balanced', random_state=42)
    elif model_name == 'ExtraTrees': model = ExtraTreesClassifier(**best_params, class_weight='balanced', random_state=42)
    elif model_name == 'XGBoost': model = XGBClassifier(**best_params, scale_pos_weight=pos_weight, eval_metric='logloss', random_state=42)
    elif model_name == 'LightGBM': model = LGBMClassifier(**best_params, scale_pos_weight=pos_weight, random_state=42, verbose=-1)
    elif model_name == 'CatBoost': model = CatBoostClassifier(**best_params, auto_class_weights='Balanced', random_state=42, verbose=0)
    elif model_name == 'AdaBoost': model = AdaBoostClassifier(**best_params, random_state=42)
    elif model_name == 'GradientBoosting': model = GradientBoostingClassifier(**best_params, random_state=42)
    elif model_name == 'SVC': model = SVC(**best_params, class_weight='balanced', probability=True, random_state=42)
    elif model_name == 'KNN': model = KNeighborsClassifier(**best_params)
    elif model_name == 'GaussianNB': model = GaussianNB(**best_params)
    elif model_name == 'MLPClassifier': model = MLPClassifier(**best_params, random_state=42, max_iter=500)
    
    if needs_scaling:
        model = Pipeline([('scaler', StandardScaler()), ('clf', model)])
    
    return model

def train_and_log_models():
    print("Starting Exhaustive Modeling Phase...")
    train, test, test_ids, base_dir = load_data()
    
    X = train.drop(columns=['is_fraud'])
    y = train['is_fraud']
    X_test = test
    
    models_dir = os.path.join(base_dir, "models")
    outputs_dir = os.path.join(base_dir, "outputs")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    
    log_path = init_logger(outputs_dir)
    objectives, pos_weight = get_objectives(X, y)
    
    N_TRIALS = 20
    
    for model_name, objective_func in objectives.items():
        print(f"\n--- Optimizing {model_name} ---")
        
        # Create nested dirs
        m_dir = os.path.join(models_dir, model_name.lower())
        o_dir = os.path.join(outputs_dir, model_name.lower())
        os.makedirs(m_dir, exist_ok=True)
        os.makedirs(o_dir, exist_ok=True)
        
        model_file = os.path.join(m_dir, "best_model.pkl")
        preds_file = os.path.join(o_dir, "predictions.csv")
        
        # Optuna Study
        study = optuna.create_study(direction="maximize")
        study.optimize(objective_func, n_trials=N_TRIALS, timeout=600) # max 10 mins per model
        
        best_params = study.best_params
        print(f"Best Params: {best_params}")
        
        # Final evaluation of best params
        final_model = get_best_model_instance(model_name, best_params, pos_weight)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_validate(final_model, X, y, cv=cv, scoring=['f1', 'average_precision', 'accuracy', 'precision', 'recall'], n_jobs=-1)
        
        # Train on full data & save
        final_model.fit(X, y)
        with open(model_file, 'wb') as f:
            pickle.dump(final_model, f)
            
        # Predict on test
        preds = final_model.predict(X_test)
        sub = pd.DataFrame({'transaction_id': test_ids['transaction_id'], 'is_fraud': preds})
        sub.to_csv(preds_file, index=False)
        
        # Log
        results = {
            'Model_Name': model_name,
            'CV_Accuracy': scores['test_accuracy'].mean(),
            'CV_F1_Score': scores['test_f1'].mean(),
            'CV_PR_AUC': scores['test_average_precision'].mean(),
            'CV_Precision': scores['test_precision'].mean(),
            'CV_Recall': scores['test_recall'].mean(),
            'Best_Parameters': best_params,
            'Model_File_Path': f"models/{model_name.lower()}/best_model.pkl",
            'Output_Predictions_Path': f"outputs/{model_name.lower()}/predictions.csv"
        }
        log_result(log_path, results)
        print(f"Finished {model_name}. F1: {results['CV_F1_Score']:.4f}")

if __name__ == "__main__":
    train_and_log_models()
