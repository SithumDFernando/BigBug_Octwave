# OctWave 3.0 - Credit Card Fraud Detection Challenge: Technical Brief

## Competition Overview
The challenge focuses on solving a real-world machine learning classification problem by building a model to detect fraudulent credit card transactions (`is_fraud`).
This competition provides an opportunity to apply data preprocessing, feature engineering, model development, and evaluation techniques to solve a real-world financial fraud detection problem.

## Timeline
- **Starts:** 9th August 2026 – 12:00 NOON
- **Ends:** 11th August 2026 – 11:59 PM
All submissions must be made within this competition period.

## Dataset Details
The dataset contains simulated credit card transaction records designed for fraud detection research and machine learning experimentation. Each transaction contains information related to transaction behavior, merchant information, device reliability, and cardholder characteristics.

### Dataset Split & Statistics
- **Train Set (`train.csv`):** 8,000 samples
- **Test Set (`test.csv`):** 2,000 samples
- **Sample Submission (`sample_submission.csv`):** 2,000 rows
- **Class Distribution (Train):** 
  - `0` (Legitimate): 7,879 (98.5%)
  - `1` (Fraudulent): 121 (1.5%)

### Features
| Feature | Type / Expected Values | Description |
|---|---|---|
| `transaction_id` | Integer (`1` – `10000`) | Unique identifier assigned to each transaction |
| `amount` | Float (`0.00` – `1471.04`) | Monetary value of the transaction |
| `transaction_hour` | Integer (`0` – `23`) | Hour of the day when the transaction occurred |
| `merchant_category` | Categorical (`Clothing`, `Electronics`, `Food`, `Grocery`, `Travel`) | Category of the merchant involved in the transaction |
| `foreign_transaction` | Binary (`0`, `1`) | Indicates whether the transaction occurred in a foreign country |
| `location_mismatch` | Binary (`0`, `1`) | Indicates mismatch between transaction location and expected location |
| `device_trust_score` | Integer (`25` – `99`) | Trust score associated with the transaction device |
| `velocity_last_24h` | Integer (`0` – `9`) | Number of transactions performed within the previous 24 hours |
| `cardholder_age` | Integer (`18` – `69`) | Age of the cardholder |

### Target Variable
`is_fraud`
- `0` represents a legitimate transaction (98.5%)
- `1` represents a fraudulent transaction (1.5%)

*Important Note on Imbalance:* The dataset contains an imbalanced class distribution, where fraudulent transactions represent a smaller percentage of total transactions. Appropriate strategies for handling class imbalance (e.g. SMOTE, class weights, focal loss, threshold tuning) should be considered.

## Competition Task & Rules
Participants must use the provided training dataset to learn patterns associated with fraudulent transactions and predict fraud probabilities for unseen test transactions.

### Tasks
- Explore and analyze the provided dataset.
- Perform necessary data preprocessing and encoding.
- Develop a machine learning classification model.
- Generate predictions for the unseen test dataset.
- Submit predictions in the required format.

### Allowed Approaches
Participants may use any suitable machine learning techniques, including Logistic Regression, Decision Trees, Random Forest, Gradient Boosting Algorithms (XGBoost, LightGBM, CatBoost), Neural Networks, and Ensemble Learning Methods.
Participants are encouraged to experiment with feature engineering and model optimization techniques. Solutions should prioritize generalization performance rather than overfitting the training data.

### Constraints & Technical Rules
- **Daily Submissions:** Maximum of 10 submissions per day.
- **Final Evaluation:** Participants may select up to 2 submissions for the final Private Leaderboard scoring.
- **External Data:** The use of private datasets or external resources that provide an unfair advantage is strictly prohibited. The competition dataset is strictly for this event and must not be redistributed.

## Evaluation
Submissions are evaluated using the **F1-score** between the predicted fraud labels and the actual `is_fraud` values. 
F1-score considers the class imbalance in fraud detection.

$$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

- **Precision** measures the proportion of correctly identified fraud cases among all predicted fraud cases.
- **Recall** measures the proportion of actual fraud cases that were successfully detected.

## Deliverables

### 1. Kaggle Submission File (`submission.csv`)
For every transaction in the test dataset, participants must predict whether the transaction is fraudulent.
The submission file should contain two columns:
- `transaction_id` (Integer matching `test.csv`)
- `is_fraud` (Binary label: `0` or `1`)

### 2. Winner Verification Deliverables
Top-performing participants will be required to submit:
- **Reproducible Source Code & Environment:** Full pipeline code to reproduce `submission.csv` along with environment specifications (`requirements.txt`).
- **Methodology Write-up:** Brief summary of preprocessing, feature engineering, modeling approaches, and validation results.
- **Team Verification:** Proof of compliance with the official assigned team name.

## Proposed Repository Structure
```text
BigBug_Octwave/
├── data/
│   ├── raw/                 # train.csv, test.csv, sample_submission.csv
│   └── processed/           # Processed datasets / extracted features
├── docs/
│   ├── reference/           # technical_brief.md & administrative_rules.md
│   └── project_summary.md   # Full methodology & architecture overview
├── models/                  # Saved model artifacts (.pkl, .bin, etc.)
├── src/                     # Python source code
│   ├── data_processing/     # Data analysis and preprocessing scripts
│   ├── inference/           # Inference scripts for predicting on the test set
│   └── modeling/            # Model training and ensemble scripts
├── notebooks/               # Jupyter notebooks for EDA, Experimentation & Final Evaluation
├── outputs/                 # Final submission CSV files
├── README.md                # Environment setup & reproduction instructions
└── requirements.txt         # Package dependencies & exact versions
```
