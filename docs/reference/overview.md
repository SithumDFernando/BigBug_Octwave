# Overview

## OctWave 3.0 - Credit Card Fraud Detection Challenge

### Goal
The goal of this competition is to develop a machine learning model capable of detecting fraudulent credit card transactions.

Participants are required to build a binary classification model that can accurately distinguish between legitimate and fraudulent transactions using the provided dataset.

This competition provides an opportunity to apply data preprocessing, feature engineering, model development, and evaluation techniques to solve a real-world financial fraud detection problem.

### Description

#### Problem Description
Financial fraud detection is a critical challenge in modern banking and payment systems. With the increasing volume of digital transactions, developing intelligent systems to identify fraudulent activities has become essential.

In this competition, participants will develop machine learning models to predict whether a given credit card transaction is fraudulent or legitimate.

The provided dataset contains simulated transaction records representing realistic financial activities. Participants must analyze the transaction patterns, select appropriate features, and develop predictive models.

### Dataset
The dataset contains simulated credit card transaction records designed for fraud detection research and machine learning experimentation.

Each transaction contains information related to transaction behavior, merchant information, device reliability, and cardholder characteristics.

The available features include:

| Feature | Description |
|---|---|
| `transaction_id` | Unique identifier assigned to each transaction |
| `amount` | Monetary value of the transaction |
| `transaction_hour` | Hour of the day when the transaction occurred |
| `merchant_category` | Category of the merchant involved in the transaction |
| `foreign_transaction` | Indicates whether the transaction occurred in a foreign country |
| `location_mismatch` | Indicates mismatch between transaction location and expected location |
| `device_trust_score` | Trust score associated with the transaction device |
| `velocity_last_24h` | Number of transactions performed within the previous 24 hours |
| `cardholder_age` | Age of the cardholder |

The target variable is:

`is_fraud`

where:
- `0` represents a legitimate transaction
- `1` represents a fraudulent transaction

Participants must use the provided training dataset to learn patterns associated with fraudulent transactions and predict fraud probabilities for unseen test transactions.

### Competition Task
Participants must:

- Explore and analyze the provided dataset.
- Perform necessary data preprocessing.
- Develop a machine learning classification model.
- Generate predictions for the unseen test dataset.
- Submit predictions in the required format.

### Allowed Approaches
Participants may use any suitable machine learning techniques, including:

- Logistic Regression
- Decision Trees
- Random Forest
- Gradient Boosting Algorithms
- Neural Networks
- Ensemble Learning Methods

Participants are encouraged to experiment with feature engineering and model optimization techniques.

### Important Notes
- The dataset contains an imbalanced class distribution, where fraudulent transactions represent a smaller percentage of total transactions.
- Participants should consider appropriate strategies for handling class imbalance.
- Solutions should prioritize generalization performance rather than overfitting the training data.

### Evaluation
Submissions are evaluated using the **F1-score** between the predicted fraud labels and the actual `is_fraud` values.

The F1-score is selected because fraud detection datasets usually contain significantly fewer fraudulent transactions compared to legitimate transactions. Accuracy alone may not represent model performance effectively.

The F1-score is calculated as:

$$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

where:
- **Precision** measures the proportion of correctly identified fraud cases among all predicted fraud cases.
- **Recall** measures the proportion of actual fraud cases that were successfully detected.

### Submission File
For every transaction in the test dataset, participants must predict whether the transaction is fraudulent.

The submission file should contain two columns:
- `transaction_id`
- `is_fraud`
