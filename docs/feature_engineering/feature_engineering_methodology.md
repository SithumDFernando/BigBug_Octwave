# Feature Engineering Methodology
*OctWave 3.0 Credit Card Fraud Detection*

## Overview
This document outlines the feature engineering steps applied to the dataset in `src/data_processing/preprocess.py`. The goal of these transformations is to extract stronger signals from the raw data, allowing models to better distinguish between legitimate and fraudulent transactions.

## Engineered Features

### 1. Time-Based Risk Features
* **`is_night_transaction` (Binary)**: Flags transactions occurring between 00:00 and 05:00. Nighttime transactions have a significantly higher fraud rate (4.91% vs. 1.51% baseline).
* **`time_of_day_category` (Categorical, One-Hot Encoded)**: Bins `transaction_hour` into `Night` (0-5), `Morning` (6-11), `Afternoon` (12-17), and `Evening` (18-23).

### 2. Trust and Anomaly Ratios
* **`amount_to_trust_ratio` (Continuous)**: Calculated as `amount / device_trust_score`. Fraudsters often attempt high-value transactions on devices with low trust scores.
* **`amount_velocity_ratio` (Continuous)**: Calculated as `amount / (velocity_last_24h + 1)`. Represents the average spend per recent transaction. High velocity coupled with large amounts is a classic indicator of account takeover.

### 3. High-Risk Location Flags
* **`is_high_risk_location` (Binary)**: Flags transactions where either `foreign_transaction` or `location_mismatch` is true. Both features individually correspond to an ~8% fraud rate.
* **`location_anomaly_score` (Integer)**: The sum of the `foreign_transaction` and `location_mismatch` flags (range 0 to 2), indicating the severity of location-based anomalies.

### 4. Strategic Binning
* **`is_high_amount` (Binary)**: Flags transactions where `amount > 500`. The fraud rate doubles for transactions above this threshold.
* **`is_low_trust` (Binary)**: Flags transactions where `device_trust_score < 40`. The fraud rate is approximately 4x higher in this range.

### 5. Categorical Encoding
* **`merchant_category` (One-Hot Encoded)**: The 5 original string categories (`Food`, `Clothing`, `Travel`, `Electronics`, `Grocery`) are converted into binary indicator columns.

## Preserved Raw Data
To ensure tree-based models (like XGBoost and Random Forest) retain access to raw distributions, the original numerical columns (`amount`, `transaction_hour`, `device_trust_score`, `velocity_last_24h`, `cardholder_age`) are kept intact alongside their engineered counterparts.

## Output
The preprocessing script generates the following files in `data/processed/`:
*   `train_engineered.csv`: 24 features + `is_fraud` target.
*   `test_engineered.csv`: 23 features (for generating predictions).
*   `test_transaction_ids.csv`: Preserved transaction IDs for final submission formatting.
