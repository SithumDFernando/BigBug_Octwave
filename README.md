# OctWave 3.0 - Credit Card Fraud Detection Challenge

## Overview
This repository contains the solution for the OctWave 3.0 challenge. The goal is to build a machine learning model capable of detecting fraudulent credit card transactions from an imbalanced dataset.

## Repository Structure
- `data/raw/`: Original datasets (`train.csv`, `test.csv`, `sample_submission.csv`)
- `data/processed/`: Cleaned and engineered features
- `docs/reference/`: Technical brief and administrative rules
- `models/`: Saved model artifacts
- `notebooks/`: Exploratory Data Analysis (EDA) and experimental notebooks
- `outputs/`: Final prediction CSVs ready for submission
- `src/`: Python source code
  - `preprocess.py`: Data loading, cleaning, and feature engineering
  - `train.py`: Model training and cross-validation
  - `predict.py`: Inference on the test set

## Setup Instructions
1. Ensure Python 3.12+ is installed.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
