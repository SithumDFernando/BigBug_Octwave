# 📚 Project Documentation Index

Welcome to the documentation repository for **Team BigBug's** solution to the **OctWave 3.0 Credit Card Fraud Detection Challenge**.

---

## 🗂️ Documentation Structure

```text
docs/
├── README.md                          # Documentation index (this file)
├── project_summary.md                 # 🏆 Main project summary & full methodology (with Mermaid diagrams)
├── data_cleaning/                     # Data cleaning specifications
│   └── data_cleaning_plan.md          # Strategy and validation for raw data cleaning
├── feature_engineering/               # Feature engineering methodology
│   └── feature_engineering_methodology.md  # 16 engineered features rationale & implementations
├── modeling/                          # Machine learning modeling documentation
│   ├── modeling_plan.md              # Industrial-level 12-model evaluation strategy
│   └── modeling_results.md           # Model Leaderboard & cross-validation analysis
├── reports/                           # Technical and evaluation reports
│   ├── project_evaluation_report.md  # Evaluation analysis & F1 strategy
│   └── overfitting_analysis_report.md# Public vs private leaderboard overfitting analysis
└── reference/                         # Official competition specifications
    ├── technical_brief.md            # OctWave 3.0 technical requirements & dataset details
    └── administrative_rules.md       # Official competition rules & tiebreakers
```

---

## 📖 Quick Links & Summary

### 🌟 Core Summary & Methodology
* **[Project Summary & Methodology](project_summary.md)**: The primary, comprehensive write-up detailing the end-to-end architecture, EDA insights, feature engineering, model training, Optuna soft voting ensemble, and validation metrics ($F_1 = 0.9959$).

---

### ⚙️ Pipeline Specifications
* **[Data Cleaning Plan](data_cleaning/data_cleaning_plan.md)**: Details on raw data checks, handling data types, missing values, and data integrity.
* **[Feature Engineering Methodology](feature_engineering/feature_engineering_methodology.md)**: Complete breakdown of domain-driven features (`is_night_transaction`, `amount_to_trust_ratio`, `is_high_risk_location`, etc.).
* **[Modeling Strategy & Plan](modeling/modeling_plan.md)**: Architectural blueprint for 12 candidate algorithms, Stratified 5-Fold CV, and Optuna tuning.
* **[Modeling Execution Results](modeling/modeling_results.md)**: Cross-validation leaderboard of all models (AdaBoost, XGBoost, CatBoost, LightGBM, Random Forest, Neural Networks, etc.).

---

### 📊 Technical Reports
* **[Project Evaluation Report](reports/project_evaluation_report.md)**: Analysis of metric progression, class imbalance handling, and threshold optimization strategies.
* **[Overfitting & Edge Case Analysis](reports/overfitting_analysis_report.md)**: Risk mitigation for Public (30%) vs. Private (70%) leaderboard split and rule-based generation analysis.

---

### 📋 Official Competition Reference
* **[Technical Brief](reference/technical_brief.md)**: Official problem statement, evaluation metric ($F_1$), dataset specs, and deliverable guidelines.
* **[Administrative Rules](reference/administrative_rules.md)**: Official Kaggle and competition rules, submission limits, and tiebreaker conditions.
