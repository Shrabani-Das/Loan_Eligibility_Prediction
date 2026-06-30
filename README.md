# 🏦 Loan Eligibility Prediction

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-green.svg)](https://xgboost.readthedocs.io/)
[![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)]()

A supervised **binary classification** project that predicts whether a loan applicant is **eligible for loan approval** based on demographic, financial and credit history attributes. The project covers the complete machine learning lifecycle — exploratory data analysis, statistical diagnostics, class-imbalance handling, model benchmarking and feature interpretability.

---

## 📌 Project Overview

Financial institutions rely on consistent, data-driven decisioning to assess credit risk and reduce loan default rates. This project builds and benchmarks multiple classification models to predict `Loan_Status` (Approved / Rejected) for applicants, simulating a real-world **credit underwriting / loan eligibility screening** use case.

**Key objectives:**
- Perform structured EDA (univariate, bivariate, correlation analysis) on applicant data
- Diagnose and treat multicollinearity using **Variance Inflation Factor (VIF)**
- Handle class imbalance using **Random Oversampling**
- Train and benchmark six classification algorithms
- Validate model robustness using **K-Fold Cross-Validation**
- Interpret model decisions through **feature importance** analysis

---

## 🧠 Tech Stack

`Machine Learning` `Binary Classification` `Credit Risk` `Loan Default Prediction` `EDA` `Feature Engineering` `Multicollinearity (VIF)` `Class Imbalance` `Random Oversampling` `Logistic Regression` `Random Forest` `XGBoost` `Bagging Classifier` `Support Vector Machine (SVM)` `Decision Tree` `K-Fold Cross-Validation` `ROC-AUC` `Confusion Matrix` `Feature Importance` `Python` `Scikit-learn` `Pandas` `Seaborn`

---

## 🗂️ Repository Structure

```
loan-eligibility-prediction/
│
├── LOAN_ELIGIBILITY_PREDICTION.ipynb   # Main notebook (EDA + modelling pipeline)
├── Loan_Data.csv                       # Raw dataset
├── README.md                           # Project documentation
└── requirements.txt                    # Python dependencies
```

---

## 📊 Dataset Description

The dataset contains historical loan application records with the following features:

| Feature | Description |
|---|---|
| `Loan_ID` | Unique application identifier |
| `Gender`, `Married`, `Dependents`, `Education`, `Self_Employed` | Applicant demographic attributes |
| `ApplicantIncome`, `CoapplicantIncome` | Income-related financial attributes |
| `LoanAmount`, `Loan_Amount_Term` | Loan request attributes |
| `Credit_History` | Historical credit repayment behavior |
| `Property_Area` | Urban / Semiurban / Rural classification |
| `Loan_Status` | **Target variable** — Approved (Y) / Rejected (N) |

---

## 🔍 Methodology

### 1. Exploratory Data Analysis (EDA)
- Univariate analysis of numerical features (income, loan amount) using boxplots and distribution plots
- Univariate analysis of categorical features (gender, education, credit history, property area)
- Bivariate analysis using pair plots and correlation heatmaps (Pearson & Spearman)

### 2. Data Preprocessing
- Missing value imputation — **mode** for categorical features, **median** for numerical features (robust to outliers)
- Multicollinearity check using **VIF**; `LoanAmount` dropped due to high collinearity with `ApplicantIncome`
- **Random Oversampling** applied to address class imbalance in the target variable
- One-hot encoding of categorical variables and label encoding of the target
- Stratified 70/30 train-test split

### 3. Model Development
Six classification algorithms were trained and evaluated:

| Model | Type |
|---|---|
| Logistic Regression | Linear baseline classifier |
| Decision Tree | Non-linear, interpretable classifier |
| Random Forest | Bagging ensemble |
| Bagging Random Forest | Ensemble of ensembles |
| XGBoost | Gradient boosting ensemble |
| Support Vector Machine (Linear Kernel) | Margin-based classifier |

### 4. Model Evaluation
Each model was evaluated on:
- **Accuracy, Precision, Recall, F1-score** (train and test)
- **ROC-AUC score** and ROC curve
- **Confusion matrix** visualization
- **5-Fold Cross-Validation** for robustness checks
- **Feature Importance** ranking (Random Forest)

---

## 🏆 Results

| Model | Test Accuracy | Test Recall | Test Precision | ROC-AUC |
|---|---|---|---|---|
| **Random Forest** ⭐ | **87.01%** | 82.68% | 90.52% | **87.01%** |
| XGBoost | 83.86% | 75.59% | 90.57% | 83.86% |
| Bagging Random Forest | 81.89% | 75.59% | 86.49% | — |
| Decision Tree | 78.35% | 69.29% | 84.62% | — |
| Logistic Regression | 58.27% | 53.54% | 59.13% | — |
| SVM (Linear Kernel) | 54.33% | 39.37% | 56.18% | — |

Based on comparative evaluation across accuracy, recall, precision, and 5-fold cross-validation, **Random Forest** was selected as the best-performing model, achieving **87.01% test accuracy** and an **ROC-AUC of 0.870**, with **XGBoost** (83.86% accuracy, 0.839 ROC-AUC) as a close competitive alternative. Credit history emerged as the most influential predictor of loan approval.

---

## ⚙️ Installation & Usage

```bash
# Clone the repository
git clone https://github.com/<your-username>/loan-eligibility-prediction.git
cd loan-eligibility-prediction

# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook LOAN_ELIGIBILITY_PREDICTION.ipynb
```

### `requirements.txt`
```
numpy
pandas
seaborn
matplotlib
scikit-learn
xgboost
scipy
statsmodels
imbalanced-learn
jupyter
```

---

## 🚀 Future Enhancements
- Hyperparameter tuning via `GridSearchCV` / `RandomizedSearchCV`
- SHAP-based model explainability for individual predictions
- Deployment as a Streamlit / Flask web application
- Integration of WOE-IV based scorecard for regulatory-aligned credit scoring

---

## 👩‍💻 Author

**Shrabani Das**
Quantitative Credit Risk Analyst | MSc Quantitative Economics, Indian Statistical Institute

