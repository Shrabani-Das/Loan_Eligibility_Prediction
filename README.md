# Loan Eligibility Prediction System

## Objective
Developed an end-to-end predictive model to automate the credit risk assessment process by determining loan eligibility. Built a robust binary classification pipeline that minimizes manual underwriting efforts while mitigating financial default risks for lending institutions.

## Data Source
* **Dataset:** Loan Approval Dataset from Kaggle (containing applicant demographics, financial histories, credit scores, and loan properties).
* **Target Variable:** `Loan_Status` (Binary: Approved / Rejected)

## Steps Followed
1. **Exploratory Data Analysis (EDA):** Analyzed feature distributions, detected outliers using IQR, and assessed feature correlations to identify key drivers of creditworthiness.
2. **Data Preprocessing & Feature Engineering:**
   * Handled missing values using median (numerical) and mode (categorical) imputation to prevent data leakage.
   * Encoded categorical attributes using One-Hot Encoding and Label Encoding.
   * Scaled numerical features using `StandardScaler` to ensure convergence for distance-based algorithms.
3. **Model Implementation:** Implemented and benchmarked three distinct algorithms:
   * **Logistic Regression:** Established a linear baseline for interpretability.
   * **Random Forest Classifier:** Leveraged bagging to capture non-linear relationships and feature interactions.
   * **XGBoost:** Deployed gradient boosting for optimized performance and regularization.
4. **Model Evaluation:** Utilized stratified K-Fold cross-validation to evaluate models based on Precision, Recall, F1-Score, and ROC-AUC metrics to ensure robustness against class imbalance.

## Observations
* **Credit History Key Driver:** Credit history emerged as the most statistically significant feature determining loan approval rates.
* **Non-Linear Relationships:** Tree-based models (Random Forest and XGBoost) captured complex interactions between income-to-loan ratios better than the baseline Logistic Regression.
* **XGBoost Superiority:** XGBoost yielded the highest ROC-AUC score, effectively balancing the trade-off between False Positives (risky loans approved) and False Negatives (good applicants rejected).

## Conclusion
The project successfully demonstrates how machine learning can streamline credit risk classification. By deploying the  ML based models, financial institutions can automate a significant portion of the loan approval pipeline with high precision, maintaining low default risk while optimizing operational efficiency.

