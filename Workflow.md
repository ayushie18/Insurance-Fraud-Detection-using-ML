🚨 Insurance Fraud Detection Using Machine Learning
🔵 Epic 1: Define Problem / Problem Understanding
✅ Activity 1.1: Specify the Business Problem

Insurance companies suffer heavy financial losses due to fraudulent claims.

Problem Statement:

Develop a Machine Learning model that predicts whether an insurance claim is fraudulent or genuine.

This is a:

Binary Classification Problem

Output:

0 → Genuine Claim

1 → Fraudulent Claim

✅ Activity 1.2: Business Requirements

Reduce financial loss due to fraud.

Detect fraud early during claim submission.

Minimize false positives (don’t block genuine customers).

Improve operational efficiency.

Real-time prediction capability.

✅ Activity 1.3: Literature Survey

You can mention:

Logistic Regression used for baseline fraud detection.

Random Forest & XGBoost give better performance.

Class imbalance handled using:

SMOTE

Undersampling

Cost-sensitive learning

You can reference:

Kaggle fraud datasets

IEEE fraud detection papers

Research on imbalanced classification

✅ Activity 1.4: Social / Business Impact
Business Impact:

Saves millions in fraudulent payouts.

Improves trust in insurance system.

Social Impact:

Reduces insurance premium inflation.

Ensures fairness.

🔵 Epic 2: Data Collection & Preparation
✅ Activity 2.1: Collect Dataset

You can use:

Kaggle: Insurance Fraud Detection Dataset

Synthetic Insurance Claims Dataset

Dataset contains features like:

Age

Claim amount

Policy type

Incident type

Police report filed

Witness count

Previous claims

Fraud reported (target)

✅ Activity 2.2: Data Preparation

Steps:

Handle Missing Values

Encode Categorical Variables

Label Encoding

One Hot Encoding

Handle Imbalanced Data

SMOTE

Feature Scaling

StandardScaler

Train-Test Split

80% Training

20% Testing

🔵 Epic 3: Exploratory Data Analysis (EDA)
✅ Activity 3.1: Descriptive Statistics

Mean

Median

Standard Deviation

Correlation Matrix

Fraud percentage distribution

Example Insight:

Fraud claims are only 15% of total data → Imbalanced Dataset.

✅ Activity 3.2: Visual Analysis

Use:

Bar Charts (Fraud vs Non-Fraud)

Correlation Heatmap

Distribution Plot (Claim Amount)

Countplot (Incident Type vs Fraud)

🔵 Epic 4: Model Building

Train multiple models:

Logistic Regression

Decision Tree

Random Forest

XGBoost

Support Vector Machine

KNN

Why Multiple Algorithms?

Because:

Some handle imbalance better.

Some give higher recall.

Some give better precision.

🔵 Epic 5: Performance Testing & Hyperparameter Tuning
✅ Activity 5.1: Evaluation Metrics

Since this is fraud detection, Accuracy is NOT enough.

Use:

Accuracy

Precision

Recall (Very Important)

F1 Score

ROC-AUC

Confusion Matrix

👉 Important for interview:

Recall is critical because we want to detect maximum fraud cases.

✅ Activity 5.2: Hyperparameter Tuning

Use:

GridSearchCV

RandomizedSearchCV

Example:

Random Forest:

n_estimators

max_depth

min_samples_split

Compare:

Before tuning accuracy

After tuning accuracy

🔵 Epic 6: Model Deployment
✅ Activity 6.1: Save Best Model

Use:

Pickle

Joblib

Example:

import joblib
joblib.dump(model, "fraud_model.pkl")
✅ Activity 6.2: Integrate with Web Framework

Use:

Flask (Recommended for beginner)

Streamlit (Easy UI)

FastAPI (Advanced)

Basic Flow:
User inputs claim details → Model predicts → Show result.

🔵 Epic 7: Project Demonstration & Documentation
✅ Activity 7.1: Record Explanation Video

Explain:

Problem Statement

Dataset

EDA Insights

Model Comparison

Final Model

Deployment Demo

Keep video 7–10 minutes.

✅ Activity 7.2: Documentation

Include:

Introduction

Literature Review

Methodology

Data Preprocessing

Model Building

Results

Conclusion

Future Scope
