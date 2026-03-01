🔵 Epic 1: Define Problem / Problem Understanding
🔹 Activity 1.1: Specify the Business Problem
Banks receive thousands of loan applications daily. Manually verifying each application is time-consuming and prone to human bias.
The business problem is:
How can we automatically predict whether a loan should be approved or rejected based on applicant details?
The goal is to build a Machine Learning model that predicts loan approval status using historical applicant data.
🔹 Activity 1.2: Business Requirements
The system must:
Predict loan approval (Approved / Rejected).
Minimize risk of loan default.
Reduce manual verification time.
Performance Requirements:
Accuracy above 80%.
Low false approval rate (important for financial safety).
Technical Requirements:
Web-based interface.
Model saved and reusable.
Fast response time (< 2 seconds).
🔹 Activity 1.3: Literature Survey
Several machine learning models are commonly used for loan prediction:
Logistic Regression – Simple and interpretable.
Decision Tree – Rule-based decision making.
Random Forest – Improves accuracy using ensemble learning.
Support Vector Machine (SVM) – Effective for classification.
Gradient Boosting – High performance on structured data.
Research shows that ensemble methods (Random Forest, XGBoost) often outperform basic algorithms in financial risk prediction.
🔹 Activity 1.4: Social or Business Impact
✅ Business Impact
Reduces loan processing time.
Reduces risk of bad loans.
Improves operational efficiency.
✅ Social Impact
Fair and unbiased automated decision-making.
Faster loan approvals for eligible customers.
🔵 Epic 2: Data Collection & Preparation
🔹 Activity 2.1: Collect the Dataset
Dataset includes:
Gender
Marital Status
Education
Applicant Income
Loan Amount
Credit History
Loan Status (Target variable)
Dataset source:
Kaggle Loan Prediction Dataset
🔹 Activity 2.2: Data Preparation
Steps performed:
Handling Missing Values
Replaced missing numerical values with mean/median.
Replaced categorical missing values with mode.
Encoding Categorical Variables
Label Encoding
One-Hot Encoding
Feature Scaling
StandardScaler / MinMaxScaler
Splitting Dataset
80% Training Data
20% Testing Data
🔵 Epic 3: Exploratory Data Analysis (EDA)
🔹 Activity 3.1: Descriptive Statistics
Performed:
Mean
Median
Standard Deviation
Correlation Matrix
Value Counts of Loan Status
Key Observations:
Credit History strongly impacts loan approval.
Higher income increases chances of approval.
🔹 Activity 3.2: Visual Analysis
Used:
Bar Charts
Histograms
Boxplots
Heatmaps
Insights:
Applicants with good credit history have high approval rate.
Loan amount and income show positive correlation.
🔵 Epic 4: Model Building
🔹 Activity 4.1: Training the Model Using Multiple Algorithms
Algorithms Used:
Logistic Regression
Decision Tree
Random Forest
Support Vector Machine
Gradient Boosting
🔹 Activity 4.2: Model Comparison
Each model evaluated using:
Accuracy
Precision
Recall
F1-Score
Initial Results (Example):
Algorithm
Accuracy
Logistic Regression
78%
Decision Tree
75%
Random Forest
85%
SVM
82%
Gradient Boosting
87%
Best performing model: Gradient Boosting
🔵 Epic 5: Performance Testing & Hyperparameter Tuning
🔹 Activity 5.1: Testing Model with Multiple Evaluation Metrics
Metrics Used:
Accuracy
Confusion Matrix
Precision
Recall
F1-Score
ROC-AUC Score
🔹 Activity 5.2: Hyperparameter Tuning
Applied:
GridSearchCV
RandomSearchCV
Example Parameters Tuned:
Number of estimators
Maximum depth
Learning rate
After tuning:
Model
Before Tuning
After Tuning
Random Forest
85%
89%
Gradient Boosting
87%
91%
Final Selected Model: Tuned Gradient Boosting (91%)
🔵 Epic 6: Model Deployment
🔹 Activity 6.1: Save the Best Model
Used:
Pickle
Joblib
Model saved as:
Copy code

loan_model.pkl
🔹 Activity 6.2: Integrate with Web Framework
Web Framework Used:
Flask / Django
Steps:
Create HTML form for user input.
Load saved model.
Take user input.
Predict loan status.
Display result.
System Architecture: User → Web Form → Backend → ML Model → Prediction → Result Display
🔵 Epic 7: Project Demonstration & Documentation
🔹 Activity 7.1: Record Explanation Video
Video Content Structure:
Introduction
Problem Statement
Dataset Explanation
EDA
Model Building
Hyperparameter Tuning
Deployment Demo
Conclusion
Duration: 8–12 minutes
🔹 Activity 7.2: Project Documentation
Documentation Includes:
Abstract
Introduction
Literature Survey
System Architecture
Methodology
Implementation
Results
Conclusion
Future Scope
