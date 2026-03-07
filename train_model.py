import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report, confusion_matrix, f1_score

import warnings
import pickle
from scipy import stats

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

plt.style.use('fivethirtyeight')

# Load dataset
df = pd.read_csv("data/insurance_claims.csv")

print("Dataset Loaded Successfully")
print("Shape:", df.shape)

# show first rows
print(df.head())
# Checking missing values
print("\nChecking Missing Values:\n")

missing_values = df.isnull().sum()

print(missing_values)
# Handling Outliers

plt.figure(figsize=(6,4))
sns.boxplot(x=df["policy_annual_premium"])
plt.title("Outliers in policy_annual_premium")
#plt.show()

# Remove outliers using IQR

Q1 = df["policy_annual_premium"].quantile(0.25)
Q3 = df["policy_annual_premium"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df = df[(df["policy_annual_premium"] >= lower_bound) &
        (df["policy_annual_premium"] <= upper_bound)]

print("Dataset shape after removing outliers:", df.shape)
# Fraud distribution
plt.figure(figsize=(6,4))
sns.countplot(x="fraud_reported", data=df)

plt.title("Fraud Reported Distribution")
plt.xlabel("Fraud Reported")
plt.ylabel("Count")

#plt.show()

# Damage type distribution
damage_counts = df["incident_severity"].value_counts()

plt.figure(figsize=(6,6))

plt.pie(
    damage_counts,
    labels=damage_counts.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Damage Visualization")

#plt.show()

# Age distribution
plt.figure(figsize=(6,4))

plt.hist(df["age"], bins=10, color="salmon", edgecolor="black")

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of People")

#plt.show()
#===========================================
# Multivariate Analysis - Correlation Heatmap
#===========================================

plt.figure(figsize=(14,10))

correlation_matrix = df.corr(numeric_only=True)

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="magma",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

#plt.show()

#=========================================
# Encoding categorical features
#=======================================



le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object' or df[col].dtype == 'category':
        df[col] = le.fit_transform(df[col])
        

# =========================
# Features and Target
# =========================

X = df.drop("fraud_reported", axis=1)
y = df["fraud_reported"]

X = df.drop("fraud_reported", axis=1)
y = df["fraud_reported"]
# =========================
# Train Test Split
# =========================



X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
# Print number of features
print("Number of features used for training:", X_train.shape[1])
# =========================
# Feature Scaling
# =========================



scaler = StandardScaler()

# Fit on training data
X_train = scaler.fit_transform(X_train)

# Transform test data
X_test = scaler.transform(X_test)

# Save scaler
with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Scaler saved successfully!")



X_train = pd.DataFrame(X_train, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)
#=========DECISION TREE MODEL==============

# Decision Tree Model

dtc = DecisionTreeClassifier()

# train the model
dtc.fit(X_train, y_train)

# prediction
y_pred = dtc.predict(X_test)

# accuracy
dtc_train_acc = accuracy_score(y_train, dtc.predict(X_train))
dtc_test_acc = accuracy_score(y_test, y_pred)

print("Decision Tree Train Accuracy:", dtc_train_acc)
print("Decision Tree Test Accuracy:", dtc_test_acc)

#===========RANDOM FOREST CLASSIFIER========
# Random Forest Model
rfc = RandomForestClassifier(
    random_state=42,
    class_weight="balanced"
)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(
    estimator=rfc,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# Best model after tuning
best_rf = grid_search.best_estimator_

print("Best Parameters:", grid_search.best_params_)

y_pred_rf = best_rf.predict(X_test)


rf_train_acc = accuracy_score(y_train, best_rf.predict(X_train))
rf_test_acc = accuracy_score(y_test, y_pred_rf)

print("Random Forest Train Accuracy:", rf_train_acc)
print("Random Forest Test Accuracy:", rf_test_acc)

#=============KNN===================
knn = KNeighborsClassifier(n_neighbors=5)

# Train model
knn.fit(X_train, y_train)

# Prediction
y_pred_knn = knn.predict(X_test)

# Accuracy
knn_train_acc = accuracy_score(y_train, knn.predict(X_train))
knn_test_acc = accuracy_score(y_test, y_pred_knn)

print("KNN Train Accuracy:", knn_train_acc)
print("KNN Test Accuracy:", knn_test_acc)

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_knn))

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_knn))

#===========LOGISTIC REGRESSION=========
lg = LogisticRegressionCV(solver='lbfgs', max_iter=5000, cv=10)

# Train model
lg.fit(X_train, y_train)

# Prediction
lrg_pred = lg.predict(X_test)

# Accuracy
print("Logistic Regression Accuracy:", accuracy_score(y_test, lrg_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, lrg_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, lrg_pred))

#==========NAIVE BAYES================


gnb = GaussianNB()

# Train model
gnb.fit(X_train, y_train)

# Prediction
y_pred_nb = gnb.predict(X_test)

# Accuracy
print("Naive Bayes Train Accuracy:", accuracy_score(y_train, gnb.predict(X_train)))
print("Naive Bayes Test Accuracy:", accuracy_score(y_test, y_pred_nb))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_nb))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_nb))

#================SVM=============

svc = SVC()

# train model
svc.fit(X_train, y_train)

# prediction
y_pred_svc = svc.predict(X_test)

# accuracy
svc_train_acc = accuracy_score(y_train, svc.predict(X_train))
svc_test_acc = accuracy_score(y_test, y_pred_svc)

print("Training accuracy of SVC:", svc_train_acc)
print("Test accuracy of SVC:", svc_test_acc)

# confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_svc))

# classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_svc))

#============Testing the Model=========


sample_claim = [[
328,521585,2012,12,250,1000,1406.91,5600,
1,100,25,25,50000,0,120,23,56,52,1,123,
2,3,1,0,2,1,150000,2,25,2002,
0,1,2,3,4,5,6,7
]]
sample_claim = pd.DataFrame(sample_claim, columns=X.columns)
sample_claim = scaler.transform(sample_claim)

prediction =  best_rf.predict(sample_claim)


print("\nTest Prediction:", prediction)

if prediction[0] == 0:
    print("Prediction: Not Fraud")
else:
    print("Prediction: Fraud")

# ==============================
# Model Comparison
# ==============================

print("\n===== Model Comparison =====")

print("Decision Tree Accuracy:", dtc_test_acc)
print("Random Forest Accuracy:", rf_test_acc)
print("KNN Accuracy:", knn_test_acc)
print("Logistic Regression Accuracy:", accuracy_score(y_test, lrg_pred))
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svc))  

#==========SAVE MODEL===========
with open("model/fraud_model.pkl", "wb") as f:
    pickle.dump(best_rf, f)

print("Model saved successfully!")