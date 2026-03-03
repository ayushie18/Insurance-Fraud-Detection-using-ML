📌 Story 1.2 – Random Forest Model
Step 1: Import Required Libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
Step 2: Initialize the Model
rf_model = RandomForestClassifier(
    n_estimators=100,      # number of trees
    max_depth=None,        # no depth limit
    random_state=42
)

👉 n_estimators = number of decision trees
👉 max_depth = controls overfitting
👉 random_state = reproducibility

Step 3: Train the Model
rf_model.fit(X_train, y_train)

Here:

X_train → Training features

y_train → Training labels (Fraud / Not Fraud)

Step 4: Make Predictions
y_pred = rf_model.predict(X_test)
Step 5: Evaluate Model Performance
🔹 Train Accuracy
train_accuracy = rf_model.score(X_train, y_train)
print("Train Accuracy:", train_accuracy)
🔹 Test Accuracy
test_accuracy = rf_model.score(X_test, y_test)
print("Test Accuracy:", test_accuracy)
Step 6: Detailed Evaluation (Very Important for Fraud Detection)

Since fraud datasets are usually imbalanced, accuracy alone is NOT enough.

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

Focus on:

Precision → How many predicted frauds were actually fraud?

Recall → How many actual frauds did we correctly detect?

F1-score → Balance between precision & recall

⚠️ In fraud detection, Recall is very important
Because missing a fraud case is costly.

🎯 Why Random Forest is Good for Fraud Detection?
Feature	Benefit
Multiple trees	Better accuracy
Bootstrapping	Less overfitting
Feature importance	Helps understand key fraud factors
Works with high dimensions	Good for insurance datasets
📊 Extra: Feature Importance
import pandas as pd

feature_importance = pd.Series(
    rf_model.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

print(feature_importance.head(10))

This helps you identify:

Most important fraud indicators

Useful insights for business decisions
