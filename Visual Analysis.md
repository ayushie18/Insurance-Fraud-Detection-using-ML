✅ Story 2.1: Univariate Analysis
🔹 What is Univariate Analysis?

Univariate analysis means:

📊 Analyzing one feature at a time to understand its distribution, frequency, and composition.

We will analyze:

Target Variable → fraud_reported

Incident Severity

Age Distribution

🔶 1️⃣ Countplot – Fraud Reported
📌 Code:
plt.figure(figsize=(6,4))
sns.countplot(x='fraud_reported', data=df)
plt.title("Fraud Reported Count")
plt.show()
📊 Insight from Countplot

From your graph:

Fraud cases (Y) ≈ 247

Non-fraud cases (N) ≈ 753

Total claims = 1000

🎯 Important Observation:

⚠ Dataset is imbalanced

Fraud = 24.7%

Non-Fraud = 75.3%

👉 This means:
Accuracy alone is NOT a good metric
We must focus on:

F1 Score

Precision

Recall

💡 Very important for placements!
