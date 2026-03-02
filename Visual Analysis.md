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

![image](https://github.com/ayushie18/Insurance-Fraud-Detection-using-ML/blob/885d395f05905f69eba1114f150c91b914baec6e/Screenshot%202026-03-02%20220307.png)



🔶 2️⃣ Pie Chart – Incident Severity
📌 Code:
severity_counts = df['incident_severity'].value_counts()

plt.figure(figsize=(6,6))
plt.pie(severity_counts,
        labels=severity_counts.index,
        autopct='%1.1f%%',
        startangle=90)
plt.title("Damage Visualization")
plt.show()
📊 Insight from Pie Chart

From your image:

Minor Damage → 35.4%

Total Loss → 28.0%

Major Damage → 27.6%

Trivial Damage → 9.0%

🎯 Interpretation:

Most claims are Minor Damage

Very few cases are Trivial Damage

Major Damage and Total Loss are almost equal

💡 This feature might be strongly related to fraud.
Later in bivariate analysis, we should check:

sns.countplot(x='incident_severity', hue='fraud_reported', data=df)

![image](https://github.com/ayushie18/Insurance-Fraud-Detection-using-ML/blob/cec4e2ab070675d26babb7cca49969ed8823bdb5/Screenshot%202026-03-02%20220321.png)
