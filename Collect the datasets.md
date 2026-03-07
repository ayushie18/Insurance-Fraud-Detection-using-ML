✅ Activity 1.1: Importing the Libraries
📌 Required Libraries

These libraries are used for:

Data handling → numpy, pandas

Visualization → matplotlib, seaborn

Model building → sklearn

Evaluation → f1_score, classification_report, confusion_matrix

Saving model → pickle

Statistical analysis → scipy

💻 Code:
# Numerical & Data Handling
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Model Selection
from sklearn.model_selection import train_test_split

# Algorithms
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Evaluation Metrics
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix

# Other Utilities
import warnings
import pickle
from scipy import stats

# Ignore warnings
warnings.filterwarnings('ignore')

# Visualization Style (Optional)
plt.style.use('fivethirtyeight')
🎯 Why We Use fivethirtyeight Style?

It improves plot readability and gives professional looking graphs.

✅ Activity 1.2: Read the Dataset

Your dataset:

insurance_claims.csv

Downloaded from Kaggle:
👉 https://www.kaggle.com/datasets/buntyshah/auto-insurance-claims-data

💻 Code to Read Dataset
# Read the CSV file
df = pd.read_csv('insurance_claims.csv')

# Display first 5 rows
df.head()
📊 Understanding Dataset Structure

To understand data properly:

# Check shape
df.shape
# Check data types
df.info()
# Statistical summary
df.describe()
✅ Checking Missing Values
📌 Step 1: Check if any null values exist
df.isna().any()
📌 Step 2: Count total null values column-wise
df.isna().sum()
🎯 Expected Result

From your provided image:
✔ There are no null values
✔ So we can skip missing value handling

🔥 Best Practice (Extra — For Stronger Understanding)

Even if there are no null values, always check:

# Check duplicate rows
df.duplicated().sum()
# Check unique target values
df['fraud_reported'].value_counts()
📌 Summary of Activity 1
Step	Work Done
Imported libraries	✅
Read CSV file	✅
Viewed first 5 rows	✅
Checked data types	✅
Checked null values	✅
Confirmed no missing data	✅


✅ Activity 1.2: Data Preparation

This step directly affects your model accuracy.
If data preparation is wrong → model performance will be poor ❌

🎯 What We Will Do in Data Preparation

Remove unnecessary columns

Handle categorical variables

Convert target variable

Detect & treat outliers (optional but good practice)

Split features & target

Train-test split

🔹 Step 1: Remove Unnecessary Columns

Some columns are not useful for prediction:

policy_number

policy_bind_date

insured_zip

incident_location

These are IDs or high-cardinality features.

💻 Code
# Drop irrelevant columns
df.drop(['policy_number', 
         'policy_bind_date', 
         'insured_zip', 
         'incident_location'], axis=1, inplace=True)
🔹 Step 2: Convert Target Variable

In this dataset, the target column is:

fraud_reported

Values:

Y → Fraud

N → Not Fraud

We convert into numeric:

💻 Code
df['fraud_reported'] = df['fraud_reported'].map({'Y':1, 'N':0})

✔ Now:

1 = Fraud

0 = Not Fraud

🔹 Step 3: Handle Categorical Variables

Let’s check categorical columns:

df.select_dtypes(include='object').columns

You will see columns like:

policy_state

policy_csl

insured_sex

insured_education_level

insured_occupation

insured_hobbies

incident_type

incident_severity

authorities_contacted

incident_state

incident_city

property_damage

police_report_available

🎯 Convert Categorical → Numeric

We use Label Encoding or One Hot Encoding

For simplicity and performance, we use:

df = pd.get_dummies(df, drop_first=True)

✔ Converts all categorical columns
✔ Avoids dummy variable trap

🔹 Step 4: Handle Outliers (Optional but Good Practice)

Example using Z-score:

z = np.abs(stats.zscore(df))
df = df[(z < 3).all(axis=1)]

✔ Removes extreme outliers
✔ Makes model stable

🔹 Step 5: Separate Features & Target
X = df.drop('fraud_reported', axis=1)
y = df['fraud_reported']
🔹 Step 6: Train-Test Split

Very important step 🔥

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.25, 
    random_state=42
)

✔ 75% Training
✔ 25% Testing
✔ random_state ensures reproducibility

📊 Final Checklist
Step	Status
Removed unnecessary columns	✅
Converted target column	✅
Encoded categorical variables	✅
Removed outliers	✅
Split features & target	✅
Train-test split	✅
