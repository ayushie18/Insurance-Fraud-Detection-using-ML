PREREQUISITES: INSURANCE FRAUD DETECTION
1. MACHINE LEARNING CONCEPTS
SUPERVISED LEARNING: Since our dataset contains historical claims labeled as "Fraud" or "Legitimate," we use Supervised Learning to train models that can map input features (like claim amount or policy type) to these known outcomes.

UNSUPERVISED LEARNING: While not the primary focus, unsupervised techniques like Clustering (K-Means) can be used during the Exploratory Data Analysis (EDA) phase to identify unusual clusters of claims that might indicate new, previously unknown fraud patterns.

CLASSIFICATION VS. REGRESSION: This project is a Binary Classification task because the output is categorical (0 for Legitimate, 1 for Fraud) rather than a continuous numerical value.

2. ALGORITHMS RESEARCH
DECISION TREE: A tree-based model that splits the data into branches based on feature values. It is highly interpretable, allowing us to see exactly which features (e.g., "Is the claim over $10,000?") lead to a fraud prediction.

RANDOM FOREST: An "Ensemble" method that builds many Decision Trees and merges their results. This reduces "overfitting" (where a model learns noise rather than patterns) and provides a more stable and accurate prediction for complex insurance data.

KNN (K-NEAREST NEIGHBORS): A distance-based algorithm that classifies a claim based on how similar it is to its "neighbors" in the dataset. It is useful for detecting fraud in cases where fraudulent claims share very specific, tight clusters of characteristics.

XGBOOST (EXTREME GRADIENT BOOSTING): A high-performance version of Gradient Boosting. It is widely used in industry for tabular data because it handles missing values well and is extremely fast and accurate for imbalanced datasets like fraud detection.



3. EVALUATION METRICS
In fraud detection, Accuracy can be misleading if 99% of claims are legitimate. We focus on:

PRECISION: Out of all claims we flagged as fraud, how many were actually fraud? High precision avoids bothering innocent customers.

RECALL (SENSITIVITY): Out of all actual fraud cases, how many did we successfully catch? High recall is vital because missing a single large fraud claim is expensive for the company.

F1-SCORE: The harmonic mean of Precision and Recall. This is our primary metric to ensure we have a balanced model that doesn't sacrifice one for the other.

ROC-AUC: A curve that helps us visualize the trade-off between catching fraud and triggering false alarms across different probability thresholds.

4. FLASK BASICS & DEPLOYMENT
MICRO-FRAMEWORK: Flask is a lightweight WSGI web application framework in Python. We use it because it is simple to scale and perfect for serving Machine Learning models via APIs.

MODEL PICKLING: We will use the pickle or joblib library to save our trained model as a file. Flask will then load this file to make "live" predictions on new data entered by the user.

ROUTING: We will create specific URL routes (e.g., /predict) that take user input from an HTML form, pass it to the model, and return the "Fraud" or "Safe" result to the browser.
