from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load your model from the 'model' folder

model = pickle.load(open("model/fraud_model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

FEATURES_LIST = [
    'months_as_customer', 'age', 'policy_number', 'policy_bind_date', 'policy_state',
    'policy_csl', 'policy_deductable', 'policy_annual_premium', 'umbrella_limit',
    'insured_zip', 'insured_sex', 'insured_education_level', 'insured_occupation',
    'insured_hobbies', 'insured_relationship', 'capital-gains', 'capital-loss',
    'incident_date', 'incident_type', 'collision_type', 'incident_severity',
    'authorities_contacted', 'incident_state', 'incident_city', 'incident_location',
    'incident_hour_of_the_day', 'number_of_vehicles_involved', 'property_damage',
    'bodily_injuries', 'witnesses', 'police_report_available', 'total_claim_amount',
    'injury_claim', 'property_claim', 'vehicle_claim', 'auto_make', 'auto_model', 'auto_year'
]

# 1. MEAN IMPUTATION DICTIONARY
# These values prevent the model from seeing "zeros" as the default state.
MEANS_DICT = {
    'months_as_customer': 204.0, 'age': 38.0, 'policy_deductable': 1136.0,
    'policy_annual_premium': 1256.0, 'umbrella_limit': 110100.0, 'capital-gains': 25126.0,
    'capital-loss': -26793.0, 'incident_hour_of_the_day': 11.0, 
    'number_of_vehicles_involved': 1.8, 'bodily_injuries': 0.99, 'witnesses': 1.48,
    'total_claim_amount': 52761.0, 'injury_claim': 7433.0, 'property_claim': 7399.0,
    'vehicle_claim': 37928.0, 'auto_year': 2005.0
}

# 2. INDIAN CURRENCY LOGIC
# 1 USD ≈ 83 INR. Converts ₹ inputs to the model's expected scale.
INR_TO_USD = 83.0

@app.route('/')
def home():
    # Shows the card selection screen by default
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    # Initialize with Mean Values instead of Zeros
    input_vector = np.zeros(len(FEATURES_LIST))
    for i, feature in enumerate(FEATURES_LIST):
        input_vector[i] = MEANS_DICT.get(feature, 0)

    category = request.form.get('category_name')
    form_data = request.form

    currency_features = [
        'policy_annual_premium', 'policy_deductable', 'umbrella_limit',
        'total_claim_amount', 'injury_claim', 'property_claim', 'vehicle_claim'
    ]

    # Map UI inputs to the correct index and handle currency
    for i, feature_name in enumerate(FEATURES_LIST):
        if feature_name in form_data and form_data[feature_name] != "":
            try:
                val = float(form_data[feature_name])

                # Apply Indian Currency Conversion
                if feature_name in currency_features:
                    val = val / INR_TO_USD

                input_vector[i] = val

            except ValueError:
                input_vector[i] = MEANS_DICT.get(feature_name, 0)

    # ✅ These must be INSIDE the function
    # Reshape and Scale input
    scaled_input = scaler.transform(input_vector.reshape(1, -1))

    # Predict probability
    proba = model.predict_proba(scaled_input)

    # Convert probability to result
    if proba[0][1] > 0.35:
        result = "⚠️ Fraud Claim Detected"
    else:
        result = "✅ Claim Looks Genuine"

    # Return active_category to keep the correct form open
    return render_template(
        "index.html",
        prediction_text=result,
        active_category=category
    )

if __name__ == '__main__':
    app.run(debug=True)