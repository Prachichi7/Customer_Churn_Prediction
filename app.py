from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load your model
model = joblib.load("churn_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Numeric inputs
        tenure = float(request.form['Tenure'])
        monthly_charges = float(request.form['MonthlyCharges'])
        total_charges = float(request.form['TotalCharges'])

        # Categorical inputs
        senior_citizen = 1 if request.form['SeniorCitizen'] == "Yes" else 0
        partner = 1 if request.form['Partner'] == "Yes" else 0
        dependents = 1 if request.form['Dependents'] == "Yes" else 0
        phone_service = 1 if request.form['PhoneService'] == "Yes" else 0

        # Multiple lines
        multiple_lines = request.form['MultipleLines']
        multiple_lines_no_phone = 1 if multiple_lines == "No phone service" else 0
        multiple_lines_yes = 1 if multiple_lines == "Yes" else 0

        # Internet Service
        internet_service = request.form['InternetService']
        internet_service_fiber = 1 if internet_service == "Fiber optic" else 0
        internet_service_no = 1 if internet_service == "No" else 0

        # Online Security
        online_security = request.form['OnlineSecurity']
        online_security_no_int = 1 if online_security == "No internet service" else 0
        online_security_yes = 1 if online_security == "Yes" else 0

        # Online Backup
        online_backup = request.form['OnlineBackup']
        online_backup_no_int = 1 if online_backup == "No internet service" else 0
        online_backup_yes = 1 if online_backup == "Yes" else 0

        # Device Protection
        device_protection = request.form['DeviceProtection']
        device_protection_no_int = 1 if device_protection == "No internet service" else 0
        device_protection_yes = 1 if device_protection == "Yes" else 0

        # Tech Support
        tech_support = request.form['TechSupport']
        tech_support_no_int = 1 if tech_support == "No internet service" else 0
        tech_support_yes = 1 if tech_support == "Yes" else 0

        # Streaming TV
        streaming_tv = request.form['StreamingTV']
        streaming_tv_no_int = 1 if streaming_tv == "No internet service" else 0
        streaming_tv_yes = 1 if streaming_tv == "Yes" else 0

        # Streaming Movies
        streaming_movies = request.form['StreamingMovies']
        streaming_movies_no_int = 1 if streaming_movies == "No internet service" else 0
        streaming_movies_yes = 1 if streaming_movies == "Yes" else 0

        # Contract
        contract = request.form['Contract']
        contract_one_year = 1 if contract == "One year" else 0
        contract_two_year = 1 if contract == "Two year" else 0

        # Paperless Billing
        paperless_billing = 1 if request.form['PaperlessBilling'] == "Yes" else 0

        # Payment Method
        payment_method = request.form['PaymentMethod']
        pay_credit = 1 if payment_method == "Credit card (automatic)" else 0
        pay_electronic = 1 if payment_method == "Electronic check" else 0
        pay_mailed = 1 if payment_method == "Mailed check" else 0

        # Tenure Group
        tenure_group = request.form['TenureGroup']
        tenure_1_2 = 1 if tenure_group == "1-2" else 0
        tenure_2_3 = 1 if tenure_group == "2-3" else 0
        tenure_3_4 = 1 if tenure_group == "3-4" else 0
        tenure_4_6 = 1 if tenure_group == "4-6" else 0

        # Build final feature vector in SAME ORDER
        input_features = [
            tenure, monthly_charges, total_charges,
            senior_citizen, partner, dependents, phone_service,
            multiple_lines_no_phone, multiple_lines_yes,
            internet_service_fiber, internet_service_no,
            online_security_no_int, online_security_yes,
            online_backup_no_int, online_backup_yes,
            device_protection_no_int, device_protection_yes,
            tech_support_no_int, tech_support_yes,
            streaming_tv_no_int, streaming_tv_yes,
            streaming_movies_no_int, streaming_movies_yes,
            contract_one_year, contract_two_year,
            paperless_billing,
            pay_credit, pay_electronic, pay_mailed,
            tenure_1_2, tenure_2_3, tenure_3_4, tenure_4_6
        ]

        # Reshape for model
        input_array = np.array(input_features).reshape(1, -1)

        # Predict
        prediction = model.predict(input_array)[0]
        result = "Yes (Customer will Churn)" if prediction == 1 else "No (Customer will Stay)"

        return render_template("index.html", prediction_text=f"Prediction: {result}")

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
