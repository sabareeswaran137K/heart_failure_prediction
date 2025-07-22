from flask import Flask, render_template, request
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load trained model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input from form
        inputs = [
            float(request.form['age']),
            float(request.form['anaemia']),
            float(request.form['creatinine_phosphokinase']),
            float(request.form['diabetes']),
            float(request.form['ejection_fraction']),
            float(request.form['high_blood_pressure']),
            float(request.form['platelets']),
            float(request.form['serum_creatinine']),
            float(request.form['serum_sodium']),
            float(request.form['sex']),
            float(request.form['smoking']),
            float(request.form['time']),
        ]

        # Scale inputs
        inputs_scaled = scaler.transform([inputs])

        # Predict
        prediction = model.predict(inputs_scaled)[0]

        # Result message
        if prediction == 1:
            result = "⚠️ High Risk of Death"
        else:
            result = "✅ Patient Likely to Survive"

        return render_template('index.html', prediction_text=result)
    
    except Exception as e:
        return f"❌ Error occurred: {e}"

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
