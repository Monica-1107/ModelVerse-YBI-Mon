# models/model3/app.py

from flask import Blueprint, render_template, request
import joblib

# Define a blueprint for model3
model3_bp = Blueprint('model3', __name__, template_folder='templates', static_folder='static')

# Load the saved model (only once)
model = joblib.load('models/model3/credit_default_model.pkl')

@model3_bp.route('/')
def model3_home():
    return render_template('model3_form.html')  # Render the Model 3 form page

@model3_bp.route('model3/predict', methods=['POST'])
def model3_predict():
    try:
        # Get inputs from the form
        income = float(request.form.get('income'))
        age = float(request.form.get('age'))
        loan = float(request.form.get('loan'))
        loan_to_income = float(request.form.get('loan_to_income'))

        # Prepare data for prediction
        data = [[income, age, loan, loan_to_income]]

        # Make prediction
        prediction = model.predict(data)
        result = "Default" if prediction[0] == 1 else "No Default"

        return render_template('model3_form.html', prediction_text=f'Prediction: {result}')

    except ValueError:
        return render_template('model3_form.html', prediction_text='Please enter valid numerical values.')
    except Exception as e:
        return render_template('model3_form.html', prediction_text='An unexpected error occurred. Please try again.')

