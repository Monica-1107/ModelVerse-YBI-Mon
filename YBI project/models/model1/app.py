# models/model1/app.py

from flask import Blueprint, render_template, request
import joblib
import numpy as np

# Create a blueprint for model1
model1_bp = Blueprint('model1', __name__, template_folder='templates', static_folder='static')

# Load the model (only load once)
model = joblib.load('C:/Users/Dell/Documents/YBI project/models/model1/loan_model.pkl')

@model1_bp.route('/')
def model1_home():
    return render_template('model1_form.html')  # Render the Model 1 form page

@model1_bp.route('model1/predict', methods=['POST'])
def model1_predict():
    try:
        # Get inputs from the form
        gender = request.form.get('gender')
        married = request.form.get('married')
        dependents = request.form.get('dependents')
        education = request.form.get('education')
        self_employed = request.form.get('self_employed')
        applicant_income = request.form.get('applicant_income')
        coapplicant_income = request.form.get('coapplicant_income')
        loan_amount = request.form.get('loan_amount')
        loan_amount_term = request.form.get('loan_amount_term')
        credit_history = request.form.get('credit_history')
        property_area = request.form.get('property_area')

        # Validate numeric inputs
        if not all([applicant_income, coapplicant_income, loan_amount, loan_amount_term, credit_history]):
            return render_template('model1_form.html', prediction_text='Please fill all the fields.')
        # Convert inputs to appropriate types
        applicant_income = float(applicant_income)
        coapplicant_income = float(coapplicant_income)
        loan_amount = float(loan_amount)
        loan_amount_term = float(loan_amount_term)
        credit_history = float(credit_history)

        # Map input data to numerical values
        data = [
            1 if gender == 'Male' else 0,
            1 if married == 'Yes' else 0,
            int(dependents) if dependents != '3+' else 3,
            1 if education == 'Graduate' else 0,
            1 if self_employed == 'Yes' else 0,
            applicant_income,
            coapplicant_income,
            loan_amount,
            loan_amount_term,
            credit_history,
            0 if property_area == 'Rural' else 1 if property_area == 'Urban' else 2
        ]

        # Make prediction
        prediction = model.predict([data])
        loan_status = "Eligible" if prediction[0] == 1 else "Not Eligible"

        return render_template('model1_form.html', prediction_text=f'Loan Status: {loan_status}')

    except ValueError as e:
        return render_template('model1_form.html', prediction_text='Please enter valid numerical values.')
    except Exception as e:
        return render_template('model1_form.html', prediction_text='An unexpected error occurred. Please try again.')
