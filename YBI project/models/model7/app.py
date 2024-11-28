from flask import Blueprint, render_template, request
import joblib
import numpy as np

# Define a blueprint for model7
model7_bp = Blueprint('model7', __name__, template_folder='templates', static_folder='static')

# Load the saved model (only once)
model = joblib.load('models/model7/ice_cream_revenue_model.pkl')

@model7_bp.route('/')
def model7_home():
    return render_template('model7_form.html')  # Render the Model 7 form page

@model7_bp.route('model7/predict', methods=['POST'])
def model7_predict():
    try:
        # Get the temperature input from the form
        temperature = float(request.form.get('temperature'))

        # Prepare data for prediction
        data = np.array([[temperature]])

        # Make prediction
        revenue = model.predict(data)

        # Format the result as a prediction
        prediction_text = f'Predicted Revenue: {revenue[0]:.2f}'

        return render_template('model7_form.html', prediction_text=prediction_text)

    except ValueError:
        return render_template('model7_form.html', prediction_text='Please enter a valid numerical value for temperature.')
    except Exception as e:
        return render_template('model7_form.html', prediction_text='An unexpected error occurred. Please try again.')
