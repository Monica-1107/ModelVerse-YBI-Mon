from flask import Blueprint, render_template, request
import joblib
import numpy as np

# Define a blueprint for model4
model5_bp = Blueprint('model5', __name__, template_folder='templates', static_folder='static')

# Load the saved model (only once)
model = joblib.load('models/model5/purchase_prediction_model.pkl')  # Path to the saved model

@model5_bp.route('/')
def model5_home():
    return render_template('model5_form.html')  # Render the Fish Weight Prediction form page

@model5_bp.route('model5/predict', methods=['POST'])
def model5_predict():
    try:
        # Get inputs from the form (convert them to float)
        age = float(request.form.get('age'))
        gender = float(request.form.get('gender'))
        education = float(request.form.get('education'))
        review = float(request.form.get('review'))

        # Prepare data for prediction (input as a 2D array for the model)
        data = np.array([[age, gender, education, review]])

        # Make prediction
        predicted_purchase = model.predict(data)[0]  # Getting the predicted result

        # Prepare result based on prediction (Purchase or No Purchase)
        result = "Purchase" if predicted_purchase == 1 else "No Purchase"

        return render_template('model5_form.html', prediction_text=f'Prediction: {result}')

    except ValueError:
        return render_template('model5_form.html', prediction_text='Please enter valid numerical values.')
    except Exception as e:
        return render_template('model5_form.html', prediction_text='An unexpected error occurred. Please try again.')
