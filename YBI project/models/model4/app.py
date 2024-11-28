from flask import Blueprint, render_template, request
import joblib
import numpy as np

# Define a blueprint for model4
model4_bp = Blueprint('model4', __name__, template_folder='templates', static_folder='static')

# Load the saved model (only once)
model = joblib.load('models/model4/fish_weight_model.pkl')

@model4_bp.route('/')
def model4_home():
    return render_template('model4_form.html')  # Render the Fish Weight Prediction form page

@model4_bp.route('/model4/predict', methods=['POST'])
def model4_predict():
    try:
        # Get inputs from the form (Convert them to float)
        length1 = request.form.get('length1')
        length2 = request.form.get('length2')
        length3 = request.form.get('length3')
        height = request.form.get('height')
        width = request.form.get('width')
        feature6 = request.form.get('feature6')  # Get the missing 6th feature

        # Check if all fields are filled
        if not all([length1, length2, length3, height, width, feature6]):
            return render_template('model4_form.html', prediction_text='Please fill in all the fields.')

        # Try converting the values to floats
        try:
            length1 = float(length1)
            length2 = float(length2)
            length3 = float(length3)
            height = float(height)
            width = float(width)
            feature6 = float(feature6)  # Convert the 6th feature to float
        except ValueError:
            return render_template('model4_form.html', prediction_text='Please enter valid numerical values for all fields.')

        # Prepare data for prediction (input as a 2D array for the model)
        data = np.array([[length1, length2, length3, height, width, feature6]])

        # Make prediction
        predicted_weight = model.predict(data)[0]  # Getting the predicted weight

        return render_template('model4_form.html', prediction_text=f'The predicted weight of the fish is: {predicted_weight:.2f} grams')

    except Exception as e:
        return render_template('model4_form.html', prediction_text=f'An unexpected error occurred: {str(e)}')
