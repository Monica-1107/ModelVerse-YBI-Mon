from flask import Blueprint, render_template, request
import joblib
import numpy as np

# Create a blueprint for model2
model2_bp = Blueprint('model2', __name__, template_folder='templates', static_folder='static')

# Attempt to load the model once and log if successful or not
try:
    model = joblib.load('C:/Users/Dell/Documents/YBI project/models/model2/cancer_model.pkl')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@model2_bp.route('/')
def model2_home():
    return render_template('model2_form.html')  # Render the Model 2 form page

@model2_bp.route('/model2/predict', methods=['POST'])
def model2_predict():
    if model is None:
        # If the model failed to load, return an error message
        return render_template('model2_form.html', prediction_text="Model failed to load. Please contact support.")

    try:
        # Print form data for debugging
        form_data = request.form
        print("Form data received:", form_data)

        # Get inputs from the form and validate them
        radius = request.form.get('radius')
        texture = request.form.get('texture')
        perimeter = request.form.get('perimeter')
        area = request.form.get('area')
        smoothness = request.form.get('smoothness')
        compactness = request.form.get('compactness')
        concavity = request.form.get('concavity')
        concave_points = request.form.get('concave_points')
        symmetry = request.form.get('symmetry')
        fractal_dimension = request.form.get('fractal_dimension')

        # Ensure all fields are filled
        if not all([radius, texture, perimeter, area, smoothness, compactness, concavity, concave_points, symmetry, fractal_dimension]):
            print("Missing field in form data.")
            return render_template('model2_form.html', prediction_text='Please fill all the fields.')

        # Convert inputs to float and log each conversion
        try:
            data = [
                float(radius), float(texture), float(perimeter), float(area),
                float(smoothness), float(compactness), float(concavity),
                float(concave_points), float(symmetry), float(fractal_dimension)
            ]
            print("Data converted for prediction:", data)
        except ValueError:
            print("Conversion error - non-numeric input detected.")
            return render_template('model2_form.html', prediction_text='Please enter valid numerical values.')

        # Make prediction and display the result
        prediction = model.predict([data])
        print("Prediction result:", prediction)
        cancer_type = "Malignant" if prediction[0] == 1 else "Benign"

        return render_template('model2_form.html', prediction_text=f'Cancer Prediction: {cancer_type}')

    except Exception as e:
        # Catch unexpected errors and print the exception for debugging
        print("Unexpected error occurred:", e)
        return render_template('model2_form.html', prediction_text='An unexpected error occurred. Please try again.')
