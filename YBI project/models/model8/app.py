from flask import Blueprint, render_template, request
import joblib

# Define a blueprint for the model8 (diabetes model)
model8_bp = Blueprint('model8', __name__, template_folder='templates', static_folder='static')

# Load the saved model (only once)
model = joblib.load('models/model8/diabetes_model.pkl')

@model8_bp.route('/')
def model8_home():
    return render_template('model8_form.html')  # Render the model8 prediction form page

@model8_bp.route('model8/predict', methods=['POST'])
def model8_predict():
    try:
        # Get inputs from the form
        pregnancies = float(request.form.get('pregnancies'))
        glucose = float(request.form.get('glucose'))
        diastolic = float(request.form.get('diastolic'))
        triceps = float(request.form.get('triceps'))
        insulin = float(request.form.get('insulin'))
        bmi = float(request.form.get('bmi'))
        dpf = float(request.form.get('dpf'))
        age = float(request.form.get('age'))

        # Prepare data for prediction
        data = [[pregnancies, glucose, diastolic, triceps, insulin, bmi, dpf, age]]

        # Make prediction
        prediction = model.predict(data)
        result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"

        return render_template('model8_form.html', prediction_text=f'Prediction: {result}')

    except ValueError:
        return render_template('model8_form.html', prediction_text='Please enter valid numerical values.')
    except Exception as e:
        return render_template('model8_form.html', prediction_text='An unexpected error occurred. Please try again.')
