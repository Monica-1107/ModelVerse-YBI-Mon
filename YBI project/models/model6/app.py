from flask import Blueprint, render_template, request
import joblib
import numpy as np

# Define a blueprint for model6
model6_bp = Blueprint('model6', __name__, template_folder='templates', static_folder='static')

# Load the saved model (only once)
model = joblib.load('models/model6/admission_chance_model.pkl')

@model6_bp.route('/')
def model6_home():
    return render_template('model6_form.html')  # Render the Model 6 form page

@model6_bp.route('model6/predict', methods=['POST'])
def model6_predict():
    try:
        # Get inputs from the form
        gre_score = float(request.form.get('gre_score'))
        toefl_score = float(request.form.get('toefl_score'))
        university_rating = float(request.form.get('university_rating'))
        sop = float(request.form.get('sop'))
        lor = float(request.form.get('lor'))
        cgpa = float(request.form.get('cgpa'))
        research = int(request.form.get('research'))

        # Prepare data for prediction
        data = np.array([[gre_score, toefl_score, university_rating, sop, lor, cgpa, research]])

        # Make prediction
        prediction = model.predict(data)[0]
        result = f"Chance of Admission: {prediction:.2f}"

        return render_template('model6_form.html', prediction_text=result)

    except ValueError:
        return render_template('model6_form.html', prediction_text='Please enter valid numerical values.')
    except Exception as e:
        return render_template('model6_form.html', prediction_text='An unexpected error occurred. Please try again.')
