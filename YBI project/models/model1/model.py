import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

# Load and preprocess the dataset
def load_data():
    print("Loading dataset...")
    url = 'https://github.com/YBI-Foundation/Dataset/raw/main/Loan%20Eligibility%20Prediction.csv'
    
    try:
        loan = pd.read_csv(url)
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None

    # Data preprocessing
    loan.replace({'Gender': {'Male': 1, 'Female': 0}}, inplace=True)
    loan.replace({'Self_Employed': {'Yes': 1, 'No': 0}}, inplace=True)
    loan.replace({'Property_Area': {'Rural': 0, 'Semiurban': 2, 'Urban': 1}}, inplace=True)
    loan.replace({'Education': {'Graduate': 1, 'Not Graduate': 0}}, inplace=True)
    loan.replace({'Married': {'Yes': 1, 'No': 0}}, inplace=True)
    loan.replace({'Dependents': {'0': 0, '1': 1, '2': 2, '3+': 3}}, inplace=True)
    loan.replace({'Loan_Status': {'Y': 1, 'N': 0}}, inplace=True)

    # Define features and target
    y = loan['Loan_Status']
    x = loan[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 
              'Applicant_Income', 'Coapplicant_Income', 'Loan_Amount', 
              'Loan_Amount_Term', 'Credit_History', 'Property_Area']]
    
    return x, y

# Train the model and save it
def train_model():
    x, y = load_data()
    if x is None or y is None:
        print("Data loading failed. Exiting training.")
        return

    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2529)
    print("Data split into training and test sets.")

    # Train the model
    model = RandomForestClassifier(random_state=42)
    print("Training the model...")
    model.fit(x_train, y_train)
    print("Model trained successfully.")

    # Save the trained model
    model_path = 'loan_model.pkl'
    joblib.dump(model, model_path)
    print(f"Model saved as '{model_path}'")

    # Evaluate accuracy
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy * 100:.2f}%")

# Predict loan eligibility
def predict_eligibility(data):
    model_path = 'loan_model.pkl'
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found. Please train the model first.")
        return None

    # Load the model
    try:
        model = joblib.load(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Make prediction
    prediction = model.predict([data])
    return prediction[0]

# Run the training function to save the model
if __name__ == "__main__":
    train_model()
