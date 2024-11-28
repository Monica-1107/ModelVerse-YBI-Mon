import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

# Load and preprocess the dataset
def load_data():
    print("Loading dataset...")
    url = 'https://github.com/YBIFoundation/Dataset/raw/main/Cancer.csv'
    
    try:
        cancer = pd.read_csv(url)
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None

    # Data preprocessing: Convert 'diagnosis' to binary (Malignant: 1, Benign: 0)
    cancer.replace({'diagnosis': {'M': 1, 'B': 0}}, inplace=True)

    # Define features and target
    y = cancer['diagnosis']
    x = cancer.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1)
    
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
    model = LogisticRegression(max_iter=5000)
    print("Training the model...")
    model.fit(x_train, y_train)
    print("Model trained successfully.")

    # Save the trained model
    model_path = 'cancer_model.pkl'
    joblib.dump(model, model_path)
    print(f"Model saved as '{model_path}'")

    # Evaluate accuracy
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy * 100:.2f}%")

# Predict cancer diagnosis
def predict_cancer(data):
    model_path = 'cancer_model.pkl'
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
    return "Malignant" if prediction[0] == 1 else "Benign"

