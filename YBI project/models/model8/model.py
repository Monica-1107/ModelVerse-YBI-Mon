import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Load the dataset
diabetes = pd.read_csv('https://github.com/YBI-Foundation/Dataset/raw/main/Diabetes.csv')

# Prepare features (X) and target (y)
y = diabetes['diabetes']
x = diabetes[['pregnancies', 'glucose', 'diastolic', 'triceps', 'insulin', 'bmi',
              'dpf', 'age']]

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=2529)

# Initialize the Logistic Regression model
model = LogisticRegression()

# Train the model
model.fit(x_train, y_train)

# Save the trained model to a .pkl file
joblib.dump(model, 'diabetes_model.pkl')

# Optionally, you can print out model coefficients and intercept for verification
print(f"Model intercept: {model.intercept_}")
print(f"Model coefficients: {model.coef_}")

# Test the model (you can skip this part if you don't need to test the model here)
y_pred = model.predict(x_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")
