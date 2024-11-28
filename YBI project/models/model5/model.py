import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
purchase = pd.read_csv('https://github.com/YBI-Foundation/Dataset/raw/main/Customer%20Purchase.csv')

# Data exploration (optional, just for viewing)
print(purchase.head())
print(purchase.info())
print(purchase.describe())

# Define target variable(y) and feature variables(x)
y = purchase[['Purchased']]
x = purchase[['Age', 'Gender', 'Education', 'Review']]

# Encoding categorical variables
x.replace({'Review': {'Poor': 0, 'Average': 1, 'Good': 2}}, inplace=True)
x.replace({'Education': {'School': 0, 'UG': 1, 'PG': 2}}, inplace=True)
x.replace({'Gender': {'Male': 0, 'Female': 1}}, inplace=True)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=2529)

# Train the Random Forest Model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Predict with the trained model
y_pred = model.predict(x_test)

# Evaluate the model's accuracy
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}')
print(f'Classification Report:\n{classification_report(y_test, y_pred)}')

# Save the trained model to a .pkl file
joblib.dump(model, 'purchase_prediction_model.pkl')

print("Model saved as purchase_prediction_model.pkl")
