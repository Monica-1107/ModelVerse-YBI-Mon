import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
credit = pd.read_csv('https://github.com/YBI-Foundation/Dataset/raw/main/Credit%20Default.csv')

# Define target variable and feature variables
y = credit['Default']
x = credit[['Income', 'Age', 'Loan', 'Loan to Income']]

# Split data into training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y)

# Select and train model (Logistic Regression or Random Forest)
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Save the model as a .pkl file
joblib.dump(model, 'credit_default_model.pkl')
print("Model saved as 'credit_default_model.pkl'")
