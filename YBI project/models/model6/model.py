# -*- coding: utf-8 -*-
"""Chances of Admission Model Training and Saving"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import joblib

# Import data
df = pd.read_csv('https://github.com/YBI-Foundation/Dataset/raw/main/Admission%20Chance.csv')

# Define target variable (y) and feature variables (x)
y = df['Chance of Admit ']
x = df.drop(['Serial No', 'Chance of Admit '], axis=1)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=2529)

# Select and train model
model = LinearRegression()
model.fit(x_train, y_train)

# Model evaluation
y_pred = model.predict(x_test)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("Mean Absolute Percentage Error:", mape)
print("Mean Squared Error:", mse)

# Save the trained model as a .pkl file
joblib.dump(model, 'admission_chance_model.pkl')
print("Model saved as admission_chance_model.pkl")
