import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import pickle


# **Import dataset**
df = pd.read_csv('https://github.com/YBI-Foundation/Dataset/raw/main/Ice%20Cream.csv')

# **Define Target variable(y) and Feature variables (x)**
y = df['Revenue']
x = df[['Temperature']]

# **Train Test Split**
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=2529)

# **Model Selection**
model = LinearRegression()

# Train the model
model.fit(x_train, y_train)

# **Test the model**
y_pred = model.predict(x_test)

# Evaluate model accuracy
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# Print evaluation metrics
print(f'Mean Absolute Error: {mae}')
print(f'Mean Absolute Percentage Error: {mape}')
print(f'Mean Squared Error: {mse}')

# **Save the trained model to a .pkl file**
with open('ice_cream_revenue_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

print("Model saved as 'ice_cream_revenue_model.pkl'")
