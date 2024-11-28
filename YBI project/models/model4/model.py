# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
import joblib  # For saving the model

# **Dataset**
weight = pd.read_csv('https://github.com/ybifoundation/Dataset/raw/main/Fish.csv')

# **Data Exploration**
weight.head()

weight.info()

weight.describe()

weight.columns

# **Define target variable(y) and feature variables(x)**
y = weight['Weight']
x = weight.drop(['Weight', 'Species'], axis=1)

# **Train-test split**
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=2529)
x_train.shape, y_train.shape, x_test.shape, y_test.shape

# **Selecting model**
model = LinearRegression()

# **Fitting model**
model.fit(x_train, y_train)

# **Model coefficients and intercept**
print(model.coef_)
print(model.intercept_)

# **Predict model**
y_pred = model.predict(x_test)

# **Model Accuracy**
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Absolute Percentage Error: {mape}")

# **Save the trained model as a .pkl file**
joblib.dump(model, 'fish_weight_model.pkl')
