import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load the data
df = pd.read_csv('cpi.csv')

# Define the target variable (y) and the feature (X)
X = df[['year']]
y = df['per capita income (US$)'] 

# Fit a linear regression model
model = LinearRegression()         # making the object 
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Calculate Mean Absolute Error
mae = mean_absolute_error(y, y_pred)
print(f'Mean Absolute Error: {mae:.2f}')

# Plot the regression line
plt.scatter(X, y)
plt.plot(X, y_pred, color='red')
plt.xlabel('Year')
plt.ylabel('Per Capita Income (US$)')
plt.legend(['org_data','predict_line'])
plt.title('Per Capita Income vs Year with Regression Line')
plt.show()

# Predict for the year 2020
year_2020 = pd.DataFrame({'year': [2020]})
income_2020 = model.predict(year_2020)
print(f'Predicted per capita income for the year 2020: ${income_2020[0]:,.2f}')
