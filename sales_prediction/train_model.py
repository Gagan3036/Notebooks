import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Import data
salary = pd.read_csv('https://github.com/YBI-Foundation/Dataset/raw/main/Salary%20Data.csv')

# Define features (X) and target (y)
y = salary['Salary']
X = salary[['Experience Years']]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2529)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model to a file
with open('salary_prediction_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Evaluate the model
y_pred = model.predict(X_test)
from sklearn.metrics import mean_absolute_percentage_error

mse = mean_absolute_percentage_error(y_test, y_pred)

print(f'Mean Absolute Percentage Error: {mse}')
