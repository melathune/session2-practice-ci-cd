import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Load training data
data = pd.read_csv('training_data.csv')

X = data[['YearsExperience']]
y = data['Salary']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, 'linear_model.pkl')

# Save coefficients
with open('linear_model.txt', 'w') as f:
    f.write(f'Coefficients: {model.coef_}\n')
    f.write(f'Intercept: {model.intercept_}\n')

print("Model files generated.")