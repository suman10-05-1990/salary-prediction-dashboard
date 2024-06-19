import streamlit as st
import pandas as pd
import numpy as np
from faker import Faker
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Initialize Faker
fake = Faker()

# Set seed for reproducibility
Faker.seed(42)
np.random.seed(42)
random.seed(42)

# Define constants
num_entries = 200000
min_age = 22
max_age = 65
min_experience = 0
max_experience = 43
min_salary = 30000
max_salary = 200000

# Generate the dataset
ages = np.random.randint(min_age, max_age, size=num_entries)
data = {
    "NAME": [fake.name() for _ in range(num_entries)],
    "AGE": ages,
    "YEARS_OF_EXPERIENCE": [random.randint(min_experience, min(age - 22, max_experience)) for age in ages],
    "CURRENT_SALARY": np.random.randint(min_salary, max_salary, size=num_entries)
}

# Create DataFrame
hr_dataset = pd.DataFrame(data)

# Drop NAME as it is not relevant for prediction
hr_dataset = hr_dataset.drop(columns=["NAME"])

# Save the raw dataset
hr_dataset.to_csv("hr_dataset_raw.csv", index=False)

# Clean the dataset (assuming the generated data is already clean)
hr_dataset_cleaned = hr_dataset.copy()

# Save the cleaned dataset
hr_dataset_cleaned.to_csv("hr_dataset_cleaned.csv", index=False)

# Split the data into training and testing sets
X = hr_dataset_cleaned[["AGE", "YEARS_OF_EXPERIENCE"]]
y = hr_dataset_cleaned["CURRENT_SALARY"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

# Train a Random Forest model
model_rf = RandomForestRegressor(random_state=42)
model_rf.fit(X_train, y_train)

# Save the Random Forest model as the best model
joblib.dump(model_rf, "best_salary_prediction_model.pkl")

# Predict on the test set for Linear Regression
y_pred_lr = model_lr.predict(X_test)

# Predict on the test set for Random Forest
y_pred_rf = model_rf.predict(X_test)

# Calculate performance metrics for Linear Regression
mse_lr = mean_squared_error(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

# Calculate performance metrics for Random Forest
mse_rf = mean_squared_error(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Function to predict salary for a new entry
def predict_salary(age, years_of_experience):
    # Create a DataFrame with the same feature names as the training data
    input_data = pd.DataFrame({'AGE': [age], 'YEARS_OF_EXPERIENCE': [years_of_experience]})
    return best_rf.predict(input_data)[0]

# Streamlit Dashboard
st.title("HR Salary Prediction Dashboard")

# Sidebar for user input
st.sidebar.title("Predicted Salary")
new_age = st.sidebar.slider('Age', min_value=min_age, max_value=max_age, value=min_age)
new_experience = st.sidebar.slider('Years of Experience', min_value=min_experience, max_value=max_experience, value=min_experience)

# Check if the model file exists
if os.path.exists("best_salary_prediction_model.pkl"):
    # Load the best Random Forest model
    try:
        best_rf = joblib.load("best_salary_prediction_model.pkl")
        predicted_salary = predict_salary(new_age, new_experience)
        st.sidebar.write(f"Predicted Salary: ${predicted_salary:,.2f}")
    except EOFError:
        st.sidebar.write("Error loading the model. Please re-train the model.")
else:
    st.sidebar.write("Model file not found. Please re-train the model.")

# Data Visualization Section
st.header("Data Visualization")

# Histogram for Age
st.subheader("Histogram for Age")
fig1, ax1 = plt.subplots()
sns.histplot(hr_dataset_cleaned["AGE"], kde=True, ax=ax1)
st.pyplot(fig1)

# Histogram for Years of Experience
st.subheader("Histogram for Years of Experience")
fig2, ax2 = plt.subplots()
sns.histplot(hr_dataset_cleaned["YEARS_OF_EXPERIENCE"], kde=True, ax=ax2)
st.pyplot(fig2)

# Scatter Plot for Age vs Years of Experience
st.subheader("Scatter Plot for Age vs Years of Experience")
fig3, ax3 = plt.subplots()
sns.scatterplot(data=hr_dataset_cleaned, x="AGE", y="YEARS_OF_EXPERIENCE", ax=ax3)
st.pyplot(fig3)

# Scatter Plot for Actual vs Predicted Salary
st.subheader("Scatter Plot for Actual vs Predicted Salary (Random Forest)")
fig4, ax4 = plt.subplots()
sns.scatterplot(x=y_test, y=y_pred_rf, ax=ax4)
ax4.set_xlabel("Actual Salary")
ax4.set_ylabel("Predicted Salary")
ax4.set_title("Actual vs Predicted Salary (Random Forest)")
st.pyplot(fig4)

# Performance Comparison
st.header("Model Performance Comparison")
st.write("### Linear Regression")
st.write(f"Mean Squared Error: {mse_lr:.2f}")
st.write(f"Mean Absolute Error: {mae_lr:.2f}")
st.write(f"R-squared: {r2_lr:.2f}")

st.write("### Random Forest")
st.write(f"Mean Squared Error: {mse_rf:.2f}")
st.write(f"Mean Absolute Error: {mae_rf:.2f}")
st.write(f"R-squared: {r2_rf:.2f}")

# Line Chart for Predicted Salaries vs Actual Salaries
st.subheader("Line Chart for Predicted Salaries vs Actual Salaries")
predicted_salaries_df = pd.DataFrame({
    'Actual Salary': y_test,
    'Predicted Salary (RF)': y_pred_rf
})
st.line_chart(predicted_salaries_df)