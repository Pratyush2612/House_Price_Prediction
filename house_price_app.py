import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Set page configuration
st.set_page_config(page_title="House Price Prediction", layout="wide")

# Set random seed for reproducibility
np.random.seed(42)

# Title and description
st.title("🏠 House Price Prediction with California Housing Dataset")
st.write("""
This app predicts house prices based on features like median income, house age, and number of rooms.
Explore the dataset, visualize distributions, and predict prices using a Linear Regression model.
""")

# Load and preprocess data
@st.cache_data
def load_and_preprocess_data():
    housing = fetch_california_housing()
    X = housing.data
    y = housing.target
    feature_names = housing.feature_names
    df = pd.DataFrame(X, columns=feature_names)
    df['MedHouseVal'] = y  # Target is median house value in $100,000s
    return df, X, y, feature_names

# Load data
df, X, y, feature_names = load_and_preprocess_data()

# Sidebar for navigation
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Choose a section", ["Data Exploration", "Model Prediction"])

# Data Exploration Section
if page == "Data Exploration":
    st.header("Data Exploration")
    st.write("View the California Housing dataset and its distributions.")

    # Show dataset
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Missing values
    st.subheader("Missing Values")
    st.write("Missing values in each column:")
    st.write(df.isnull().sum())

    # Basic statistics
    st.subheader("Dataset Statistics")
    st.write(df.describe())

    # Feature distributions
    st.subheader("Feature Distributions")
    fig, ax = plt.subplots(figsize=(15, 10))
    for i, feature in enumerate(feature_names):
        plt.subplot(3, 3, i+1)
        sns.histplot(df[feature], kde=True)
        plt.title(feature)
    plt.tight_layout()
    st.pyplot(fig)

    # Price distribution
    st.subheader("House Price Distribution")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df['MedHouseVal'], kde=True)
    ax.set_title('Distribution of Median House Values')
    ax.set_xlabel('Price ($100,000s)')
    st.pyplot(fig)

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    ax.set_title('Correlation Heatmap of Features and Price')
    st.pyplot(fig)

# Model Prediction Section
elif page == "Model Prediction":
    st.header("Model Prediction")
    st.write("Train a Linear Regression model and predict house prices.")

    # Split and preprocess data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    st.subheader("Model Performance")
    st.write(f"Mean Squared Error (MSE) on Test Set: {mse:.2f}")

    # Plot actual vs predicted
    st.subheader("Actual vs Predicted Prices")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_test, y_pred, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel('Actual Price ($100,000s)')
    ax.set_ylabel('Predicted Price ($100,000s)')
    ax.set_title('Actual vs Predicted House Prices')
    st.pyplot(fig)

    # Feature importance
    st.subheader("Feature Importance")
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_
    })
    feature_importance['Absolute Coefficient'] = feature_importance['Coefficient'].abs()
    feature_importance = feature_importance.sort_values(by='Absolute Coefficient', ascending=False)
    st.dataframe(feature_importance[['Feature', 'Coefficient']])

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Absolute Coefficient', y='Feature', data=feature_importance)
    ax.set_title('Feature Importance (Absolute Coefficients)')
    st.pyplot(fig)

    # User input for prediction
    st.subheader("Predict House Price")
    st.write("Enter feature values to predict the house price (in $100,000s):")
    user_input = {}
    for feature in feature_names:
        # Use dataset mean as default value
        default_value = float(df[feature].mean())
        user_input[feature] = st.number_input(f"{feature}", value=default_value, step=0.1)

    # Predict with user input
    if st.button("Predict"):
        input_array = np.array([user_input[feature] for feature in feature_names]).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)[0]
        st.success(f"Predicted House Price: ${prediction*100:.2f}K")