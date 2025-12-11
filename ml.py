import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# --- Configuration and Setup ---
st.set_page_config(
    page_title="Streamlit Linear Regression App",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def generate_data(n_samples=100, slope=2, intercept=5, noise_level=10):
    """Generates synthetic linear data."""
    np.random.seed(42)
    X = np.linspace(0, 100, n_samples).reshape(-1, 1)
    y = slope * X + intercept + np.random.randn(n_samples, 1) * noise_level
    df = pd.DataFrame(X, columns=['Feature_X'])
    df['Target_Y'] = y
    return df

@st.cache_resource
def train_model(X_train, y_train):
    """Trains the Linear Regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# --- Main Application Function ---
def main():
    st.title("Interactive Linear Regression Model")
    st.markdown("---")

    # --- Sidebar for Model Tuning and Data Generation ---
    st.sidebar.header("Data & Model Configuration")

    # Data generation parameters
    st.sidebar.subheader("1. Data Generation")
    n_samples = st.sidebar.slider("Number of Samples", 50, 500, 100)
    slope = st.sidebar.slider("True Slope", 0.5, 5.0, 2.0, 0.1)
    intercept = st.sidebar.slider("True Intercept", -10.0, 10.0, 5.0, 0.5)
    noise_level = st.sidebar.slider("Noise Level", 1, 50, 10)
    test_size = st.sidebar.slider("Test Set Size (%)", 10, 50, 20) / 100.0

    df = generate_data(n_samples, slope, intercept, noise_level)

    # --- Data Preparation ---
    X = df[['Feature_X']]
    y = df['Target_Y']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # --- Main Content Area ---
    
    st.header("1. Generated Data")
    st.dataframe(df.head())
    
    st.write(f"Data Split: {len(X_train)} training samples, {len(X_test)} testing samples.")

    # --- Model Training ---
    st.header("2. Model Training")
    if st.button("Train Linear Regression Model"):
        # Train the model (using st.cache_resource to prevent re-training on every interaction)
        model = train_model(X_train, y_train)
        
        # --- Model Evaluation ---
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        if st.button("Train Linear Regression Model"):
    # ... Training and Evaluation code is here ...
    model = train_model(X_train, y_train)

    col1.metric("Model Slope (Coefficient)", f"{model.coef_[0][0]:.3f}")
    
        col1, col2, col3 = st.columns(3)
        col1.metric("Model Slope (Coefficient)", f"{model.coef_[0][0]:.3f}")
        col2.metric("Mean Squared Error (MSE)", f"{mse:.3f}")
        col3.metric("R-squared (R²)", f"{r2:.3f}")

        # --- Visualization ---
        st.header("3. Model Visualization")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot training data
        ax.scatter(X_train, y_train, color='blue', label='Training Data')
        
        # Plot testing data
        ax.scatter(X_test, y_test, color='green', label='Test Data (Actual)')
        
        # Plot regression line
        ax.plot(X, model.predict(X), color='red', linewidth=2, label='Regression Line')
        
        ax.set_xlabel('Feature X')
        ax.set_ylabel('Target Y')
        ax.set_title('Linear Regression Fit')
        ax.legend()
        st.pyplot(fig)
        
        # --- Prediction Widget ---
        st.markdown("---")
        st.header("4. Make a Prediction")
        
        # Input for prediction
        new_x = st.slider("Select X value for prediction", 0, 100, 50)
        
        # Predict based on the slider value
        prediction = model.predict(np.array([[new_x]]))[0][0]
        st.info(f"For X = **{new_x}**, the predicted Y value is **{prediction:.3f}**")

    else:
        st.info("Click the 'Train Linear Regression Model' button above to start the analysis.")


if __name__ == "__main__":
    main()
