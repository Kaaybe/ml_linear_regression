import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import requests

# --- Configuration and Setup ---
st.set_page_config(
    page_title="Wine Quality Regression App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Session State Initialization ---
if 'model' not in st.session_state:
    st.session_state.model = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'feature_cols' not in st.session_state:
    st.session_state.feature_cols = None
if 'target_col' not in st.session_state:
    st.session_state.target_col = None

@st.cache_data
def load_data_from_url(url, wine_type='red'):
    """Loads wine quality dataset from URL."""
    try:
        # Fetch the data
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Parse CSV
        df = pd.read_csv(StringIO(response.text), sep=';')
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_resource
def train_linear_model(X_train, y_train):
    """Trains the Linear Regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def calculate_metrics(model, X_test, y_test):
    """Calculate comprehensive metrics."""
    y_pred = model.predict(X_test)
    return {
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'y_pred': y_pred
    }

# --- Main Application ---
def main():
    st.title("🍷 Wine Quality Prediction - Linear Regression")
    st.markdown("Predict wine quality based on physicochemical properties using linear regression.")
    st.markdown("---")

    # --- Sidebar Configuration ---
    st.sidebar.header("⚙️ Configuration")
    
    # Dataset Selection
    st.sidebar.subheader("1. Dataset Selection")
    wine_type = st.sidebar.selectbox(
        "Wine Type",
        ["Red Wine", "White Wine"],
        help="Choose between red or white wine dataset"
    )
    
    # URL mapping
    wine_urls = {
        "Red Wine": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
        "White Wine": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
    }
    
    # Load data button
    if st.sidebar.button("📥 Load Dataset"):
        with st.spinner(f"Loading {wine_type} dataset..."):
            df = load_data_from_url(wine_urls[wine_type])
            if df is not None:
                st.session_state.data = df
                st.session_state.model = None  # Reset model when new data is loaded
                st.session_state.metrics = None
                st.success(f"✅ Successfully loaded {len(df)} samples!")
    
    # Check if data is loaded
    if st.session_state.data is None:
        st.info("👆 Click 'Load Dataset' in the sidebar to begin!")
        st.markdown("""
        ### About the Dataset
        The Wine Quality dataset contains physicochemical properties and quality ratings for Portuguese wines.
        
        **Features include:**
        - Fixed acidity
        - Volatile acidity
        - Citric acid
        - Residual sugar
        - Chlorides
        - Free sulfur dioxide
        - Total sulfur dioxide
        - Density
        - pH
        - Sulphates
        - Alcohol
        
        **Target:** Quality rating (0-10)
        """)
        return
    
    df = st.session_state.data
    
    # --- Model Configuration ---
    st.sidebar.subheader("2. Model Configuration")
    
    # Feature selection
    all_features = [col for col in df.columns if col != 'quality']
    selected_features = st.sidebar.multiselect(
        "Select Features",
        all_features,
        default=all_features[:5],
        help="Choose which features to use for prediction"
    )
    
    if len(selected_features) == 0:
        st.warning("⚠️ Please select at least one feature!")
        return
    
    # Train-test split
    test_size = st.sidebar.slider(
        "Test Set Size (%)",
        min_value=10,
        max_value=40,
        value=20,
        step=5
    ) / 100.0
    
    random_state = st.sidebar.number_input(
        "Random State",
        min_value=0,
        max_value=100,
        value=42,
        help="Set for reproducibility"
    )
    
    # --- Display Dataset Info ---
    st.header("1. 📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Samples", len(df))
    col2.metric("Features", len(df.columns) - 1)
    col3.metric("Selected Features", len(selected_features))
    col4.metric("Wine Type", wine_type.split()[0])
    
    with st.expander("View Dataset Sample"):
        st.dataframe(df.head(10), use_container_width=True)
    
    with st.expander("Statistical Summary"):
        st.dataframe(df[selected_features + ['quality']].describe(), use_container_width=True)
    
    # --- Feature Correlation ---
    with st.expander("Feature Correlation Heatmap"):
        fig, ax = plt.subplots(figsize=(10, 8))
        corr_matrix = df[selected_features + ['quality']].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        st.pyplot(fig)
    
    # --- Data Preparation ---
    X = df[selected_features]
    y = df['quality']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    st.write(f"**Data Split:** {len(X_train)} training samples, {len(X_test)} testing samples")
    
    # --- Model Training ---
    st.header("2. 🎯 Model Training")
    
    def handle_train_click():
        model = train_linear_model(X_train, y_train)
        metrics = calculate_metrics(model, X_test, y_test)
        
        # Store in session state
        st.session_state.model = model
        st.session_state.metrics = metrics
        st.session_state.feature_cols = selected_features
        st.session_state.target_col = 'quality'
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.button(
            "🚀 Train Model",
            on_click=handle_train_click,
            type="primary",
            use_container_width=True
        )
    
    # --- Model Results ---
    if st.session_state.model is not None:
        st.success("✅ Model trained successfully!")
        
        # Display Metrics
        st.subheader("Model Performance Metrics")
        metrics = st.session_state.metrics
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("R² Score", f"{metrics['r2']:.4f}")
        col2.metric("RMSE", f"{metrics['rmse']:.4f}")
        col3.metric("MSE", f"{metrics['mse']:.4f}")
        col4.metric("MAE", f"{metrics['mae']:.4f}")
        
        # Feature Importance
        st.subheader("Feature Coefficients")
        coef_df = pd.DataFrame({
            'Feature': st.session_state.feature_cols,
            'Coefficient': st.session_state.model.coef_
        }).sort_values('Coefficient', key=abs, ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['red' if x < 0 else 'green' for x in coef_df['Coefficient']]
        ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, alpha=0.7)
        ax.set_xlabel('Coefficient Value')
        ax.set_title('Feature Importance (Coefficient Values)')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        plt.tight_layout()
        st.pyplot(fig)
        
        # --- Predictions vs Actual ---
        st.header("3. 📈 Model Visualization")
        
        tab1, tab2 = st.tabs(["Predictions vs Actual", "Residual Plot"])
        
        with tab1:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(y_test, metrics['y_pred'], alpha=0.6, edgecolors='k', linewidth=0.5)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                   'r--', lw=2, label='Perfect Prediction')
            ax.set_xlabel('Actual Quality')
            ax.set_ylabel('Predicted Quality')
            ax.set_title('Predicted vs Actual Wine Quality')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
        
        with tab2:
            residuals = y_test - metrics['y_pred']
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(metrics['y_pred'], residuals, alpha=0.6, edgecolors='k', linewidth=0.5)
            ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
            ax.set_xlabel('Predicted Quality')
            ax.set_ylabel('Residuals')
            ax.set_title('Residual Plot')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
        
        # --- Prediction Widget ---
        st.header("4. 🔮 Make Predictions")
        st.markdown("Adjust the feature values to predict wine quality:")
        
        # Create input fields for each feature
        input_data = {}
        cols = st.columns(3)
        
        for idx, feature in enumerate(st.session_state.feature_cols):
            col_idx = idx % 3
            with cols[col_idx]:
                min_val = float(df[feature].min())
                max_val = float(df[feature].max())
                mean_val = float(df[feature].mean())
                
                input_data[feature] = st.number_input(
                    feature.replace('_', ' ').title(),
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    step=(max_val - min_val) / 100,
                    format="%.2f"
                )
        
        # Make prediction
        input_df = pd.DataFrame([input_data])
        prediction = st.session_state.model.predict(input_df)[0]
        
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px;'>
                <h2 style='color: #0e1117;'>Predicted Quality</h2>
                <h1 style='color: #ff4b4b; font-size: 60px;'>{prediction:.2f}</h1>
                <p style='color: #0e1117;'>Out of 10</p>
            </div>
            """, unsafe_allow_html=True)
    
    else:
        st.info("👆 Click 'Train Model' to build and evaluate the regression model.")

if __name__ == "__main__":
    main()
