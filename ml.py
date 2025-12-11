# Wine Quality Prediction - Complete Setup Guide

## 🍷 Dataset Information
- **Source**: UCI Machine Learning Repository - Wine Quality Dataset
- **Task**: Predict wine quality (0-10 scale) based on physicochemical properties
- **Features**: 11 chemical properties (acidity, pH, alcohol, sulfates, etc.)
- **Samples**: 4,898 white wine samples

---

## 📋 Step 1: Google Colab Setup

### Cell 1: Install Dependencies & Configure Git
```python
# Install required packages
!pip install streamlit pandas scikit-learn matplotlib seaborn plotly ucimlrepo -q

### Cell 2: Create Streamlit App File
```python
%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Page Configuration
st.set_page_config(
    page_title="🍷 Wine Quality Predictor",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    h1 {
        color: #722F37;
        padding-bottom: 1rem;
    }
    h2 {
        color: #B4656F;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and Description
st.title("🍷 Wine Quality Prediction System")
st.markdown("""
    This machine learning application predicts wine quality based on physicochemical properties.
    Upload your own dataset or use the built-in UCI Wine Quality dataset.
""")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    data_source = st.radio(
        "Select Data Source:",
        ["UCI Wine Quality Dataset", "Upload Custom CSV"],
        help="Choose between the built-in dataset or upload your own"
    )
    
    st.markdown("---")
    
    model_choice = st.selectbox(
        "Choose ML Model:",
        ["Linear Regression", "Random Forest", "Gradient Boosting"],
        help="Different models offer varying complexity and accuracy"
    )
    
    if model_choice == "Random Forest":
        n_estimators = st.slider("Number of Trees:", 10, 200, 100, 10)
        max_depth = st.slider("Max Depth:", 3, 30, 10)
    elif model_choice == "Gradient Boosting":
        n_estimators = st.slider("Number of Estimators:", 10, 200, 100, 10)
        learning_rate = st.slider("Learning Rate:", 0.01, 0.3, 0.1, 0.01)
    
    st.markdown("---")
    
    test_size = st.slider("Test Set Size:", 0.1, 0.4, 0.2, 0.05)
    
    st.markdown("---")
    st.markdown("### 📊 About")
    st.info("""
        **Dataset:** UCI Wine Quality
        
        **Features:** 11 physicochemical properties
        
        **Target:** Quality score (0-10)
        
        **Source:** [UCI ML Repository](https://archive.ics.uci.edu/)
    """)

# Load Data Function
@st.cache_data
def load_uci_wine_data():
    """Load the UCI Wine Quality dataset"""
    try:
        from ucimlrepo import fetch_ucirepo
        wine_quality = fetch_ucirepo(id=186)
        
        # Combine features and target
        X = wine_quality.data.features
        y = wine_quality.data.targets
        
        df = pd.concat([X, y], axis=1)
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        return df
    except Exception as e:
        st.error(f"Error loading UCI dataset: {str(e)}")
        return None

# Main Content
if data_source == "UCI Wine Quality Dataset":
    with st.spinner("Loading UCI Wine Quality dataset..."):
        df = load_uci_wine_data()
    
    if df is not None:
        st.success("✅ Successfully loaded UCI Wine Quality dataset!")
else:
    uploaded_file = st.file_uploader(
        "Upload your CSV file (last column should be the target)",
        type="csv",
        help="Ensure numeric features only, with target variable as the last column"
    )
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
            st.success("✅ Successfully uploaded custom dataset!")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            df = None
    else:
        df = None
        st.info("👆 Please upload a CSV file to continue")

# Process Data if Available
if df is not None:
    
    # Data Overview Section
    st.header("📊 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Samples", f"{len(df):,}")
    with col2:
        st.metric("Features", len(df.columns) - 1)
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    with col4:
        st.metric("Avg Quality", f"{df.iloc[:, -1].mean():.2f}")
    
    # Show dataset preview
    with st.expander("🔍 View Dataset Preview"):
        st.dataframe(df.head(10), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Statistical Summary")
            st.dataframe(df.describe(), use_container_width=True)
        with col2:
            st.subheader("Data Info")
            st.write(f"**Shape:** {df.shape}")
            st.write(f"**Columns:** {list(df.columns)}")
            st.write(f"**Data Types:**")
            st.write(df.dtypes)
    
    # Prepare Features and Target
    X = df.iloc[:, :-1].select_dtypes(include=[np.number])
    y = df.iloc[:, -1]
    
    # Handle missing values
    if X.isnull().sum().sum() > 0:
        imputer = SimpleImputer(strategy='mean')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        st.warning("⚠️ Missing values detected and filled with column means")
    
    # Data Visualization Section
    st.header("📈 Data Visualization")
    
    viz_tabs = st.tabs(["Distribution", "Correlations", "Feature Importance"])
    
    with viz_tabs[0]:
        col1, col2 = st.columns(2)
        
        with col1:
            # Target distribution
            fig = px.histogram(
                df, 
                x=df.columns[-1],
                title="Target Variable Distribution",
                nbins=30,
                color_discrete_sequence=['#722F37']
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Feature distribution
            selected_feature = st.selectbox("Select Feature:", X.columns)
            fig = px.box(
                df,
                y=selected_feature,
                title=f"{selected_feature.replace('_', ' ').title()} Distribution",
                color_discrete_sequence=['#B4656F']
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with viz_tabs[1]:
        # Correlation heatmap
        corr_matrix = df.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 8}
        ))
        
        fig.update_layout(
            title="Feature Correlation Matrix",
            height=600,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Top correlations with target
        target_corr = corr_matrix.iloc[-1, :-1].abs().sort_values(ascending=False)
        st.subheader("Top Features by Correlation with Target")
        
        fig = px.bar(
            x=target_corr.values,
            y=target_corr.index,
            orientation='h',
            title="Absolute Correlation with Target",
            color=target_corr.values,
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with viz_tabs[2]:
        st.info("Feature importance will be calculated after model training")
    
    # Model Training Section
    st.header("🤖 Model Training & Evaluation")
    
    if st.button("🚀 Train Model", type="primary", use_container_width=True):
        with st.spinner("Training model..."):
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            if model_choice == "Linear Regression":
                model = LinearRegression()
            elif model_choice == "Random Forest":
                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42,
                    n_jobs=-1
                )
            else:  # Gradient Boosting
                model = GradientBoostingRegressor(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    random_state=42
                )
            
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train_scaled, y_train, cv=5, scoring='r2'
            )
            
            st.success("✅ Model trained successfully!")
            
            # Display metrics
            st.subheader("📊 Model Performance")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### Training Set")
                st.metric("R² Score", f"{train_r2:.4f}")
                st.metric("MAE", f"{train_mae:.4f}")
                st.metric("RMSE", f"{train_rmse:.4f}")
            
            with col2:
                st.markdown("### Test Set")
                st.metric("R² Score", f"{test_r2:.4f}", 
                         delta=f"{test_r2 - train_r2:.4f}")
                st.metric("MAE", f"{test_mae:.4f}",
                         delta=f"{test_mae - train_mae:.4f}",
                         delta_color="inverse")
                st.metric("RMSE", f"{test_rmse:.4f}",
                         delta=f"{test_rmse - train_rmse:.4f}",
                         delta_color="inverse")
            
            with col3:
                st.markdown("### Cross-Validation")
                st.metric("Mean CV R²", f"{cv_scores.mean():.4f}")
                st.metric("Std CV R²", f"{cv_scores.std():.4f}")
                st.write("")
                st.write("")
            
            # Prediction vs Actual plots
            st.subheader("📉 Prediction Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.scatter(
                    x=y_test,
                    y=y_test_pred,
                    labels={'x': 'Actual Quality', 'y': 'Predicted Quality'},
                    title="Test Set: Actual vs Predicted",
                    trendline="ols",
                    color_discrete_sequence=['#722F37']
                )
                fig.add_shape(
                    type="line", line=dict(dash='dash', color='gray'),
                    x0=y_test.min(), y0=y_test.min(),
                    x1=y_test.max(), y1=y_test.max()
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                residuals = y_test - y_test_pred
                fig = px.scatter(
                    x=y_test_pred,
                    y=residuals,
                    labels={'x': 'Predicted Quality', 'y': 'Residuals'},
                    title="Residual Plot",
                    color_discrete_sequence=['#B4656F']
                )
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig, use_container_width=True)
            
            # Feature Importance
            if hasattr(model, 'feature_importances_'):
                st.subheader("🎯 Feature Importance")
                
                importance_df = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=True)
                
                fig = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title=f"{model_choice} Feature Importance",
                    color='Importance',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Prediction Interface
            st.header("🔮 Make New Predictions")
            st.markdown("Adjust the sliders below to predict wine quality:")
            
            with st.form("prediction_form"):
                input_cols = st.columns(3)
                user_inputs = []
                
                for idx, col in enumerate(X.columns):
                    with input_cols[idx % 3]:
                        col_mean = float(X[col].mean())
                        col_min = float(X[col].min())
                        col_max = float(X[col].max())
                        
                        value = st.slider(
                            label=col.replace('_', ' ').title(),
                            min_value=col_min,
                            max_value=col_max,
                            value=col_mean,
                            step=(col_max - col_min) / 100
                        )
                        user_inputs.append(value)
                
                submit_button = st.form_submit_button(
                    "🍷 Predict Quality",
                    type="primary",
                    use_container_width=True
                )
                
                if submit_button:
                    # Scale input
                    input_scaled = scaler.transform([user_inputs])
                    prediction = model.predict(input_scaled)[0]
                    
                    # Display prediction
                    st.balloons()
                    
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.markdown("### 🎯 Prediction Result")
                        st.markdown(f"""
                            <div style='text-align: center; padding: 2rem; 
                                 background: linear-gradient(135deg, #722F37 0%, #B4656F 100%); 
                                 border-radius: 1rem; color: white;'>
                                <h1 style='font-size: 4rem; margin: 0; color: white;'>
                                    {prediction:.2f}
                                </h1>
                                <h3 style='margin: 0.5rem 0 0 0; color: white;'>
                                    Predicted Wine Quality
                                </h3>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Quality interpretation
                        if prediction >= 7:
                            quality_text = "🌟 Excellent Quality Wine"
                            color = "#2ecc71"
                        elif prediction >= 6:
                            quality_text = "✨ Good Quality Wine"
                            color = "#3498db"
                        elif prediction >= 5:
                            quality_text = "👍 Average Quality Wine"
                            color = "#f39c12"
                        else:
                            quality_text = "⚠️ Below Average Quality"
                            color = "#e74c3c"
                        
                        st.markdown(f"""
                            <p style='text-align: center; font-size: 1.5rem; 
                                 color: {color}; margin-top: 1rem; font-weight: bold;'>
                                {quality_text}
                            </p>
                        """, unsafe_allow_html=True)
            
            # Store model in session state for reuse
            st.session_state['trained_model'] = model
            st.session_state['scaler'] = scaler

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>🍷 Wine Quality Prediction System | Built with Streamlit & Scikit-learn</p>
        <p>Dataset: UCI Machine Learning Repository | 
        <a href='https://archive.ics.uci.edu/dataset/186/wine+quality' target='_blank'>
        Learn More</a></p>
    </div>
""", unsafe_allow_html=True)
```

### Cell 3: Create Requirements File
```python
%%writefile requirements.txt
streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
plotly
ucimlrepo
```

### Cell 4: Test the App Locally (Optional)
```python
# This will open a tunnel to test your app in Colab
!streamlit run app.py & npx localtunnel --port 8501
```

---

## 🚀 Step 2: Deploy to GitHub

### Cell 5: Initialize and Push to GitHub
```python
# Create a new repository on GitHub first, then run:

!git init
!git add app.py requirements.txt
!git commit -m "Initial commit: Wine Quality ML App"

# Replace YOUR_USERNAME and YOUR_REPO with your GitHub info
!git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
!git branch -M main
!git push -u origin main
```

**If you need authentication:**
```python
# Generate a Personal Access Token on GitHub first
# Settings > Developer settings > Personal access tokens > Generate new token

# Then use it in place of password when prompted
```

---

## 🌐 Step 3: Deploy on Streamlit Cloud

1. **Go to**: [share.streamlit.io](https://share.streamlit.io)

2. **Click**: "New app"

3. **Fill in**:
   - **Repository**: `YOUR_USERNAME/YOUR_REPO`
   - **Branch**: `main`
   - **Main file path**: `app.py`

4. **Click**: "Deploy"

5. **Wait**: 2-5 minutes for deployment

6. **Access**: Your app at `https://YOUR_USERNAME-YOUR_REPO.streamlit.app`

---

## ✨ Features of This App

### 📊 Data Handling
- Load UCI Wine Quality dataset automatically
- Upload custom CSV files
- Handle missing values automatically
- Data preprocessing and scaling

### 🤖 Machine Learning
- Three model options:
  - Linear Regression (baseline)
  - Random Forest (ensemble method)
  - Gradient Boosting (advanced)
- Hyperparameter tuning via UI
- Train/test split customization
- Cross-validation scoring

### 📈 Visualizations
- Target distribution histogram
- Feature distribution box plots
- Correlation heatmap
- Feature importance charts
- Actual vs Predicted scatter plots
- Residual analysis plots

### 🔮 Predictions
- Interactive sliders for all features
- Real-time prediction
- Quality interpretation
- Visual feedback

### 📱 User Experience
- Professional color scheme (wine colors!)
- Responsive layout
- Loading indicators
- Error handling
- Helpful tooltips
- Celebration effects (balloons!)

---

## 🎯 Dataset Details

### Wine Quality Dataset (UCI)
- **Samples**: 4,898 white wine samples
- **Features**: 11 physicochemical properties
- **Target**: Quality score (0-10)

### Features:
1. Fixed acidity
2. Volatile acidity
3. Citric acid
4. Residual sugar
5. Chlorides
6. Free sulfur dioxide
7. Total sulfur dioxide
8. Density
9. pH
10. Sulphates
11. Alcohol

---

## 🔧 Troubleshooting

### Common Issues:

**1. Module not found error**
- Make sure `requirements.txt` is complete
- Streamlit Cloud will auto-install packages

**2. Git authentication failed**
- Use Personal Access Token instead of password
- Generate token: GitHub Settings > Developer settings > Personal access tokens

**3. App won't start**
- Check logs in Streamlit Cloud dashboard
- Verify file paths are correct

**4. Dataset not loading**
- Check internet connection
- UCI API might be temporarily down
- Use custom CSV upload as backup

---

## 📝 Notes

- The app uses the **white wine** variant of the UCI dataset (4,898 samples)
- All models include proper train/test split to prevent overfitting
- Feature scaling is applied for better model performance
- Cross-validation ensures robust performance estimation
- The app is fully self-contained and production-ready

---

## 🎨 Customization Tips

1. **Change colors**: Modify the CSS in the `st.markdown()` section
2. **Add models**: Import from sklearn and add to model_choice
3. **More features**: Add feature engineering in the data processing section
4. **Different datasets**: Upload any CSV with numeric columns

Enjoy your ML app! 🍷✨
