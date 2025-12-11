import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import warnings
import io
warnings.filterwarnings('ignore')

# Configuration
APP_CONFIG = {
    "title": "🔬 Breast Cancer Wisconsin Diagnostic Analyzer",
    "version": "3.0",
    "description": "AI-Powered Analysis of Breast Cancer Diagnostic Features"
}

# Custom CSS for better styling
def load_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
        text-align: center;
    }
    
    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.2);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .metric-card h3 {
        font-size: 0.9rem;
        font-weight: 500;
        margin: 0 0 0.5rem 0;
        opacity: 0.9;
    }
    
    .metric-card h2 {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }
    
    .info-box {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #0ea5e9;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .success-box {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #22c55e;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 55px;
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border-radius: 10px;
        padding: 0 28px;
        font-weight: 600;
        font-size: 0.95rem;
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(135deg, #e2e8f0 0%, #cbd5e1 100%);
        border-color: #667eea;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: 2px solid #5a67d8;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.6rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(102, 126, 234, 0.4);
    }
    
    .data-source-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .data-source-card:hover {
        border-color: #667eea;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.1);
    }
    
    .section-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e293b;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }
    
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    }
    
    .sidebar-section {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

def load_dataset_from_url():
    """Load the Breast Cancer Wisconsin dataset directly from UCI"""
    try:
        # Direct download URL for the dataset
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
        
        # Column names based on dataset documentation
        column_names = ['ID', 'Diagnosis'] + [
            f'{feature}_{stat}' 
            for feature in ['radius', 'texture', 'perimeter', 'area', 'smoothness', 
                          'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension']
            for stat in ['mean', 'se', 'worst']
        ]
        
        df = pd.read_csv(url, header=None, names=column_names)
        return df
    except Exception as e:
        st.error(f"Error loading dataset from URL: {e}")
        return None

def load_default_dataset():
    """Load the Breast Cancer Wisconsin dataset using ucimlrepo"""
    try:
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=17)
        X = dataset.data.features
        y = dataset.data.targets
        df = pd.concat([X, y], axis=1)
        return df
    except:
        # Fallback to direct URL download
        return load_dataset_from_url()

def load_dataset_from_custom_url(url):
    """Load dataset from a custom URL provided by user"""
    try:
        # Try different common formats
        df = None
        error_messages = []
        
        # Try CSV
        try:
            df = pd.read_csv(url)
            return df
        except Exception as e:
            error_messages.append(f"CSV: {str(e)}")
        
        # Try with different separators
        try:
            df = pd.read_csv(url, sep='\t')
            return df
        except Exception as e:
            error_messages.append(f"TSV: {str(e)}")
        
        # Try Excel
        try:
            df = pd.read_excel(url)
            return df
        except Exception as e:
            error_messages.append(f"Excel: {str(e)}")
        
        raise Exception("\n".join(error_messages))
        
    except Exception as e:
        st.error(f"Unable to load dataset from URL. Errors encountered:\n{e}")
        return None

def get_feature_info():
    """Return information about features in the dataset"""
    return {
        "radius": "Mean of distances from center to points on the perimeter",
        "texture": "Standard deviation of gray-scale values",
        "perimeter": "Perimeter of the cell nucleus",
        "area": "Area of the cell nucleus",
        "smoothness": "Local variation in radius lengths",
        "compactness": "Perimeter² / area - 1.0",
        "concavity": "Severity of concave portions of the contour",
        "concave_points": "Number of concave portions of the contour",
        "symmetry": "Symmetry of the cell nucleus",
        "fractal_dimension": "Coastline approximation - 1"
    }

def display_dataset_info():
    """Display information about the dataset"""
    st.markdown("""
    <div class="info-box">
        <h3>🔬 About the Breast Cancer Wisconsin Dataset</h3>
        <p><strong>Source:</strong> UCI Machine Learning Repository</p>
        <p><strong>Instances:</strong> 569 samples</p>
        <p><strong>Features:</strong> 30 numeric features (mean, SE, and worst values for 10 measurements)</p>
        <p><strong>Target:</strong> Diagnosis (B = Benign, M = Malignant)</p>
        <p><strong>Purpose:</strong> Diagnostic prediction based on fine needle aspirate (FNA) images of breast masses</p>
    </div>
    """, unsafe_allow_html=True)

def perform_eda(df):
    """Perform comprehensive exploratory data analysis"""
    
    st.markdown('<h2 class="section-header">📊 Exploratory Data Analysis</h2>', unsafe_allow_html=True)
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview", 
        "📉 Distributions", 
        "🔗 Correlations", 
        "📦 Box Plots",
        "🎯 Feature Importance"
    ])
    
    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>Total Samples</h3>
                <h2>{}</h2>
            </div>
            """.format(len(df)), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>Features</h3>
                <h2>{}</h2>
            </div>
            """.format(len(df.columns)-1 if 'Diagnosis' in df.columns else len(df.columns)), unsafe_allow_html=True)
        
        with col3:
            if 'Diagnosis' in df.columns:
                benign_count = (df['Diagnosis'] == 'B').sum()
                st.markdown("""
                <div class="metric-card">
                    <h3>Benign (B)</h3>
                    <h2>{}</h2>
                </div>
                """.format(benign_count), unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="metric-card">
                    <h3>Missing Values</h3>
                    <h2>{}</h2>
                </div>
                """.format(df.isnull().sum().sum()), unsafe_allow_html=True)
        
        with col4:
            if 'Diagnosis' in df.columns:
                malignant_count = (df['Diagnosis'] == 'M').sum()
                st.markdown("""
                <div class="metric-card">
                    <h3>Malignant (M)</h3>
                    <h2>{}</h2>
                </div>
                """.format(malignant_count), unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="metric-card">
                    <h3>Numeric Cols</h3>
                    <h2>{}</h2>
                </div>
                """.format(len(df.select_dtypes(include=[np.number]).columns)), unsafe_allow_html=True)
        
        st.markdown("### 📋 Data Sample")
        st.dataframe(df.head(15), use_container_width=True, height=400)
        
        st.markdown("### 📊 Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
        
        if 'Diagnosis' in df.columns:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### 🥧 Diagnosis Distribution")
                fig = px.pie(df, names='Diagnosis', 
                            title='Distribution of Diagnosis',
                            color='Diagnosis',
                            color_discrete_map={'B':'#22c55e', 'M':'#ef4444'},
                            hole=0.4)
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### 📊 Count by Diagnosis")
                diagnosis_counts = df['Diagnosis'].value_counts().reset_index()
                diagnosis_counts.columns = ['Diagnosis', 'Count']
                fig = px.bar(diagnosis_counts, x='Diagnosis', y='Count',
                           color='Diagnosis',
                           color_discrete_map={'B':'#22c55e', 'M':'#ef4444'},
                           text='Count')
                fig.update_traces(textposition='outside')
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### 📊 Feature Distributions")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            col1, col2 = st.columns([2, 1])
            with col1:
                selected_features = st.multiselect(
                    "Select features to visualize:",
                    numeric_cols,
                    default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
                )
            with col2:
                chart_type = st.radio("Chart type:", ["Histogram", "Violin Plot", "Both"])
            
            if selected_features:
                for feature in selected_features:
                    if chart_type == "Both":
                        fig = make_subplots(
                            rows=1, cols=2,
                            subplot_titles=(f'{feature} - Distribution', f'{feature} - Box Plot')
                        )
                        
                        fig.add_trace(
                            go.Histogram(x=df[feature], name=feature, nbinsx=30, 
                                       marker_color='#667eea'),
                            row=1, col=1
                        )
                        
                        fig.add_trace(
                            go.Box(y=df[feature], name=feature, marker_color='#764ba2'),
                            row=1, col=2
                        )
                        
                        fig.update_layout(height=350, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                    elif chart_type == "Histogram":
                        fig = px.histogram(df, x=feature, nbins=30, 
                                         title=f'{feature} Distribution',
                                         color_discrete_sequence=['#667eea'])
                        st.plotly_chart(fig, use_container_width=True)
                    else:  # Violin Plot
                        fig = px.violin(df, y=feature, box=True, 
                                      title=f'{feature} Distribution',
                                      color_discrete_sequence=['#764ba2'])
                        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### 🔗 Correlation Analysis")
        numeric_df = df.select_dtypes(include=[np.number])
        
        if not numeric_df.empty:
            col1, col2 = st.columns([3, 1])
            
            with col2:
                st.markdown("#### Settings")
                show_values = st.checkbox("Show correlation values", value=False)
                color_scale = st.selectbox("Color scheme:", 
                                          ["RdBu_r", "Viridis", "Plasma", "Turbo"])
            
            with col1:
                corr_matrix = numeric_df.corr()
                
                fig = px.imshow(corr_matrix,
                              text_auto='.2f' if show_values else False,
                              aspect='auto',
                              color_continuous_scale=color_scale,
                              title='Feature Correlation Heatmap')
                fig.update_layout(height=700)
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 🔍 Highly Correlated Features")
            threshold = st.slider("Correlation threshold:", 0.5, 1.0, 0.8, 0.05)
            
            high_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > threshold:
                        high_corr.append({
                            'Feature 1': corr_matrix.columns[i],
                            'Feature 2': corr_matrix.columns[j],
                            'Correlation': round(corr_matrix.iloc[i, j], 3)
                        })
            
            if high_corr:
                high_corr_df = pd.DataFrame(high_corr).sort_values('Correlation', 
                                                                   key=abs, 
                                                                   ascending=False)
                st.dataframe(high_corr_df, use_container_width=True)
            else:
                st.info(f"No correlations above {threshold}")
    
    with tab4:
        st.markdown("### 📦 Feature Comparison")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                selected_feature = st.selectbox(
                    "Select feature for comparison:",
                    numeric_cols
                )
            
            with col2:
                plot_type = st.radio("Plot type:", ["Box", "Violin", "Strip"])
            
            if 'Diagnosis' in df.columns and selected_feature:
                if plot_type == "Box":
                    fig = px.box(df, x='Diagnosis', y=selected_feature,
                               color='Diagnosis',
                               title=f'{selected_feature} by Diagnosis',
                               color_discrete_map={'B':'#22c55e', 'M':'#ef4444'},
                               points="all")
                elif plot_type == "Violin":
                    fig = px.violin(df, x='Diagnosis', y=selected_feature,
                                  color='Diagnosis',
                                  title=f'{selected_feature} by Diagnosis',
                                  color_discrete_map={'B':'#22c55e', 'M':'#ef4444'},
                                  box=True)
                else:  # Strip
                    fig = px.strip(df, x='Diagnosis', y=selected_feature,
                                 color='Diagnosis',
                                 title=f'{selected_feature} by Diagnosis',
                                 color_discrete_map={'B':'#22c55e', 'M':'#ef4444'})
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistical test
                benign = df[df['Diagnosis'] == 'B'][selected_feature].dropna()
                malignant = df[df['Diagnosis'] == 'M'][selected_feature].dropna()
                
                if len(benign) > 0 and len(malignant) > 0:
                    t_stat, p_value = stats.ttest_ind(benign, malignant)
                    
                    st.markdown(f"""
                    <div class="{'success-box' if p_value < 0.05 else 'info-box'}">
                    <strong>📊 Statistical Test (T-Test):</strong><br><br>
                    <strong>T-statistic:</strong> {t_stat:.4f}<br>
                    <strong>P-value:</strong> {p_value:.4e}<br>
                    <strong>Result:</strong> {'<span style="color: #22c55e; font-weight: 600;">Significant difference detected! (p < 0.05)</span>' if p_value < 0.05 else '<span style="color: #f59e0b; font-weight: 600;">No significant difference (p ≥ 0.05)</span>'}
                    </div>
                    """, unsafe_allow_html=True)
            elif selected_feature:
                fig = px.histogram(df, x=selected_feature, nbins=30,
                                 title=f'{selected_feature} Distribution')
                st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.markdown("### 🎯 Feature Importance Analysis")
        
        if 'Diagnosis' in df.columns:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols and len(numeric_cols) > 1:
                col1, col2 = st.columns([3, 1])
                
                with col2:
                    top_n = st.slider("Number of features:", 5, 20, 15)
                    show_all = st.checkbox("Show all features")
                
                X = df[numeric_cols].fillna(df[numeric_cols].mean())
                y = df['Diagnosis'].map({'B': 0, 'M': 1})
                
                with st.spinner("Computing feature importance..."):
                    rf = RandomForestClassifier(n_estimators=100, random_state=42)
                    rf.fit(X, y)
                
                importance_df = pd.DataFrame({
                    'Feature': numeric_cols,
                    'Importance': rf.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                if not show_all:
                    importance_df = importance_df.head(top_n)
                
                with col1:
                    fig = px.bar(importance_df, 
                               x='Importance', 
                               y='Feature',
                               orientation='h',
                               title=f'Top {len(importance_df)} Most Important Features',
                               color='Importance',
                               color_continuous_scale='viridis')
                    fig.update_layout(height=max(400, len(importance_df) * 25))
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("### 📋 Feature Importance Table")
                st.dataframe(importance_df.reset_index(drop=True), use_container_width=True)

def perform_ml_analysis(df):
    """Perform machine learning analysis"""
    
    st.markdown('<h2 class="section-header">🤖 Machine Learning Analysis</h2>', unsafe_allow_html=True)
    
    if 'Diagnosis' not in df.columns:
        st.warning("⚠️ Diagnosis column not found. ML analysis requires a target variable.")
        return
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.warning("⚠️ Not enough numeric features for ML analysis.")
        return
    
    # Sidebar for ML configuration
    with st.expander("⚙️ Model Configuration", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            test_size = st.slider("Test set size:", 0.1, 0.5, 0.2, 0.05)
        with col2:
            n_estimators = st.slider("Number of trees:", 50, 300, 100, 50)
        with col3:
            random_state = st.number_input("Random seed:", 1, 100, 42)
    
    # Prepare data
    X = df[numeric_cols].fillna(df[numeric_cols].mean())
    y = df['Diagnosis'].map({'B': 0, 'M': 1})
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    with st.spinner("🔄 Training Random Forest model..."):
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        rf.fit(X_train_scaled, y_train)
        
        train_score = rf.score(X_train_scaled, y_train)
        test_score = rf.score(X_test_scaled, y_test)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Training Accuracy</h3>
            <h2>{train_score*100:.2f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Testing Accuracy</h3>
            <h2>{test_score*100:.2f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Training Samples</h3>
            <h2>{len(X_train)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Testing Samples</h3>
            <h2>{len(X_test)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Predictions
    y_pred = rf.predict(X_test_scaled)
    y_pred_proba = rf.predict_proba(X_test_scaled)[:, 1]
    
    # Create two columns for visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        
        fig = px.imshow(cm, 
                        text_auto=True,
                        labels=dict(x="Predicted", y="Actual"),
                        x=['Benign (0)', 'Malignant (1)'],
                        y=['Benign (0)', 'Malignant (1)'],
                        color_continuous_scale='Blues')
        fig.update_layout(title='Confusion Matrix', height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📈 ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode='lines', 
            name=f'ROC (AUC = {roc_auc:.3f})',
            line=dict(color='#667eea', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode='lines', 
            name='Random',
            line=dict(dash='dash', color='gray', width=2)
        ))
        fig.update_layout(
            title='Receiver Operating Characteristic',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Classification Report
    st.markdown("### 📋 Detailed Classification Report")
    report = classification_report(y_test, y_pred, 
                                   target_names=['Benign', 'Malignant'], 
                                   output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    
    # Style the dataframe
    st.dataframe(
        report_df.style.background_gradient(cmap='RdYlGn', subset=['precision', 'recall', 'f1-score']),
        use_container_width=True
    )

def perform_pca_analysis(df):
    """Perform PCA dimensionality reduction"""
    
    st.markdown('<h2 class="section-header">🔬 Principal Component Analysis</h2>', unsafe_allow_html=True)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.warning("⚠️ Need at least 2 numeric features for PCA.")
        return
    
    X = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    # PCA Configuration
    with st.expander("⚙️ PCA Configuration", expanded=True):
        n_components = st.slider("Number of components:", 2, min(10, len(numeric_cols)), 
                                min(5, len(numeric_cols)))
    
    # Standardize
    scaler = StandardScaler()
