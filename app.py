
###### TO RUN THE CODE USE BELOW LINE TO RUN IN TERMINAL. THANKS 
###### python -m streamlit run app.py


#!/usr/bin/env python3

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import json
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.covariance import EllipticEnvelope

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="E-Commerce Delivery Delay Prediction on Logistics Data using ML",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# ENHANCED STYLING - ATTRACTIVE DASHBOARD
# ============================================================================
st.markdown('''
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&family=Roboto:wght@400;500;700&display=swap');

    /* Dark mode background */
    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
        color: #e6edf3;
        font-family: 'Roboto', sans-serif;
    }

    /* Main container */
    .main {
        background-color: transparent;
        color: #e6edf3;
    }

    /* Headings - GRADIENT TEXT for attractive look */
    h1 {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-family: 'Poppins', sans-serif !important;
        font-size: 3rem !important;
        letter-spacing: -0.5px !important;
        background: linear-gradient(135deg, #58a6ff 0%, #79c0ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem !important;
    }

    h2 {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-family: 'Poppins', sans-serif !important;
        font-size: 2rem !important;
        letter-spacing: -0.3px !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
    }

    h3 {
        color: #58a6ff !important;
        font-weight: 600 !important;
        font-family: 'Poppins', sans-serif !important;
        font-size: 1.5rem !important;
    }

    /* All text elements */
    p, div, span, label {
        color: #c9d1d9 !important;
        font-family: 'Roboto', sans-serif;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
        border-right: 1px solid #30363d;
    }

    /* Metric cards - ENHANCED with gradient borders */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #30363d;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s, box-shadow 0.2s;
    }

    [data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 12px rgba(88, 166, 255, 0.2);
        border: 1px solid #58a6ff;
    }

    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
        font-family: 'Poppins', sans-serif !important;
    }

    [data-testid="stMetricLabel"] {
        color: #8b949e !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    [data-testid="stMetricDelta"] {
        font-size: 0.85rem !important;
    }

    /* Buttons - GRADIENT */
    .stButton > button {
        background: linear-gradient(135deg, #238636 0%, #2ea043 100%);
        color: #ffffff !important;
        border-radius: 8px;
        border: none;
        padding: 12px 28px;
        font-weight: 600;
        font-size: 1rem;
        font-family: 'Poppins', sans-serif;
        transition: all 0.3s ease;
        box-shadow: 0 4px 8px rgba(35, 134, 54, 0.3);
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #2ea043 0%, #238636 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(35, 134, 54, 0.5);
    }

    /* Text input */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        background-color: #161b22;
        color: #e6edf3 !important;
        border: 2px solid #30363d;
        border-radius: 8px;
        padding: 10px;
        font-size: 1rem;
        transition: border 0.3s;
    }

    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border: 2px solid #58a6ff;
        outline: none;
    }

    /* Dataframe */
    .stDataFrame {
        background-color: #161b22;
        border-radius: 8px;
        border: 1px solid #30363d;
    }

    /* Info/Success/Error boxes - ENHANCED */
    .stAlert {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px;
        color: #e6edf3;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #161b22;
        color: #8b949e;
        border-radius: 8px 8px 0 0;
        padding: 12px 24px;
        font-weight: 600;
        font-family: 'Poppins', sans-serif;
        border: 1px solid #30363d;
        transition: all 0.3s;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #238636 0%, #2ea043 100%);
        color: #ffffff;
        border: 1px solid #238636;
    }

    /* Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #30363d, transparent);
        margin: 2rem 0;
    }

    /* Footer */
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: linear-gradient(180deg, transparent 0%, #0d1117 50%);
        padding: 15px;
        text-align: center;
        font-size: 0.9rem;
        color: #8b949e;
        border-top: 1px solid #30363d;
        z-index: 999;
    }

    .footer a {
        color: #58a6ff;
        text-decoration: none;
        font-weight: 600;
    }

    .footer a:hover {
        color: #79c0ff;
        text-decoration: underline;
    }
</style>
''', unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL AND DATA
# ============================================================================
@st.cache_resource
def load_model_and_data():
    try:
        model = joblib.load('delivery_delay_best_model.joblib')
        data = pd.read_csv('Train.csv', low_memory=False)
        with open('model_results.json', 'r') as f:
            results = json.load(f)
        return model, data, results, None
    except FileNotFoundError as e:
        return None, None, None, str(e)

model, full_data, results, error_msg = load_model_and_data()

# ============================================================================
# SIDEBAR NAVIGATION 
# ============================================================================
with st.sidebar:
    st.markdown("### 📍 Navigation")

    selected = option_menu(
        menu_title=None,
        options=["Home", "Dashboard", "EDA", "Model Performance", "Predictions", "Data Info"],
        icons=["house-fill", "speedometer2", "graph-up-arrow", "trophy-fill", "bullseye", "info-circle-fill"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#161b22"},
            "icon": {"color": "#58a6ff", "font-size": "20px"},
            "nav-link": {
                "font-size": "16px",
                "font-family": "Poppins",
                "font-weight": "500",
                "text-align": "left",
                "margin": "5px",
                "padding": "12px",
                "border-radius": "8px",
                "color": "#c9d1d9",
                "background-color": "transparent",
            },
            "nav-link-selected": {
                "background": "linear-gradient(135deg, #238636 0%, #2ea043 100%)",
                "color": "#ffffff",
                "font-weight": "600",
            },
        },
    )

    st.markdown("---")
    st.markdown('''
    ### ℹ️ About This App

    **Delivery Delay Prediction System**

    ✨ Features:
    - Interactive dashboard
    - Real-time predictions
    - ML model comparison
    - Data visualization

    📊 Built with machine learning to predict delivery delays with high accuracy.
    ''')

    st.markdown("---")
    st.markdown('''
    <div style="text-align: center; padding: 10px; background: #161b22; border-radius: 8px; border: 1px solid #30363d;">
        <p style="margin: 0; font-size: 0.85rem; color: #8b949e;">💡 Tip: Hover over charts for details</p>
    </div>
    ''', unsafe_allow_html=True)

# ============================================================================
# HOME PAGE
# ============================================================================
if selected == "Home":
    st.markdown("# 📦 Predictive Modelling of Delivery Timeliness Using Machine Learning Techniques on E-Commerce Logistics Data")
    st.markdown("### Advanced Machine Learning Dashboard")

    if error_msg:
        st.error(f"❌ Error loading files: {error_msg}")
    else:
        st.markdown('''
        Welcome to the **Predictive Modelling of Delivery Timeliness Using ML Techniques on E-Commerce Logistics Data**! This application leverages cutting-edge 
        machine learning algorithms to predict whether a delivery will arrive on-time or be delayed.

        ### ✨ Key Features
        ''')

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('''
            #### 📊 Dashboard
            Real-time metrics and KPIs with beautiful visualizations
            ''')

        with col2:
            st.markdown('''
            #### 🔍 EDA
            Exploratory data analysis with interactive charts
            ''')

        with col3:
            st.markdown('''
            #### 📈 Performance
            Compare 10 ML models side-by-side
            ''')

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('''
            #### 🎯 Predictions
            Single and batch predictions
            ''')

        with col2:
            st.markdown('''
            #### 📋 Data Info
            Statistical analysis and insights
            ''')

        with col3:
            st.markdown('''
            #### 💾 Export
            Download results as CSV
            ''')

        if model is not None and results is not None:
            st.markdown("---")
            st.markdown("## 🏆 Best Model Summary")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "🤖 Best Model",
                    results['Best Model'],
                    "Top Performer"
                )

            with col2:
                st.metric(
                    "✅ Accuracy",
                    f"{results['Test Accuracy']:.4f}",
                    f"+{results['Test Accuracy']*100:.1f}%"
                )

            with col3:
                st.metric(
                    "📊 ROC-AUC",
                    f"{results['Test ROC-AUC']:.4f}",
                    "Excellent"
                )

            with col4:
                st.metric(
                    "🎯 F1-Score",
                    f"{results['Test F1-Score']:.4f}",
                    "Balanced"
                )

# ============================================================================
# ENHANCED DASHBOARD PAGE
# ============================================================================
elif selected == "Dashboard":
    st.markdown("# 📊 Dashboard Overview")
    st.markdown("### Real-Time Performance Metrics & Analytics")

    if error_msg:
        st.error(f"❌ {error_msg}")
    elif model is not None and full_data is not None and results is not None:

        st.markdown("## 🎯 Key Performance Indicators")

        # Enhanced metric cards in 2 rows
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "🎯 Accuracy", 
                f"{results['Test Accuracy']:.4f}", 
                f"{results['Test Accuracy']*100:.1f}%",
                help="Overall prediction accuracy"
            )

        with col2:
            st.metric(
                "⚖️ Precision", 
                f"{results['Test Precision']:.4f}", 
                f"{results['Test Precision']*100:.1f}%",
                help="Positive prediction accuracy"
            )

        with col3:
            st.metric(
                "🔍 Recall", 
                f"{results['Test Recall']:.4f}", 
                f"{results['Test Recall']*100:.1f}%",
                help="True positive rate"
            )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "🎼 F1-Score", 
                f"{results['Test F1-Score']:.4f}", 
                f"{results['Test F1-Score']*100:.1f}%",
                help="Harmonic mean of precision and recall"
            )

        with col2:
            st.metric(
                "📈 ROC-AUC", 
                f"{results['Test ROC-AUC']:.4f}", 
                f"{results['Test ROC-AUC']*100:.1f}%",
                help="Area under ROC curve"
            )

        with col3:
            st.metric(
                "💎 PR-AUC", 
                f"{results['Test PR-AUC']:.4f}", 
                f"{results['Test PR-AUC']*100:.1f}%",
                help="Precision-Recall AUC"
            )

        st.markdown("---")
        st.markdown("## 📈 Model Performance Comparison")

        models_df = pd.DataFrame(results['All Models Performance'])
        models_df = models_df.sort_values('Test_ROC_AUC', ascending=False)

        # Enhanced chart with gradient colors
        fig = go.Figure()

        colors = ['#0969da', '#1f6feb', '#58a6ff', '#79c0ff']
        metrics = ['Test_Accuracy', 'Test_Precision', 'Test_Recall', 'Test_F1']
        names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

        for idx, (metric, name) in enumerate(zip(metrics, names)):
            fig.add_trace(go.Bar(
                x=models_df['Model'],
                y=models_df[metric],
                name=name,
                marker_color=colors[idx],
                marker_line=dict(width=1.5, color='#30363d')
            ))

        fig.update_layout(
            barmode='group',
            title={
                'text': 'Model Performance Metrics Comparison',
                'font': {'size': 24, 'family': 'Poppins', 'color': '#ffffff'}
            },
            xaxis_title='Model',
            yaxis_title='Score',
            height=550,
            hovermode='x unified',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='#161b22',
            font=dict(color='#e6edf3', size=13, family='Roboto'),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor='rgba(22, 27, 34, 0.8)',
                bordercolor='#30363d',
                borderwidth=1
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("## 🏆 ROC-AUC Score Ranking")

        fig_roc = go.Figure()
        models_sorted = models_df.sort_values('Test_ROC_AUC', ascending=True)

        fig_roc.add_trace(go.Bar(
            y=models_sorted['Model'],
            x=models_sorted['Test_ROC_AUC'],
            orientation='h',
            marker=dict(
                color=models_sorted['Test_ROC_AUC'],
                colorscale='Blues',
                showscale=True,
                colorbar=dict(
                    title="ROC-AUC Score",
                    tickcolor='#e6edf3',
                    titlefont=dict(size=14, family='Poppins')
                ),
                line=dict(color='#30363d', width=2)
            ),
            text=models_sorted['Test_ROC_AUC'].round(4),
            textposition='outside',
            textfont=dict(color='#ffffff', size=13, family='Roboto', weight='bold')
        ))

        fig_roc.update_layout(
            title={
                'text': 'Model Ranking by ROC-AUC Score',
                'font': {'size': 24, 'family': 'Poppins', 'color': '#ffffff'}
            },
            xaxis_title='ROC-AUC Score',
            yaxis_title='',
            height=650,
            showlegend=False,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='#161b22',
            font=dict(color='#e6edf3', size=13, family='Roboto'),
            xaxis=dict(gridcolor='#30363d', gridwidth=1),
            yaxis=dict(gridcolor='#30363d', gridwidth=1)
        )

        st.plotly_chart(fig_roc, use_container_width=True)

        st.markdown("---")
        st.markdown("## 📊 Data Distribution & Statistics")

        TARGET_COL = 'Reached.on.Time_Y.N'
        y = full_data[TARGET_COL].astype(int)

        col1, col2 = st.columns(2)

        with col1:
            # CORRECT: 0=On-Time, 1=Delayed. Sort by index for consistent order.
            counts = y.value_counts().sort_index()

            fig_pie = go.Figure(data=[go.Pie(
                labels=['On-Time', 'Delayed'],  # CORRECT: 0=On-Time, 1=Delayed
                values=counts.values,
                hole=0.4,
                marker=dict(
                    colors=['#238636', '#da3633'],
                    line=dict(color='#0d1117', width=3)
                ),
                textinfo='label+percent',
                textfont=dict(color='#ffffff', size=15, family='Poppins', weight='bold'),
                pull=[0.05, 0.05]
            )])
            fig_pie.update_layout(
                title={
                    'text': 'Delivery Status Distribution',
                    'font': {'size': 20, 'family': 'Poppins', 'color': '#ffffff'}
                },
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e6edf3', family='Roboto'),
                height=400
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # CORRECT: Use sorted counts for consistent order
            counts = y.value_counts().sort_index()

            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=['On-Time', 'Delayed'],  # CORRECT: 0=On-Time, 1=Delayed
                y=counts.values,
                marker=dict(
                    color=['#238636', '#da3633'],
                    line=dict(color='#30363d', width=2)
                ),
                text=y.value_counts().values,
                textposition='outside',
                textfont=dict(color='#ffffff', size=15, family='Poppins', weight='bold')
            ))
            fig_bar.update_layout(
                title={
                    'text': 'Delivery Count Distribution',
                    'font': {'size': 20, 'family': 'Poppins', 'color': '#ffffff'}
                },
                showlegend=False,
                xaxis_title='Status',
                yaxis_title='Count',
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='#161b22',
                font=dict(color='#e6edf3', size=13, family='Roboto'),
                height=400,
                xaxis=dict(gridcolor='#30363d'),
                yaxis=dict(gridcolor='#30363d')
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")
        st.markdown("## 📈 Statistical Summary")

        stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)

        with stats_col1:
            st.info(f"**📊 Total Records**\n\n{len(full_data):,}", icon="📊")

        with stats_col2:
            st.success(f"**✅ On-Time**\n\n{(y==0).sum():,} ({(y==0).sum()/len(y)*100:.1f}%)", icon="✅")

        with stats_col3:
            st.error(f"**❌ Delayed**\n\n{(y==1).sum():,} ({(y==1).sum()/len(y)*100:.1f}%)", icon="❌")

        with stats_col4:
            st.warning(f"**⚙️ Features**\n\n{len(full_data.columns)-1}", icon="⚙️")

    else:
        st.error("❌ Model or data files not found.")

# ============================================================================
# EDA PAGE
# ============================================================================
elif selected == "EDA":
    st.markdown("# 🔍 Exploratory Data Analysis")
    st.markdown("### Discover Patterns & Insights in Your Data")

    if error_msg:
        st.error(f"❌ {error_msg}")
    elif full_data is not None:
        TARGET_COL = 'Reached.on.Time_Y.N'
        X = full_data.drop(columns=[TARGET_COL]).copy()
        y = full_data[TARGET_COL].astype(int).copy()
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        tab1, tab2, tab3, tab4 = st.tabs(["📊 Distributions", "🔥 Correlation", "⚠️ Anomalies", "🎯 Relationships"])

        with tab1:
            st.markdown("## Feature Distributions")

            col1, col2 = st.columns([3, 1])

            with col2:
                num_features = st.slider("Number of features", 5, min(15, len(numeric_cols)), 10)

            with col1:
                fig = make_subplots(
                    rows=3, cols=4,
                    subplot_titles=numeric_cols[:num_features]
                )

                colors_palette = sns.color_palette("husl", num_features)

                for idx, col in enumerate(numeric_cols[:num_features]):
                    row = idx // 4 + 1
                    col_pos = idx % 4 + 1

                    fig.add_trace(
                        go.Histogram(
                            x=X[col],
                            name=col,
                            marker_color=f'rgba({int(colors_palette[idx][0]*255)},{int(colors_palette[idx][1]*255)},{int(colors_palette[idx][2]*255)},0.8)',
                            showlegend=False
                        ),
                        row=row, col=col_pos
                    )

                fig.update_layout(
                    height=800,
                    title_text="Distribution of Numeric Features",
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='#161b22',
                    font=dict(color='#e6edf3', size=11, family='Roboto')
                )
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.markdown("## Correlation Heatmap")

            correlation_matrix = X[numeric_cols].corr()

            fig_heat = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=numeric_cols,
                y=numeric_cols,
                colorscale='RdBu',
                zmid=0,
                text=np.round(correlation_matrix.values, 2),
                texttemplate='%{text}',
                textfont={"size": 9, "color": "#e6edf3"},
                colorbar=dict(title="Correlation", tickcolor='#e6edf3'),
                hovertemplate='%{y} vs %{x}: %{z:.2f}<extra></extra>'
            ))

            fig_heat.update_layout(
                height=800,
                title='Correlation Matrix - Numeric Features',
                xaxis_title='Features',
                yaxis_title='Features',
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='#161b22',
                font=dict(color='#e6edf3', size=11, family='Roboto')
            )

            st.plotly_chart(fig_heat, use_container_width=True)

        with tab3:
            st.markdown("## Anomaly Detection")

            ee = EllipticEnvelope(random_state=42, contamination=0.1)
            anomaly_scores = ee.fit_predict(X[numeric_cols].fillna(0))
            anomalies = np.where(anomaly_scores == -1)[0]

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Outlier Detection")

                feature_idx = st.selectbox("Select feature", range(len(numeric_cols)), key="anomaly_feature")
                feature_name = numeric_cols[feature_idx]

                fig_anom = go.Figure()

                normal_mask = anomaly_scores == 1
                fig_anom.add_trace(go.Scatter(
                    x=np.where(normal_mask)[0],
                    y=X.loc[normal_mask, feature_name],
                    mode='markers',
                    name='Normal',
                    marker=dict(size=6, color='#238636', line=dict(width=0.5, color='#30363d')),
                ))

                if len(anomalies) > 0:
                    fig_anom.add_trace(go.Scatter(
                        x=anomalies,
                        y=X.iloc[anomalies][feature_name],
                        mode='markers',
                        name='Outliers',
                        marker=dict(size=12, color='#da3633', symbol='x', line=dict(width=1.5, color='#ffffff'))
                    ))

                fig_anom.update_layout(
                    title=f'Outlier Detection: {feature_name}',
                    xaxis_title='Sample Index',
                    yaxis_title='Value',
                    height=500,
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='#161b22',
                    font=dict(color='#e6edf3', size=12, family='Roboto'),
                    hovermode='closest'
                )

                st.plotly_chart(fig_anom, use_container_width=True)

            with col2:
                st.markdown("### Anomaly Statistics")
                st.metric("Total Samples", f"{len(X):,}")
                st.metric("Anomalies Detected", f"{len(anomalies):,}")
                st.metric("Anomaly %", f"{(len(anomalies)/len(X)*100):.2f}%")

        with tab4:
            st.markdown("## Feature Relationships")

            col1, col2 = st.columns(2)

            with col1:
                feature1_idx = st.selectbox("X-axis feature", range(len(numeric_cols)), key="scatter_x")

            with col2:
                feature2_idx = st.selectbox("Y-axis feature", range(len(numeric_cols)), key="scatter_y", 
                                          index=1 if len(numeric_cols) > 1 else 0)

            if feature1_idx != feature2_idx:
                fig_scatter = go.Figure()

                fig_scatter.add_trace(go.Scatter(
                    x=X.iloc[:, feature1_idx],
                    y=X.iloc[:, feature2_idx],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=y,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Delivery Status"),
                        line=dict(width=0.5, color='#e6edf3'),
                        opacity=0.7
                    ),
                    text=[f"On-Time" if val == 0 else "Delayed" for val in y],
                    hovertemplate='<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
                ))

                fig_scatter.update_layout(
                    title=f'{numeric_cols[feature1_idx]} vs {numeric_cols[feature2_idx]}',
                    xaxis_title=numeric_cols[feature1_idx],
                    yaxis_title=numeric_cols[feature2_idx],
                    height=600,
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='#161b22',
                    font=dict(color='#e6edf3', size=12, family='Roboto'),
                    hovermode='closest'
                )

                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.warning("⚠️ Please select different features for X and Y axes")

    else:
        st.error("❌ Data file not found.")

# ============================================================================
# MODEL PERFORMANCE PAGE
# ============================================================================
elif selected == "Model Performance":
    st.markdown("# 📊 Model Performance Analysis")
    st.markdown("### Detailed Comparison of 10 Machine Learning Models")

    if error_msg:
        st.error(f"❌ {error_msg}")
    elif results is not None:
        st.markdown("## 📋 Detailed Model Metrics")

        models_df = pd.DataFrame(results['All Models Performance'])
        models_df = models_df.sort_values('Test_ROC_AUC', ascending=False)

        st.dataframe(
            models_df[['Model', 'Test_Accuracy', 'Test_Precision', 'Test_Recall', 'Test_F1', 'Test_ROC_AUC', 'Test_PR_AUC']],
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")
        st.markdown("## 📊 Metric Comparison")

        metric_options = ['Test_Accuracy', 'Test_Precision', 'Test_Recall', 'Test_F1', 'Test_ROC_AUC', 'Test_PR_AUC']
        selected_metric = st.selectbox("Select metric to compare", metric_options)

        fig = go.Figure()
        models_sorted = models_df.sort_values(selected_metric, ascending=True)

        fig.add_trace(go.Bar(
            y=models_sorted['Model'],
            x=models_sorted[selected_metric],
            orientation='h',
            marker=dict(
                color=models_sorted[selected_metric],
                colorscale='Blues',
                showscale=True,
                colorbar=dict(title="Score", tickcolor='#e6edf3'),
                line=dict(color='#30363d', width=2)
            ),
            text=models_sorted[selected_metric].round(4),
            textposition='outside',
            textfont=dict(color='#ffffff', size=13, family='Roboto', weight='bold')
        ))

        fig.update_layout(
            title=f'{selected_metric.replace("Test_", "")} Comparison',
            xaxis_title='Score',
            height=600,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='#161b22',
            font=dict(color='#e6edf3', size=13, family='Roboto')
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("## 🏆 Best Model Details")

        best_model_data = models_df.iloc[0]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("🤖 Model Name", best_model_data['Model'])
        with col2:
            st.metric("✅ Accuracy", f"{best_model_data['Test_Accuracy']:.4f}")
        with col3:
            st.metric("📈 ROC-AUC", f"{best_model_data['Test_ROC_AUC']:.4f}")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("⚖️ Precision", f"{best_model_data['Test_Precision']:.4f}")
        with col2:
            st.metric("🔍 Recall", f"{best_model_data['Test_Recall']:.4f}")
        with col3:
            st.metric("🎼 F1-Score", f"{best_model_data['Test_F1']:.4f}")

    else:
        st.error("❌ Results file not found.")

# ============================================================================
# PREDICTIONS PAGE
# ============================================================================
elif selected == "Predictions":
    st.markdown("# 🎯 Make Predictions")
    st.markdown("### Predict Delivery Status for Single or Batch Orders")

    if error_msg:
        st.error(f"❌ {error_msg}")
    elif model is not None and full_data is not None:
        st.markdown("## 📝 Choose Input Method")

        input_method = st.radio("", ["✍️ Manual Entry", "📤 Upload CSV"], horizontal=True)

        if input_method == "✍️ Manual Entry":
            st.markdown("---")
            st.markdown("### Enter Feature Values")

            TARGET_COL = 'Reached.on.Time_Y.N'
            X = full_data.drop(columns=[TARGET_COL]).copy()

            input_data = {}

            col1, col2, col3 = st.columns(3)

            for idx, col in enumerate(X.columns[:min(len(X.columns), 15)]):
                if idx % 3 == 0:
                    col_obj = col1
                elif idx % 3 == 1:
                    col_obj = col2
                else:
                    col_obj = col3

                if X[col].dtype in [np.float64, np.int64]:
                    input_data[col] = col_obj.number_input(f"{col}", value=float(X[col].mean()), step=0.1)
                else:
                    input_data[col] = col_obj.text_input(f"{col}", value=str(X[col].iloc[0]))

            st.markdown("---")

            if st.button("🚀 Make Prediction", key="predict_button", use_container_width=True):
                try:
                    input_df = pd.DataFrame([input_data])

                    prediction = model.predict(input_df)[0]
                    probability = model.predict_proba(input_df)[0]

                    st.markdown("---")
                    st.markdown("### 📊 Prediction Results")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        if prediction == 0:
                            st.success("### ✅ ON TIME")
                        else:
                            st.error("### ❌ DELAYED")

                    with col2:
                        st.metric("On-Time Probability", f"{probability[0]:.2%}")

                    with col3:
                        st.metric("Delayed Probability", f"{probability[1]:.2%}")

                    fig = go.Figure(data=[go.Bar(
                        x=['On Time', 'Delayed'],
                        y=probability,
                        marker=dict(
                            color=['#238636', '#da3633'],
                            line=dict(color='#30363d', width=2)
                        ),
                        text=[f'{p:.2%}' for p in probability],
                        textposition='outside',
                        textfont=dict(color='#ffffff', size=15, family='Poppins', weight='bold'),
                        hovertemplate='<b>%{x}</b><br>Probability: %{y:.2%}<extra></extra>'
                    )])
                    fig.update_layout(
                        title='Prediction Probability Distribution',
                        yaxis_title='Probability',
                        showlegend=False,
                        template='plotly_dark',
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='#161b22',
                        font=dict(color='#e6edf3', size=13, family='Roboto')
                    )

                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

        else:
            st.markdown("---")
            st.markdown("### 📤 Upload CSV File")

            uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

            if uploaded_file is not None:
                try:
                    upload_data = pd.read_csv(uploaded_file)

                    st.markdown("---")
                    st.markdown("### 👀 Preview Data")
                    st.dataframe(upload_data.head(10), use_container_width=True)

                    if st.button("🚀 Make Batch Predictions", key="batch_predict_button", use_container_width=True):
                        predictions = model.predict(upload_data)
                        probabilities = model.predict_proba(upload_data)

                        results_df = upload_data.copy()
                        results_df['Prediction'] = predictions
                        results_df['Prediction_Label'] = results_df['Prediction'].map({0: 'On Time', 1: 'Delayed'})
                        results_df['On_Time_Probability'] = probabilities[:, 0]
                        results_df['Delayed_Probability'] = probabilities[:, 1]

                        st.markdown("---")
                        st.markdown("### 📊 Prediction Results")
                        st.dataframe(results_df, use_container_width=True)

                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name="predictions.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

                        st.markdown("---")
                        st.markdown("### 📈 Summary Statistics")

                        col1, col2, col3 = st.columns(3)

                        on_time_count = (predictions == 0).sum()
                        delayed_count = (predictions == 1).sum()

                        with col1:
                            st.metric("Total Predictions", f"{len(predictions):,}")

                        with col2:
                            st.metric("✅ On-Time", f"{on_time_count:,}")

                        with col3:
                            st.metric("❌ Delayed", f"{delayed_count:,}")

                        fig = go.Figure(data=[go.Pie(
                            labels=['On Time', 'Delayed'],
                            values=[on_time_count, delayed_count],
                            marker=dict(
                                colors=['#238636', '#da3633'],
                                line=dict(color='#0d1117', width=3)
                            ),
                            textinfo='label+percent',
                            textfont=dict(color='#ffffff', size=15, family='Poppins', weight='bold'),
                            hole=0.4
                        )])
                        fig.update_layout(
                            title='Batch Prediction Distribution',
                            template='plotly_dark',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#e6edf3', family='Roboto')
                        )

                        st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    else:
        st.error("❌ Model file not found.")

# ============================================================================
# DATA INFO PAGE
# ============================================================================
elif selected == "Data Info":
    st.markdown("# 📋 Data Information")
    st.markdown("### Dataset Statistics & Analysis")

    if error_msg:
        st.error(f"❌ {error_msg}")
    elif full_data is not None:
        TARGET_COL = 'Reached.on.Time_Y.N'
        X = full_data.drop(columns=[TARGET_COL]).copy()
        y = full_data[TARGET_COL].astype(int).copy()

        st.markdown("## 📊 Dataset Overview")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("📁 Total Samples", f"{len(full_data):,}")

        with col2:
            st.metric("📊 Total Features", len(X.columns))

        with col3:
            st.metric("🎯 Target Classes", len(y.unique()))

        with col4:
            st.metric("⚠️ Delay Rate", f"{y.mean():.2%}")

        st.markdown("---")
        st.markdown("## 📈 Target Distribution")

        # CORRECT: Sort by index ensures 0=On-Time, 1=Delayed order
        counts = y.value_counts().sort_index()

        fig1 = go.Figure()
        fig1.add_trace(go.Bar(
            x=['On-Time', 'Delayed'],  # CORRECT: 0=On-Time, 1=Delayed
            y=counts.values,
            marker=dict(
                color=['#238636', '#da3633'],
                line=dict(color='#30363d', width=2)
            ),
            text=y.value_counts().sort_index().values,
            textposition='outside',
            textfont=dict(color='#ffffff', size=15, family='Poppins', weight='bold')
        ))
        fig1.update_layout(
            title='Target Distribution (Count)',
            showlegend=False,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='#161b22',
            font=dict(color='#e6edf3', size=13, family='Roboto')
        )
        st.plotly_chart(fig1, use_container_width=True)

        st.markdown("---")
        st.markdown("## 📊 Feature Statistics")

        tab1, tab2 = st.tabs(["🔢 Numeric Features", "📝 Categorical Features"])

        with tab1:
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            numeric_stats = X[numeric_cols].describe().T
            st.dataframe(numeric_stats, use_container_width=True)

        with tab2:
            categorical_cols = X.select_dtypes(exclude=[np.number]).columns

            if len(categorical_cols) > 0:
                for col in categorical_cols:
                    st.markdown(f"### {col}")
                    st.write(X[col].value_counts())
            else:
                st.info("ℹ️ No categorical features in dataset")

        st.markdown("---")
        st.markdown("## ⚠️ Missing Values")

        missing_df = pd.DataFrame({
            'Column': full_data.columns,
            'Missing_Count': full_data.isnull().sum(),
            'Missing_Percentage': (full_data.isnull().sum() / len(full_data) * 100).round(2)
        }).sort_values('Missing_Count', ascending=False)

        missing_data = missing_df[missing_df['Missing_Count'] > 0]
        if len(missing_data) > 0:
            st.dataframe(missing_data, use_container_width=True)
        else:
            st.success("✅ No missing values in dataset!")

    else:
        st.error("❌ Data file not found.")

# ============================================================================
# FOOTER - CREATED BY KUNAL
# ============================================================================
st.markdown("---")
st.markdown('''
<div class="footer">
    <p style="margin: 5px 0; font-size: 0.95rem;">
         <strong>Delivery Delay Prediction System</strong> | 
        Built with <span style="color: #da3633;"></span> using Streamlit & Plotly
    </p>
    <p style="margin: 5px 0; font-size: 0.9rem; color: #58a6ff;">
        <strong>Created by Kunal</strong> | 2026
    </p>
    <p style="margin: 5px 0; font-size: 0.85rem;">
        Machine Learning • Data Science • Predictive Analytics
    </p>
</div>
''', unsafe_allow_html=True)