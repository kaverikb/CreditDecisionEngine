# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import config
import sys
sys.path.append('src')

from src.model import load_model
from src.collections_model import load_collections_model
from src.data_prep import prepare_data
from src.approval_strategy import segment_risk, simulate_approval_strategy, calculate_business_metrics
from src.shap_explainer import generate_shap_explainer
from sklearn.metrics import roc_curve, roc_auc_score

st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")
st.title("💳 Credit Risk Model Dashboard")

@st.cache_resource
def load_models():
    try:
        default_model = load_model(config.DEFAULT_MODEL)
        collections_model = load_collections_model(config.COLLECTIONS_MODEL)
        return default_model, collections_model
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        st.stop()

@st.cache_data
def load_data():
    try:
        X_train, X_val, X_test, y_train, y_val, y_test, encoders = prepare_data()
        return X_test, y_test
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

# Load
default_model, collections_model = load_models()
X_test, y_test = load_data()

# Predictions (default model only)
y_pred_default = default_model.predict(X_test)
default_auc = roc_auc_score(y_test, y_pred_default)

# Collections metrics from training (hardcoded from main.py output)
collections_auc = 0.7443
collections_recovery_rate = 0.1887

# Navigation
page = st.sidebar.radio("Navigation", [
    "Overview", 
    "Model Performance", 
    "Risk Segmentation", 
    "Approval Strategies", 
    "SHAP Explainability", 
    "Collections"
])

# PAGE: Overview
if page == "Overview":
    st.header("📊 Dashboard Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Default Model AUC", f"{default_auc:.4f}")
    with col2:
        st.metric("Collections Model AUC", f"{collections_auc:.4f}")
    with col3:
        st.metric("Test Borrowers", f"{len(X_test):,}")
    
    st.markdown("---")
    st.subheader("Project Overview")
    st.write("""
    This dashboard demonstrates a complete credit risk modeling pipeline:
    
    ✅ **Default Prediction**: LightGBM model predicting probability of loan default  
    ✅ **Risk Segmentation**: Classifies borrowers into Low/Medium/High risk categories  
    ✅ **Approval Strategy**: Simulates 3 lending strategies with business impact analysis  
    ✅ **Collections Model**: Predicts recovery likelihood for defaulted borrowers  
    ✅ **SHAP Explainability**: Explains individual predictions with feature contributions  
    """)

elif page == "Model Performance":
    st.header("📈 Default Prediction Model Performance")
    
    st.write(f"**Model AUC-ROC: {default_auc:.4f}**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_pred_default)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, label=f'AUC = {default_auc:.4f}', linewidth=2.5, color='#1f77b4')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.set_title('ROC Curve - Default Prediction', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.2)
        st.pyplot(fig, use_container_width=True)
    
    with col2:
        st.subheader("Probability Distribution")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(y_pred_default, bins=60, edgecolor='black', alpha=0.7, color='#1f77b4')
        ax.axvline(config.RISK_LOW, color='green', linestyle='--', linewidth=2, label=f'Low: {config.RISK_LOW}')
        ax.axvline(config.RISK_MEDIUM, color='orange', linestyle='--', linewidth=2, label=f'Medium: {config.RISK_MEDIUM}')
        ax.set_xlabel('Probability of Default', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Distribution of Default Predictions', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        st.pyplot(fig, use_container_width=True)

elif page == "Risk Segmentation":
    st.header("🎯 Risk Segmentation")
    
    risk_segments = segment_risk(y_pred_default)
    low_cnt = (risk_segments == 'Low').sum()
    med_cnt = (risk_segments == 'Medium').sum()
    high_cnt = (risk_segments == 'High').sum()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Low Risk", low_cnt, f"{low_cnt/len(risk_segments)*100:.1f}%")
    with col2:
        st.metric("Medium Risk", med_cnt, f"{med_cnt/len(risk_segments)*100:.1f}%")
    with col3:
        st.metric("High Risk", high_cnt, f"{high_cnt/len(risk_segments)*100:.1f}%")
    
    st.subheader("Distribution by Risk Level")
    risk_data = pd.Series(risk_segments).value_counts().sort_index(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#d62728', '#ff7f0e', '#2ca02c']
    risk_data.plot(kind='bar', ax=ax, color=colors)
    ax.set_title('Borrower Distribution by Risk Level', fontsize=12, fontweight='bold')
    ax.set_xlabel('Risk Level', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    plt.xticks(rotation=0)
    st.pyplot(fig, use_container_width=True)

elif page == "Approval Strategies":
    st.header("💰 Approval Strategy Simulation")
    
    strategy = st.selectbox("Select Strategy", ["Conservative", "Moderate", "Aggressive"])
    
    approved, threshold = simulate_approval_strategy(X_test, y_test, y_pred_default, strategy.lower())
    metrics = calculate_business_metrics(y_test, y_pred_default, approved, strategy)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Approval Rate", f"{metrics['Approval Rate']:.2%}")
    with col2:
        st.metric("Default Rate", f"{metrics['Default Rate (Approved)']:.2%}")
    with col3:
        st.metric("Expected Loss", f"{metrics['Expected Loss']:.2%}")
    with col4:
        st.metric("Approved Count", f"{metrics['Total Approved']:,}")
    
    st.info(f"**Decision Threshold**: {threshold:.3f} | **Defaults in Approved**: {metrics['Defaults in Approved']:,}")
    
    st.subheader("Compare All Strategies")
    strategies_list = []
    for strat in ['conservative', 'moderate', 'aggressive']:
        app, _ = simulate_approval_strategy(X_test, y_test, y_pred_default, strat)
        met = calculate_business_metrics(y_test, y_pred_default, app, strat.title())
        strategies_list.append(met)
    
    comp_df = pd.DataFrame(strategies_list)
    st.dataframe(comp_df, use_container_width=True)

elif page == "SHAP Explainability":
    st.header("🔍 SHAP Model Explainability")
    
    st.write("Select a borrower to understand their default prediction:")
    sample_idx = st.slider("Borrower Index", 0, len(X_test) - 1, 0)
    
    pred_prob = y_pred_default[sample_idx]
    risk_level = segment_risk(np.array([pred_prob]))[0]
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Default Probability", f"{pred_prob:.2%}")
    with col2:
        risk_emoji = {'Low': '🟢', 'Medium': '🟡', 'High': '🔴'}
        st.metric("Risk Level", f"{risk_emoji.get(risk_level, '')} {risk_level}")
    
    st.subheader("Top 10 Contributing Factors")
    
    try:
        explainer, shap_values = generate_shap_explainer(default_model, X_test.values, X_test.values)
        shap_vals = shap_values[1] if isinstance(shap_values, list) else shap_values
        shap_sample = shap_vals[sample_idx]
        
        top_indices = abs(shap_sample).argsort()[-10:][::-1]
        
        factors_list = []
        for idx in top_indices:
            feat_name = X_test.columns[idx]
            feat_val = X_test.iloc[sample_idx, idx]
            shap_contrib = shap_sample[idx]
            direction = '↑ Increases Risk' if shap_contrib > 0 else '↓ Decreases Risk'
            
            factors_list.append({
                'Feature': feat_name,
                'Value': f"{feat_val:.2f}",
                'SHAP Impact': f"{shap_contrib:.4f}",
                'Effect': direction
            })
        
        factors_df = pd.DataFrame(factors_list)
        st.dataframe(factors_df, use_container_width=True)
    except Exception as e:
        st.error(f"Error generating SHAP: {e}")

elif page == "Collections":
    st.header("🏦 Collections Recovery Model")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Collections AUC", f"{collections_auc:.4f}")
    with col2:
        st.metric("Recovery Rate (>50%)", f"{collections_recovery_rate:.2%}")
    with col3:
        st.metric("Defaulted Borrowers", "268,559")
    
    st.subheader("Purpose & Use Cases")
    st.write("""
    The Collections Recovery Model predicts which defaulted borrowers are most likely to repay 
    >50% of their principal after default.
    
    **Key Applications:**
    - 🎯 **Targeted Recovery**: Prioritize collection efforts on high-recovery borrowers
    - 💼 **Resource Optimization**: Allocate staff and budget efficiently  
    - 📊 **Recovery Scoring**: Rank defaulted portfolio by repayment probability
    - 💰 **Financial Impact**: Maximize recoveries, minimize write-offs
    """)
    
    st.subheader("Model Metrics")
    st.success(f"**AUC-ROC**: {collections_auc:.4f} | **Recovery Rate (>50%)**: {collections_recovery_rate:.2%}")