# Customer Decision Engine

A comprehensive credit risk modeling and decision automation system that predicts borrower default probability, simulates lending approval strategies, and optimizes collections resource allocation using machine learning and explainable AI.

## Project Overview

This project implements a production-ready credit risk assessment pipeline that combines:
- Default probability prediction using LightGBM
- Risk segmentation (Low/Medium/High) for borrower classification
- Approval strategy simulation with business impact analysis
- Collections recovery model for defaulted borrowers
- SHAP-based model explainability for decision justification
- Interactive Streamlit dashboard for real-time model exploration

## Directory Structure

CDE/
- data/ (raw, processed, collections subdirectories)
- models/ (default_model.pkl, collections_model.pkl, encoders.pkl)
- notebooks/ (EDA, data prep, model training)
- src/ (data_prep.py, model.py, approval_strategy.py, collections_model.py, shap_explainer.py, business_metrics.py)
- outputs/ (model_performance, strategy_simulations, shap_plots, business_impact subdirectories)
- config.py (configuration)
- main.py (pipeline execution)
- app.py (Streamlit dashboard)
- requirements.txt (dependencies)

## Approach

### 1. Data Preparation

Source: Lending Club dataset (2.26M historical loan records, 2007-2018)
Target: Binary classification (Fully Paid=0, Charged Off=1)
Features: 23 credit features (grade, income, employment, delinquency history)
Processing: Missing value imputation, categorical encoding, stratified train/val/test split (60%/20%/20%)

### 2. Default Prediction Model

Algorithm: LightGBM gradient boosting
Hyperparameters: max_depth=7, learning_rate=0.05, 100 boosting rounds with early stopping
Output: Probability of default (0-1) per borrower
Handling: class_weight='balanced' for 20% default rate

### 3. Risk Segmentation

Low Risk: probability < 0.15
Medium Risk: probability 0.15-0.40
High Risk: probability > 0.40

### 4. Approval Strategy Simulation

Conservative: Approve if default probability < 0.25 (14.7% approval, 4.81% default rate, 4.09% loss)
Moderate: Approve if default probability < 0.40 (38.5% approval, 8.40% default rate, 7.14% loss)
Aggressive: Approve if default probability < 0.60 (76.3% approval, 14.28% default rate, 12.14% loss)

### 5. Collections Recovery Model

Target: Borrowers who recovered >50% of original loan principal
Data: 268,559 defaulted borrowers
Model: LightGBM classifier predicting repayment likelihood
Use Case: Prioritize collection efforts on high-recovery borrowers

### 6. SHAP Explainability

Method: TreeExplainer for per-borrower feature importance
Outputs: Summary plots, force plots, individual prediction justification
Use Case: Compliance, stakeholder communication, model validation

### 7. Interactive Dashboard

Streamlit application with pages: Overview, Model Performance, Risk Segmentation, Approval Strategies, SHAP Explainability, Collections

## Results

Default Prediction Model: AUC-ROC 0.7171, Precision 0.32, Recall 0.68 on 269K test borrowers

Risk Segmentation: Low Risk 12,576 (4.7%), Medium Risk 91,014 (33.8%), High Risk 165,472 (61.5%)

Approval Strategy Impact: Business tradeoff between coverage and risk across three strategies

Collections Recovery Model: AUC-ROC 0.7443, Recovery Rate 18.87% (>50% principal repayment)

Model Explainability: Top drivers are credit grade, open accounts, debt-to-income ratio, months since delinquency

## Running the Pipeline

Setup: pip install -r requirements.txt

Training: python main.py (generates models, plots, strategy comparison)

Dashboard: streamlit run app.py (visit http://localhost:8501)

## Tech Stack

Data Processing: Pandas, NumPy
Machine Learning: LightGBM, scikit-learn
Explainability: SHAP
Visualization: Matplotlib, Plotly, Streamlit
Evaluation: ROC curves, AUC-ROC, classification reports

## Key Insights

Default prediction achieves 0.72 AUC—sufficient for risk-based lending decisions. Approval strategy involves business tradeoff: conservative reduces default but limits growth; aggressive increases volume but risk. Collections recovery is predictable at 0.74 AUC. SHAP explainability reveals credit grade and account history as strongest default signals. 18.87% recovery rate on defaulters is realistic with targeted collection strategy.

## Limitations

Dataset is historical (2007-2018). Collections model uses simplified recovery definition; production uses time-bounded windows. No collection effort data included. Dashboard caches predictions; production would batch-score new applicants. SHAP computation expensive for large datasets.
