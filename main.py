# main.py

import sys
sys.path.append('src')

import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve
import config
import os


from src.data_prep import prepare_data, load_accepted_data, create_target, select_features, handle_missing, encode_categoricals
from src.model import train_default_model, evaluate_model, save_model, load_model
from src.approval_strategy import segment_risk, compare_strategies
from src.collections_model import prepare_collections_data, select_collections_features, train_collections_model, evaluate_collections_model, save_collections_model
from src.shap_explainer import generate_shap_explainer, plot_summary, plot_force, explain_prediction
from src.business_metrics import plot_strategy_comparison, plot_roc_curve, plot_risk_distribution, business_impact_report

def main():
    print("="*60)
    print("CREDIT RISK MODEL PIPELINE")
    print("="*60)
    
    # Step 1: Data Preparation
    print("\n[1/7] Data Preparation...")
    X_train, X_val, X_test, y_train, y_val, y_test, encoders = prepare_data()
    
    # Step 2: Train Default Model
    print("\n[2/7] Training Default Prediction Model...")
    model = train_default_model(X_train, y_train, X_val, y_val)
    
    # Step 3: Evaluate Default Model
    print("\n[3/7] Evaluating Default Model...")
    auc_test, y_pred_proba, y_pred = evaluate_model(model, X_test, y_test, set_name='Test')
    save_model(model, config.DEFAULT_MODEL)
    
    # Step 4: Risk Segmentation
    print("\n[4/7] Risk Segmentation...")
    risk_segments = segment_risk(y_pred_proba)
    print(f"Low Risk: {(risk_segments == 'Low').sum()}")
    print(f"Medium Risk: {(risk_segments == 'Medium').sum()}")
    print(f"High Risk: {(risk_segments == 'High').sum()}")
    
    # Step 5: Approval Strategy Simulation
    print("\n[5/7] Approval Strategy Simulation...")
    strategies_df = compare_strategies(X_test, y_test, y_pred_proba)
    business_impact_report(strategies_df)
    strategies_df.to_csv(os.path.join(config.OUTPUTS_DIR, 'strategy_simulations', 'strategy_comparison.csv'), index=False)
    
    # Step 6: Collections Model
    print("\n[6/7] Collections Model...")
    df_accepted = load_accepted_data()
    df_accepted = create_target(df_accepted)
    defaulted_df = prepare_collections_data(df_accepted)
    
    # Select features before creating target
    defaulted_features = select_collections_features(defaulted_df)
    defaulted_features = handle_missing(defaulted_features)
    defaulted_features, _ = encode_categoricals(defaulted_features, fit_encoders=True)
    
    if len(defaulted_features) > 100:
        from sklearn.model_selection import train_test_split
        X_coll = defaulted_features.drop('collections_target', axis=1)
        y_coll = defaulted_features['collections_target']
        
        X_coll_train, X_coll_test, y_coll_train, y_coll_test = train_test_split(
            X_coll, y_coll, test_size=0.2, random_state=config.RANDOM_STATE, stratify=y_coll
        )
        X_coll_train, X_coll_val, y_coll_train, y_coll_val = train_test_split(
            X_coll_train, y_coll_train, test_size=0.2, random_state=config.RANDOM_STATE, stratify=y_coll_train
        )
        
        collections_model = train_collections_model(X_coll_train, y_coll_train, X_coll_val, y_coll_val)
        auc_coll, y_coll_proba = evaluate_collections_model(collections_model, X_coll_test, y_coll_test)
        save_collections_model(collections_model, config.COLLECTIONS_MODEL)
    
    # Step 7: SHAP Explainability
    print("\n[7/7] SHAP Explainability...")
    explainer, shap_values = generate_shap_explainer(model, X_train.values, X_test.values)
    X_test_df = pd.DataFrame(X_test.values, columns=X_test.columns) if isinstance(X_test, pd.DataFrame) else X_test
    plot_summary(shap_values, X_test_df, output_name='summary_plot.png')
    plot_force(explainer, shap_values, X_test.values, sample_idx=0, output_name='force_plot_sample.html')
    explain_prediction(explainer, shap_values, X_test, sample_idx=0, top_n=5)
    
    # Visualizations
    print("\nGenerating visualizations...")
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plot_roc_curve(fpr, tpr, auc_test)
    plot_risk_distribution(y_pred_proba)
    plot_strategy_comparison(strategies_df)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)

def prepare_data_split(X, y=None):
    """Helper for train/test split"""
    from sklearn.model_selection import train_test_split
    if y is None:
        y = X['collections_target']
        X = X.drop('collections_target', axis=1)
    return train_test_split(X, y, test_size=0.2, random_state=config.RANDOM_STATE, stratify=y)

if __name__ == '__main__':
    main()