# src/shap_explainer.py

import shap
import matplotlib.pyplot as plt
import numpy as np
import config
import os

def generate_shap_explainer(model, X_train, X_test):
    """Generate SHAP explainer"""
    
    print("Generating SHAP explainer...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    return explainer, shap_values

def plot_summary(shap_values, X_test, output_name='summary_plot.png'):
    """Plot SHAP summary plot"""
    
    plt.figure(figsize=(12, 8))
    shap_vals = shap_values[1] if isinstance(shap_values, list) else shap_values
    shap.summary_plot(shap_vals, X_test.values, plot_type='bar', show=False)
    
    output_path = os.path.join(config.OUTPUTS_DIR, 'shap_plots', output_name)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {output_path}")
    plt.close()

def plot_dependence(shap_values, X_test, feature_name, output_name='dependence_plot.png'):
    """Plot SHAP dependence plot for a specific feature"""
    
    plt.figure(figsize=(10, 6))
    shap_vals = shap_values[1] if isinstance(shap_values, list) else shap_values
    shap.dependence_plot(feature_name, shap_vals, X_test.values, show=False)
    
    output_path = os.path.join(config.OUTPUTS_DIR, 'shap_plots', output_name)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {output_path}")
    plt.close()

def plot_force(explainer, shap_values, X_test, sample_idx=0, output_name='force_plot.html'):
    """Plot SHAP force plot for individual prediction"""
    
    shap_vals = shap_values[1] if isinstance(shap_values, list) else shap_values
    expected_val = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
    
    force_plot = shap.force_plot(
        expected_val,
        shap_vals[sample_idx],
        X_test[sample_idx],
        show=False
    )
    
    output_path = os.path.join(config.OUTPUTS_DIR, 'shap_plots', output_name)
    shap.save_html(output_path, force_plot)
    print(f"Saved to {output_path}")

def explain_prediction(explainer, shap_values, X_test, sample_idx, top_n=5):
    """Explain individual prediction"""
    
    shap_vals = shap_values[1] if isinstance(shap_values, list) else shap_values
    shap_sample = shap_vals[sample_idx]
    
    top_indices = abs(shap_sample).argsort()[-top_n:][::-1]
    
    print(f"\nTop {top_n} factors for prediction:")
    for idx in top_indices:
        feature = X_test.columns[idx]
        value = X_test.iloc[sample_idx, idx]
        contribution = shap_sample[idx]
        print(f"  {feature}: {value:.2f} (SHAP: {contribution:.4f})")