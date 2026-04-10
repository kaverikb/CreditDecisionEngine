# src/business_metrics.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import config
import os

def calculate_expected_revenue(y_pred_proba, approved, loan_amounts, interest_rate=config.INTEREST_RATE_AVG):
    """Calculate expected revenue from approved loans"""
    
    approved_mask = approved == 1
    
    if approved_mask.sum() == 0:
        return 0
    
    approved_loan_amounts = loan_amounts[approved_mask]
    expected_revenue = (approved_loan_amounts * interest_rate).sum()
    
    return expected_revenue

def calculate_expected_loss(y_true, approved, loan_amounts, lgd=config.LOSS_GIVEN_DEFAULT):
    """Calculate expected loss from approved loans"""
    
    approved_mask = approved == 1
    
    if approved_mask.sum() == 0:
        return 0
    
    approved_defaults = y_true[approved_mask]
    approved_loans = loan_amounts[approved_mask]
    
    expected_loss = (approved_defaults * approved_loans * lgd).sum()
    
    return expected_loss

def calculate_net_profit(expected_revenue, expected_loss):
    """Calculate net profit"""
    return expected_revenue - expected_loss

def business_impact_report(strategies_df):
    """Generate business impact report"""
    
    print("\n" + "="*60)
    print("BUSINESS IMPACT REPORT")
    print("="*60)
    
    for idx, row in strategies_df.iterrows():
        print(f"\n{row['Strategy'].upper()} Strategy:")
        print(f"  Approval Rate: {row['Approval Rate']:.2%}")
        print(f"  Default Rate (Approved): {row['Default Rate (Approved)']:.2%}")
        print(f"  Expected Loss Rate: {row['Expected Loss']:.2%}")
        print(f"  Total Approved: {row['Total Approved']}")
        print(f"  Defaults in Approved: {row['Defaults in Approved']}")
    
    print("\n" + "="*60)

def plot_strategy_comparison(strategies_df, output_name='strategy_comparison.png'):
    """Plot strategy comparison"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Approval Rate
    axes[0].bar(strategies_df['Strategy'], strategies_df['Approval Rate'], color='skyblue')
    axes[0].set_ylabel('Approval Rate')
    axes[0].set_title('Approval Rate by Strategy')
    axes[0].set_ylim(0, 1)
    
    # Default Rate
    axes[1].bar(strategies_df['Strategy'], strategies_df['Default Rate (Approved)'], color='salmon')
    axes[1].set_ylabel('Default Rate')
    axes[1].set_title('Default Rate (Approved) by Strategy')
    axes[1].set_ylim(0, 1)
    
    # Expected Loss
    axes[2].bar(strategies_df['Strategy'], strategies_df['Expected Loss'], color='lightgreen')
    axes[2].set_ylabel('Expected Loss Rate')
    axes[2].set_title('Expected Loss Rate by Strategy')
    axes[2].set_ylim(0, 1)
    
    plt.tight_layout()
    output_path = os.path.join(config.OUTPUTS_DIR, 'business_impact', output_name)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {output_path}")
    plt.close()

def plot_roc_curve(fpr, tpr, auc, output_name='roc_curve.png'):
    """Plot ROC curve"""
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Default Prediction Model')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = os.path.join(config.OUTPUTS_DIR, 'model_performance', output_name)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {output_path}")
    plt.close()

def plot_risk_distribution(y_pred_proba, output_name='risk_distribution.png'):
    """Plot distribution of predicted default probabilities"""
    
    plt.figure(figsize=(10, 6))
    plt.hist(y_pred_proba, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Probability of Default')
    plt.ylabel('Frequency')
    plt.title('Distribution of Default Probabilities')
    plt.axvline(config.RISK_LOW, color='green', linestyle='--', label=f'Low Risk: {config.RISK_LOW}')
    plt.axvline(config.RISK_MEDIUM, color='orange', linestyle='--', label=f'Medium Risk: {config.RISK_MEDIUM}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = os.path.join(config.OUTPUTS_DIR, 'business_impact', output_name)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {output_path}")
    plt.close()