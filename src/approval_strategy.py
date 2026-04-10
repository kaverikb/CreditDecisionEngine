# src/approval_strategy.py

import pandas as pd
import numpy as np
import config

def segment_risk(probability_default):
    """Segment borrowers into risk categories"""
    risk = np.where(
        probability_default < config.RISK_LOW, 'Low',
        np.where(probability_default < config.RISK_MEDIUM, 'Medium', 'High')
    )
    return risk

def simulate_approval_strategy(X_test, y_test, y_pred_proba, strategy='moderate'):
    """Simulate approval decisions based on risk thresholds"""
    
    if strategy == 'conservative':
        threshold = config.APPROVE_THRESHOLD_CONSERVATIVE
    elif strategy == 'moderate':
        threshold = config.APPROVE_THRESHOLD_MODERATE
    elif strategy == 'aggressive':
        threshold = config.APPROVE_THRESHOLD_AGGRESSIVE
    else:
        threshold = 0.5
    
    # Approval decision: approve if probability of default < threshold
    approved = (y_pred_proba < threshold).astype(int)
    
    return approved, threshold

def calculate_business_metrics(y_true, y_pred_proba, approved, strategy_name):
    """Calculate approval rate, default rate, expected loss"""
    
    # Filter to approved applicants only
    approved_mask = approved == 1
    
    if approved_mask.sum() == 0:
        return None
    
    approved_defaults = y_true[approved_mask].sum()
    approved_total = approved_mask.sum()
    
    approval_rate = approved_total / len(y_true)
    default_rate_approved = approved_defaults / approved_total if approved_total > 0 else 0
    
    # Expected loss: default_rate * loss_given_default * avg_loan_amount
    # Simplified: default_rate * loss_given_default
    expected_loss = default_rate_approved * config.LOSS_GIVEN_DEFAULT
    
    metrics = {
        'Strategy': strategy_name,
        'Approval Rate': approval_rate,
        'Default Rate (Approved)': default_rate_approved,
        'Expected Loss': expected_loss,
        'Total Approved': approved_total,
        'Defaults in Approved': approved_defaults
    }
    
    return metrics

def compare_strategies(X_test, y_test, y_pred_proba):
    """Compare all three approval strategies"""
    
    results = []
    
    for strategy in ['conservative', 'moderate', 'aggressive']:
        approved, threshold = simulate_approval_strategy(X_test, y_test, y_pred_proba, strategy)
        metrics = calculate_business_metrics(y_test, y_pred_proba, approved, strategy)
        results.append(metrics)
    
    results_df = pd.DataFrame(results)
    print("\nStrategy Comparison:")
    print(results_df.to_string(index=False))
    
    return results_df