# src/model.py

import pandas as pd
import lightgbm as lgb
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
import config

def train_default_model(X_train, y_train, X_val, y_val):
    """Train LightGBM model for default prediction"""
    
    # Create LightGBM dataset
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Parameters
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'max_depth': config.MAX_DEPTH,
        'learning_rate': config.LEARNING_RATE,
        'num_leaves': 31,
        'verbose': -1,
        'random_state': config.RANDOM_STATE,
        'is_unbalance': True  # Handle class imbalance
    }
    
    # Train
    print("Training model...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=config.N_ESTIMATORS,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(10)]
    )
    
    return model

def evaluate_model(model, X_test, y_test, set_name='Test'):
    """Evaluate model performance"""
    
    # Predictions
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    print(f"\n{set_name} Set Performance:")
    print(f"AUC-ROC: {auc:.4f}")
    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"True Negatives: {tn}")
    print(f"False Negatives: {fn}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    
    return auc, y_pred_proba, y_pred

def save_model(model, path):
    """Save trained model"""
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {path}")

def load_model(path):
    """Load trained model"""
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model

def get_feature_importance(model, X_train):
    """Get feature importance"""
    importance = model.feature_importance()
    feature_names = X_train.columns
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return importance_df