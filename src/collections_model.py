# src/collections_model.py

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import config
import pickle

def prepare_collections_data(df_accepted):
    """Prepare defaulted borrowers for collections model"""
    
    # Filter only defaulted loans
    defaulted = df_accepted[df_accepted['loan_status'] == 'Charged Off'].copy()
    
    # Create target: 1 = recovered >50% of principal, 0 = recovered <50%
    defaulted['collections_target'] = (defaulted['total_rec_prncp'] / defaulted['loan_amnt'] > 0.5).astype(int)
    
    print(f"Defaulted borrowers: {len(defaulted)}")
    print(f"Recovered >50%: {defaulted['collections_target'].mean():.2%}")
    
    # Save defaulted borrowers
    defaulted.to_csv(config.DEFAULTED_CSV, index=False)
    print(f"Saved to {config.DEFAULTED_CSV}")
    
    return defaulted

def select_collections_features(df):
    """Select features for collections recovery model"""
    
    features = [
        'loan_amnt', 'int_rate', 'installment', 'grade',
        'annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths',
        'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc',
        'mths_since_last_delinq', 'mths_since_last_major_derog',
        'emp_length', 'home_ownership'
    ]
    
    available = [f for f in features if f in df.columns]
    return df[available + ['collections_target']].copy()

def train_collections_model(X_train, y_train, X_val, y_val):
    """Train LightGBM for collections recovery prediction"""
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'max_depth': config.MAX_DEPTH,
        'learning_rate': config.LEARNING_RATE,
        'num_leaves': 31,
        'verbose': -1,
        'random_state': config.RANDOM_STATE
    }
    
    print("Training collections model...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=config.N_ESTIMATORS,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(10)]
    )
    
    return model

def evaluate_collections_model(model, X_test, y_test):
    """Evaluate collections model"""
    
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    recovery_rate = y_test.mean()
    
    print(f"\nCollections Model Performance:")
    print(f"AUC-ROC: {auc:.4f}")
    print(f"Recovery Rate (>50%): {recovery_rate:.2%}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    
    return auc, y_pred_proba

def save_collections_model(model, path):
    """Save collections model"""
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Collections model saved to {path}")

def load_collections_model(path):
    """Load collections model"""
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model