# src/data_prep.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import config
import os

def load_accepted_data():
    """Load accepted loans dataset"""
    df = pd.read_csv(config.ACCEPTED_CSV)
    return df

def create_target(df):
    """Create binary target: 1 = defaulted, 0 = paid"""
    # loan_status values: 'Fully Paid', 'Charged Off', 'Current', 'Late', etc.
    # We only want Fully Paid (0) vs Charged Off (1)
    df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])].copy()
    df['target'] = (df['loan_status'] == 'Charged Off').astype(int)
    return df

def select_features(df):
    """Select core features for modeling"""
    features = [
        'loan_amnt', 'term', 'int_rate', 'installment',
        'grade', 'sub_grade', 'emp_length', 'home_ownership',
        'annual_inc', 'verification_status', 'purpose', 'addr_state',
        'dti', 'delinq_2yrs', 'inq_last_6mths', 'mths_since_last_delinq',
        'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc',
        'mths_since_last_major_derog', 'application_type'
    ]
    
    # Keep only features that exist in dataset
    available_features = [f for f in features if f in df.columns]
    return df[available_features + ['target']].copy()

def handle_missing(df):
    """Handle missing values"""
    # Drop rows with >50% missing
    df = df.dropna(thresh=len(df.columns) * 0.5)
    
    # Fill numeric missing with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)
    
    # Fill categorical missing with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    return df

def encode_categoricals(df, fit_encoders=True, encoders=None):
    """Encode categorical variables"""
    if encoders is None:
        encoders = {}
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        if col == 'target':
            continue
        
        if fit_encoders:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            df[col] = encoders[col].transform(df[col].astype(str))
    
    return df, encoders

def prepare_data(fit_encoders=True, encoders=None):
    """Full pipeline: load -> clean -> engineer -> split"""
    print("Loading data...")
    df = load_accepted_data()
    print(f"Loaded {len(df)} records")
    
    print("Creating target...")
    df = create_target(df)
    print(f"Default rate: {df['target'].mean():.2%}")
    
    print("Selecting features...")
    df = select_features(df)
    
    print("Handling missing values...")
    df = handle_missing(df)
    
    print("Encoding categoricals...")
    df, encoders = encode_categoricals(df, fit_encoders=fit_encoders, encoders=encoders)
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Train/val/test split
    print("Splitting data...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=config.VAL_SIZE, random_state=config.RANDOM_STATE, stratify=y_temp
    )
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, encoders