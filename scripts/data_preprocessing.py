import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Encode the target variable
def encode_target(df, target_col='diagnosis'):
    df[target_col] = df[target_col].map({'B': 0, 'M': 1})
    return df

# Scale numeric features
def scale_features(df, feature_cols):
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df

# Split dataset into train and test sets
def split_data(df, target_col='diagnosis', test_size=0.2, random_state=42):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

# Full preprocessing pipeline
def preprocess_pipeline(file_path):
    # Load data
    df = load_data(file_path)
    
    # Encode target
    df = encode_target(df)
    
    # Select features based on VIF reduction from EDA
    feature_cols = [
        'concave points_mean','area_worst','fractal_dimension_worst','smoothness_worst','symmetry_worst',
        'fractal_dimension_mean','smoothness_mean','compactness_se','fractal_dimension_se','concave points_se',
        'symmetry_se','perimeter_se','concavity_se','symmetry_mean','smoothness_se','texture_se','texture_mean'
    ]
    
    # Scale features
    df = scale_features(df, feature_cols)
    
    # Split into train/test
    X_train, X_test, y_train, y_test = split_data(df)
    
    return X_train, X_test, y_train, y_test

