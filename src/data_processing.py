
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from src.utils import map_ip_to_country
def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from CSV, Excel, or JSON files based on file extension.
    Supported formats: .csv, .xlsx, .xls, .json
    """
    # Extract the file extension by splitting at the dot and taking the last part
    # Example: "data.csv" → "csv"
    file_ext = file_path.lower().split('.')[-1]

    try:
        # If file extension is CSV → use pandas read_csv()
        if file_ext == "csv":
            return pd.read_csv(file_path)
         # If file extension is Excel (xlsx or xls) → use pandas read_excel()
        elif file_ext in ("xlsx", "xls"):
            return pd.read_excel(file_path)
         # If file extension is JSON → use pandas read_json()
        elif file_ext == "json":
            return pd.read_json(file_path)
        else:
            # If file format is not supported, raise an explicit error
            raise ValueError(f"Unsupported file type: .{file_ext}")
    except Exception as e:
        # If any error occurs (e.g., file missing, corrupt), wrap it in a ValueError
        # This gives a clear and consistent error message for the user
        raise ValueError(f"Failed to load file '{file_path}': {e}")



def clean_data(df):
    """
    Handle missing values, remove duplicates, and correct data types.
    """
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values (for this dataset, we'll check if any)
    # Most numerical features in these sets don't have many NaNs, but we'll be safe.
    df = df.dropna() 
    
    return df



def feature_engineer_fraud(df, ip_df):
    """
    Feature engineering for Fraud_Data.csv.
    - Geolocation mapping
    - Time-based features
    - Transaction frequency/velocity
    """
    # 1. Geolocation Mapping
    df = map_ip_to_country(df, ip_df)
    
    # 2. Time-based features
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    
    df['hour_of_day'] = df['purchase_time'].dt.hour
    df['day_of_week'] = df['purchase_time'].dt.dayofweek
    df["ip_address"] = df["ip_address"].astype(int)
    # 3. Time since signup
    df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds()
    
    # 4. Transaction frequency/velocity
    # Number of transactions per device in this dataset
    df['device_id_count'] = df.groupby('device_id')['user_id'].transform('count')
    # Number of transactions per IP
    df['ip_address_count'] = df.groupby('ip_address')['user_id'].transform('count')
    
    return df

def transform_data(df, categorical_features, numerical_features):
    """
    Normalize/Scale numerical features and Encode categorical features.
    """
    scaler = StandardScaler()
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', scaler, numerical_features),
            ('cat', encoder, categorical_features)
        ],
        remainder='passthrough'
    )
    
    # This usually returns a numpy array, we might want to convert back to DF for EDA
    transformed_data = preprocessor.fit_transform(df)
    
    # Get feature names for the new dataframe
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    all_feature_names = list(numerical_features) + list(cat_feature_names) + [col for col in df.columns if col not in categorical_features + numerical_features]
    
    transformed_df = pd.DataFrame(transformed_data, columns=all_feature_names)
    return transformed_df, preprocessor

def handle_imbalance(X, y, strategy='smote'):
    """
    Apply SMOTE or undersampling.
    """
    if strategy == 'smote':
        sampler = SMOTE(random_state=42)
    elif strategy == 'undersample':
        sampler = RandomUnderSampler(random_state=42)
    else:
        raise ValueError("Strategy must be 'smote' or 'undersample'")
    
    X_res, y_res = sampler.fit_resample(X, y)
    return X_res, y_res