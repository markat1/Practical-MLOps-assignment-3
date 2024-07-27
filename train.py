import warnings
import pandas as pd
import os
import numpy as np
import argparse
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from preprocessing import *


# Suppress warnings
warnings.filterwarnings("ignore")

def main(args):
    # Load training data
    train_data_path = os.path.join(args.train, "train.csv")
    output_dir = "/opt/ml/model"

    environment = os.getenv("ENVIRONMENT", "local")

    if environment == "local":
        train_data_path = "data/train/train.csv"
        output_dir = "output"

    # df_train = pd.read_csv("./dataset/raw_data.csv")  # Load Data
    df_train = pd.read_csv(train_data_path)  # Load Data
    
    dp = DataProcessing()
    dp.cleaning_steps(df_train)                                # Perform Cleaning
    dp.extract_label_value(df_train)                           # Extract Label Value
    dp.perform_feature_engineering(df_train)                   # Perform feature engineering

    # Split features & label
    X = df_train.drop('Time_taken(min)', axis=1)               # Features
    y = df_train['Time_taken(min)']                            # Target variable

    label_encoders = dp.label_encoding(X)                      # Label Encoding
    X_train, X_test, y_train, y_test = dp.data_split(X, y)     # Test Train Split
    X_train, X_test, scaler = dp.standardize(X_train, X_test)  # Standardization

    # Build Model
    model = xgb.XGBRegressor(n_estimators=20, max_depth=9)
    model.fit(X_train, y_train)

    # Evaluate Model
    y_pred = model.predict(X_test)
    dp.evaluate_model(y_test, y_pred)

    # Save model
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save model
    model_path = os.path.join(output_dir, "model.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        type=str,
        help="Path to the training data",
        default="/opt/ml/input/data/train",
    )
    parser.add_argument("--n-estimators", type=int, default=20)

    args = parser.parse_args()
    main(args)
