import warnings
import pandas as pd
import os
import numpy as np
import argparse
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def standardize(self, X_train, X_test):
        """
        Standardizes the training and testing feature sets:
        - Fits a StandardScaler on X_train
        - Transforms X_train and X_test using the fitted StandardScaler
        - Returns X_train, X_test, and the fitted StandardScaler
        """
        
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test, scaler

def evaluate_model(self, y_test, y_pred):
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        print("Mean Absolute Error (MAE):", round(mae, 2))
        print("Mean Squared Error (MSE):", round(mse, 2))
        print("Root Mean Squared Error (RMSE):", round(rmse, 2))
        print("R-squared (R2) Score:", round(r2, 2))


# Suppress warnings
warnings.filterwarnings("ignore")



def main(args):
    # could have had a preprocessing step here but chose to take the cleaned data directly
    train_data = pd.read_csv("./data/data_cleaned.csv")

    environment = os.getenv("ENVIRONMENT", "local")
    output_dir = "/opt/ml/model"

    if environment == "local":
        train_data_path = "data/train/train.csv"
        output_dir = "output"

    X = train_data.drop('Time_taken(min)', axis=1)
    y = train_data['Time_taken(min)']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test, scaler = standardize(X_train, X_test)

    model = xgb.XGBRegressor(n_estimators=20, max_depth=9)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    evaluate_model(y_test, y_pred)

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
        parser.add_argument("--n-estimators", type=int, default=100)

        args = parser.parse_args()
        main(args)