import warnings
import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

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


# Suppress warnings
warnings.filterwarnings("ignore")

# could have had a preprocessing step here but chose to take the cleaned data directly
train_data = pd.read_csv("./data/data_cleaned.csv")

X = train_data.drop('Time_taken(min)', axis=1)
y = train_data['Time_taken(min)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test, scaler = standardize(X_train, X_test)

model = xgb.XGBRegressor(n_estimators=20, max_depth=9)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

