import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

class RetailAnalyticsSystem:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.scaler = StandardScaler()
        self.forecaster = RandomForestRegressor(n_estimators=100, random_state=42)
        self.anomaly_detector = IsolationForest(contamination=0.05, random_state=42)

    def preprocess(self):
        # Feature Engineering: Lag features (Lab 1-2 Concept)
        self.df['Lag_1'] = self.df['Sales'].shift(1)
        self.df['Lag_7'] = self.df['Sales'].shift(7)
        self.df.dropna(inplace=True)
        
        X = self.df[['Day_of_Week', 'Month', 'Lag_1', 'Lag_7']]
        y = self.df['Sales']
        return train_test_split(X, y, test_size=0.2, shuffle=False)

    def train_forecast(self, X_train, y_train):
        self.forecaster.fit(X_train, y_train)
        print("Forecasting Model Trained.")

    def detect_anomalies(self):
        # Unsupervised Learning (Lab 8-9 Concept)
        sales_reshaped = self.df['Sales'].values.reshape(-1, 1)
        self.df['Anomaly_Score'] = self.anomaly_detector.fit_predict(sales_reshaped)
        # -1 means anomaly, 1 means normal
        return self.df[self.df['Anomaly_Score'] == -1]