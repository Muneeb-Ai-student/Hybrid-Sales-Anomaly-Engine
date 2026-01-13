from data_generator import generate_data
from model import RetailAnalyticsSystem
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # 1. Generate Data
    generate_data()
    
    # 2. Initialize System
    system = RetailAnalyticsSystem('sales_data.csv')
    
    # 3. Preprocess and Train
    X_train, X_test, y_train, y_test = system.preprocess()
    system.train_forecast(X_train, y_train)
    
    # 4. Evaluate Forecast
    predictions = system.forecaster.predict(X_test)
    print(f"Model R2 Score: {system.forecaster.score(X_test, y_test):.2f}")
    
    # 5. Detect Anomalies
    anomalies = system.detect_anomalies()
    print(f"Detected {len(anomalies)} anomalies in the data.")

    # 6. Visualization (Lab 12 Concept)
    plt.figure(figsize=(12, 6))
    plt.plot(system.df['Date'], system.df['Sales'], label='Actual Sales', color='blue', alpha=0.5)
    plt.scatter(anomalies['Date'], anomalies['Sales'], color='red', label='Anomalies', zorder=5)
    plt.title("Retail Sales Forecast & Anomaly Detection")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()