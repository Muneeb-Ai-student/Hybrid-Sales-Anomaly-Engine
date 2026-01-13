import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_data():
    np.random.seed(42)
    days = 400
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(days)]
    
    # Base sales with seasonality
    base_sales = 200 + 50 * np.sin(np.linspace(0, 8 * np.pi, days))
    noise = np.random.normal(0, 10, days)
    sales = base_sales + noise
    
    # Adding Anomalies (Fraud or abrupt shifts)
    sales[50] += 300  # Spike
    sales[150] -= 150 # Drop
    sales[300:305] += 250 # Sudden shift
    
    df = pd.DataFrame({'Date': dates, 'Sales': sales})
    df['Day_of_Week'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    df.to_csv('sales_data.csv', index=False)
    print("Dataset 'sales_data.csv' created successfully!")

if __name__ == "__main__":
    generate_data()