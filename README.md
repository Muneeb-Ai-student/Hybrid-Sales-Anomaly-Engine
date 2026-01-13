# Predictive Retail Analytics & Anomaly Detection System 📈🛡️

## 📝 Project Overview
This project is an end-to-end Machine Learning pipeline designed to solve two critical retail problems: **Demand Forecasting** and **Fraud/Anomaly Detection**. By integrating supervised and unsupervised learning, the system provides a holistic view of business health.

Developed as part of the **Complex Computing Activity (Lab 14)**, this project demonstrates the integration of multi-faceted ML concepts into a single functional application.

## 🚀 Key Features
- **Sales Forecasting:** Uses a Random Forest Regressor to predict future sales based on temporal features and historical lags.
- **Anomaly Detection:** Implements an Isolation Forest algorithm to flag suspicious spikes or abrupt drops in sales data.
- **Automated Feature Engineering:** Transforms raw time-series data into trainable features (Lags, Day-of-Week, Seasonality).
- **Dynamic Visualization:** Real-time plotting of sales trends with highlighted anomaly markers for business stakeholders.

## 🛠️ Lab Integrations
This project explicitly integrates concepts from the following labs:
1. **Lab 1-2 (Preprocessing):** Handling time-series data, feature scaling, and lag-based feature engineering.
2. **Lab 5-7 (Ensemble Models):** Implementation of Random Forest for robust regression.
3. **Lab 8-9 (Unsupervised Learning):** Utilization of Isolation Forest for outlier detection without labeled anomaly data.

## 📂 Project Structure
```text
├── data_generator.py   # Script to simulate realistic retail data
├── model.py            # Logic for Forecasting and Anomaly models
├── app.py              # Main execution script and visualization
├── requirements.txt    # List of dependencies
└── README.md           # Project documentation
⚙️ Setup & Installation
Clone the repository:

Bash

git clone [https://github.com/Muneeb-Ai-student/Hybrid-Sales-Anomaly-Engine.git)
cd Predictive-Retail-Analytics
Install dependencies:

Bash

pip install -r requirements.txt
Run the application:

Bash

python app.py
📊 Results & Interpretation
R² Score: Typically achieves >0.85, indicating strong predictive power.

Anomaly Identification: The system successfully flags injected data irregularities (e.g., massive spikes or sudden drops), which are visualized as red markers on the generated plot.

🛡️ Ethics & Limitations
While highly effective, this system assumes that historical patterns will repeat. It is designed as a Decision Support System, meaning human intervention is recommended before acting on "Anomaly" flags to ensure fairness to customers.
