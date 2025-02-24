import pandas as pd
import numpy as np
import datetime
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("gold_details.csv", index_col=0, parse_dates=True)

# Select relevant features (excluding future price to prevent leakage)
features = [
    'Gold_Open', 'Gold_High', 'Gold_Low', 'Gold_Close', 'Gold_Volume',
    'Gold_Day_Low_High_Change', 'Gold_Open_Close_Change', 
    'Gold_Day_Low_High_Change_%', 'Gold_Open_Close_Change_%',
    'Gold_5_Day_MA', 'Gold_10_Day_MA', 'Gold_20_Day_MA', 'Gold_50_Day_MA', 'Gold_RSI'
]

# Adjust target: Predicting Gold price at the next Friday close
df['Gold_Next_Friday_Close'] = df['Gold_Close'].shift(-5)  # Assume 5 days ahead
df = df.dropna()  # Drop NaN rows caused by shifting

# Define training and prediction data
X = df[features]
y = df['Gold_Next_Friday_Close']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a predictive model (RandomForestRegressor)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Save the trained model and scaler
joblib.dump(model, "gold_weekly_model.pkl")
joblib.dump(scaler, "gold_weekly_scaler.pkl")

# ---- FIND NEXT FRIDAY FOR PREDICTION ----
today = datetime.datetime.today()
days_until_friday = (4 - today.weekday()) % 7  # 4 represents Friday (Monday=0, Sunday=6)
if days_until_friday == 0:  
    days_until_friday = 7  # If today is Friday, predict next week's Friday

prediction_date = today + datetime.timedelta(days=days_until_friday)

# Predict gold price for next Friday
latest_features = df[features].iloc[-1:].copy()  # Fix: Keep as DataFrame
latest_features_scaled = scaler.transform(latest_features)
predicted_price = model.predict(latest_features_scaled)[0]

# Save the prediction to a CSV file
formatted_date = prediction_date.strftime('%d%m%y')
prediction_filename = f"gold_prediction_{formatted_date}.csv"

pd.DataFrame({'Date': [prediction_date.strftime('%Y-%m-%d')], 'Predicted_Close': [predicted_price]}).to_csv(prediction_filename, index=False)

print(f"Predicted gold closing price for {prediction_date.strftime('%Y-%m-%d')}: {predicted_price}")
print(f"Saved prediction to {prediction_filename}")
