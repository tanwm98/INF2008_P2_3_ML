import pandas as pd
import numpy as np
import datetime
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("gold_details.csv", index_col=0, parse_dates=True)

features = [
    'Gold_Open', 'Gold_High', 'Gold_Low', 'Gold_Close', 'Gold_Volume',
    'Gold_Day_Low_High_Change', 'Gold_Open_Close_Change', 
    'Gold_Day_Low_High_Change_%', 'Gold_Open_Close_Change_%',
    'Gold_5_Day_MA', 'Gold_10_Day_MA', 'Gold_20_Day_MA', 'Gold_50_Day_MA', 'Gold_RSI'
]

target = 'Gold_Next_Day_Close'
df = df.dropna()

# Define training and prediction data
X = df[features]
y = df[target]

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
joblib.dump(model, "gold_price_model.pkl")
joblib.dump(scaler, "gold_price_scaler.pkl")

latest_features = df[features].iloc[-1:].copy()  # Keep as DataFrame to retain column names
latest_features_scaled = scaler.transform(latest_features) 
predicted_price = model.predict(latest_features_scaled)[0]

prediction_date = datetime.datetime.today().strftime('%d%m%y')
prediction_filename = f"gold_prediction_{prediction_date}.csv"

pd.DataFrame({'Date': [datetime.datetime.today().strftime('%Y-%m-%d')], 'Predicted_Close': [predicted_price]}).to_csv(prediction_filename, index=False)

print(f"Predicted gold closing price for {datetime.datetime.today().strftime('%Y-%m-%d')}: {predicted_price}")
print(f"Saved prediction to {prediction_filename}")
