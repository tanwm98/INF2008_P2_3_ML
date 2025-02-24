import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import time
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def fetch_data_with_retry(ticker, start_date, end_date, max_retries=3, delay=5):
    retries = 0
    while retries < max_retries:
        try:
            df = yf.download(ticker, start=start_date, end=end_date)
            if not df.empty:
                return df
            else:
                print(f"No data fetched for {ticker}.")
                return pd.DataFrame()
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}. Retrying in {delay} seconds...")
            retries += 1
            time.sleep(delay)
    print(f"Max retries reached for {ticker}. Skipping...")
    return pd.DataFrame()

def fetch_main_data(start_date, end_date):
    # Define the tickers for the assets
    tickers = {
        'Gold': 'GC=F',  # Gold futures
        'Silver': 'SI=F',  # Silver futures
        'Crude Oil': 'CL=F',  # Crude oil futures
        'DXY': 'DX-Y.NYB',  # US Dollar Index
        'S&P500': '^GSPC',  # S&P 500 Index
    }
    
    # Initialize an empty DataFrame
    main_data = pd.DataFrame()
    
    # Fetch data for each ticker with retry and delay
    for name, ticker in tickers.items():
        print(f"Fetching data for {name}...")
        df = fetch_data_with_retry(ticker, start_date, end_date)
        if not df.empty:
            main_data[name] = df['Close']
        else:
            print(f"Warning: No data fetched for {name} ({ticker})")
        time.sleep(2)  # Add a delay between requests
    
    return main_data

def fetch_gold_data(start_date, end_date):
    print("Fetching detailed data for Gold...")
    gold_data = fetch_data_with_retry('GC=F', start_date, end_date)

    if gold_data.empty:
        print("No data fetched for Gold.")
        return pd.DataFrame()

    # Validate required columns
    required_columns = ['Open', 'High', 'Low', 'Close']
    for col in required_columns:
        if col not in gold_data.columns:
            raise ValueError(f"Missing required column: {col}")

    # Calculate additional details
    gold_data['Day_Low_High_Change'] = gold_data['High'] - gold_data['Low']
    gold_data['Open_Close_Change'] = gold_data['Close'] - gold_data['Open']
    gold_data['5_Day_MA'] = gold_data['Close'].rolling(window=5).mean()
    gold_data['10_Day_MA'] = gold_data['Close'].rolling(window=10).mean()
    gold_data['20_Day_MA'] = gold_data['Close'].rolling(window=20).mean()
    gold_data['50_Day_MA'] = gold_data['Close'].rolling(window=50).mean()
    gold_data['RSI'] = calculate_rsi(gold_data['Close'])
    
    return gold_data

def predict_gold_price(gold_data):
    # Prepare features and target
    gold_data['Next_Day_Close'] = gold_data['Close'].shift(-1)  # Target variable
    gold_data = gold_data.dropna()  # Drop rows with missing values
    
    # Features (independent variables)
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Day_Low_High_Change', 
                'Open_Close_Change', '5_Day_MA', '10_Day_MA', '20_Day_MA', '50_Day_MA', 'RSI']
    X = gold_data[features]
    y = gold_data['Next_Day_Close']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict the next day's close price
    predicted_price = model.predict(X_test)
    
    # Evaluate the model (optional)
    score = model.score(X_test, y_test)
    print(f"Model R^2 Score: {score}")
    
    # Predict the next day's close price for the latest data point
    latest_data = gold_data.iloc[-1][features].values.reshape(1, -1)
    next_day_prediction = model.predict(latest_data)
    print(f"Predicted Next Day Close Price: {next_day_prediction[0]}")
    
    return next_day_prediction[0]

if __name__ == "__main__":
    # Specify the date range (fetch data for the last 2 years)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365 * 2)).strftime('%Y-%m-%d')
    
    # Fetch the main data
    main_df = fetch_main_data(start_date, end_date)
    
    # Fetch detailed gold data
    gold_df = fetch_gold_data(start_date, end_date)
    
    if not gold_df.empty:
        # Predict the next day's gold price
        predicted_price = predict_gold_price(gold_df)
        
        # Add the predicted price to the main DataFrame
        main_df['Predicted_Gold_Close'] = predicted_price
        
        # Generate a dynamic filename based on the current date
        current_date = datetime.now().strftime('%d%m')  # Format: DDMM
        output_filename = f"gold_predictions_{current_date}.csv"
        
        # Save to a CSV file or display
        print(main_df.tail())
        main_df.to_csv(output_filename, index=True)
        print(f"Data saved to {output_filename}")