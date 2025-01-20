import yfinance as yf
import pandas as pd
from datetime import datetime
import numpy as np

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

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
    
    # Fetch data for each ticker
    for name, ticker in tickers.items():
        print(f"Fetching data for {name}...")
        df = yf.download(ticker, start=start_date, end=end_date)
        if not df.empty:
            main_data[name] = df['Close']
        else:
            print(f"Warning: No data fetched for {name} ({ticker})")
    
    return main_data

def fetch_gold_data(start_date, end_date):
    print("Fetching detailed data for Gold...")
    gold_data = yf.download('GC=F', start=start_date, end=end_date)

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
    #gold_data['Day_Low_High_Change_%'] = (
    #    gold_data['Day_Low_High_Change'].astype(float) / gold_data['Low'].astype(float)
    #) * 100
    #gold_data['Open_Close_Change_%'] = (
    #    gold_data['Open_Close_Change'].astype(float) / gold_data['Open'].astype(float)
    #) * 100
    gold_data['5_Day_MA'] = gold_data['Close'].rolling(window=5).mean()
    gold_data['10_Day_MA'] = gold_data['Close'].rolling(window=10).mean()
    gold_data['20_Day_MA'] = gold_data['Close'].rolling(window=20).mean()
    gold_data['50_Day_MA'] = gold_data['Close'].rolling(window=50).mean()
    gold_data['RSI'] = calculate_rsi(gold_data['Close'])
    
    # Calculate trend: 1 for >0.5% up, 0 for -0.5% to 0.5%, -1 for <-0.5%
    #gold_data['Trend'] = gold_data['Open_Close_Change_%'].apply(
    #    lambda x: 1 if x > 0.5 else (-1 if x < -0.5 else 0)
    #)
    
    return gold_data



if __name__ == "__main__":
    # Specify the date range
    start_date = "2020-01-01"
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Fetch the main data
    main_df = fetch_main_data(start_date, end_date)
    
    # Fetch detailed gold data
    gold_df = fetch_gold_data(start_date, end_date)
    
    if not gold_df.empty:
        # Merge main data with detailed gold data
        detailed_gold_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume', 
            #'Day_Low_High_Change',
            'Open_Close_Change', 
            #'Day_Low_High_Change_%', 
            #'Open_Close_Change_%',
            '5_Day_MA', '10_Day_MA', '20_Day_MA', '50_Day_MA', 'RSI', 
            #'Trend',
            'Next_Day_Close'
        ]
        for col in detailed_gold_columns:
            main_df[f"Gold_{col}"] = gold_df[col]
        
        # Drop rows with missing values (e.g., due to rolling calculations)
        main_df = main_df.dropna()
        
        # Save to a CSV file or display
        print(main_df.head())
        main_df.to_csv("gold_now.csv", index=True)
