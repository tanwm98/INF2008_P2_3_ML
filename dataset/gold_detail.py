import yfinance as yf
import pandas as pd
from datetime import datetime


def fetch_main_data(start_date, end_date):
    """Fetches financial data for Gold, Silver, Crude Oil, DXY, and S&P 500."""
    tickers = {
        'Gold': 'GC=F',  # Gold futures
        'Silver': 'SI=F',  # Silver futures
        'Crude Oil': 'CL=F',  # Crude oil futures
        'DXY': 'DX-Y.NYB',  # US Dollar Index
        'S&P500': '^GSPC',  # S&P 500 Index
    }

    # Fetch data and retain only Close prices
    data = pd.DataFrame()
    for name, ticker in tickers.items():
        print(f"Fetching data for {name}...")
        df = yf.download(ticker, start=start_date, end=end_date)
        if not df.empty:
            data[name] = df['Close']
        else:
            print(f"Warning: No data fetched for {name} ({ticker})")

    # Add Date column, reset index, and format as DD-MM-YYYY
    data.reset_index(inplace=True)
    data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')  # Ensures proper format

    return data


if __name__ == "__main__":
    # Define date range
    start_date = "2015-01-01"
    end_date = "2025-01-01"

    # Fetch the required data
    main_df = fetch_main_data(start_date, end_date)

    # Save to CSV
    main_df.to_csv("financial_data.csv", index=False)

    # Display first few rows
    print(main_df.head())
