#!/usr/bin/env python
import os
import pandas as pd
import yfinance as yf


def fetch_financial_data(start_date, end_date):
    """
    Fetches financial data for Gold, Silver, Crude Oil, DXY, and S&P500.
    This follows the logic of new_dataset_gen.py.
    """
    tickers = {
        'Gold': 'GC=F',
        'Silver': 'SI=F',
        'Crude Oil': 'CL=F',
        'DXY': 'DX-Y.NYB',
        'S&P500': '^GSPC',
    }
    data = pd.DataFrame()
    for name, ticker in tickers.items():
        print(f"Fetching data for {name}...")
        df = yf.download(ticker, start=start_date, end=end_date)
        if not df.empty:
            data[name] = df['Close']
        else:
            print(f"Warning: No data fetched for {name} ({ticker})")
    data.reset_index(inplace=True)
    # Format Date exactly as in new_dataset_gen.py (YYYY-MM-DD)
    data['Date'] = pd.to_datetime(data['Date']).dt.strftime('%Y-%m-%d')
    return data


def generate_compiled_data(start_date, end_date, output_path):
    """
    Generates compiled_data.csv using the financial data.
    """
    data = fetch_financial_data(start_date, end_date)
    data.to_csv(output_path, index=False)
    print(f"Compiled financial data saved to {output_path}")
    return data


def process_cpi(cpi_path):
    """
    Reads BETTERCPI.csv, converts dates, and aggregates to a unique
    month-year record. Here we take the mean if there are duplicates.
    """
    cpi_df = pd.read_csv(cpi_path)
    cpi_df['observation_date'] = pd.to_datetime(cpi_df['observation_date'])
    cpi_df['month'] = cpi_df['observation_date'].dt.month
    cpi_df['year'] = cpi_df['observation_date'].dt.year
    # Aggregate so that each (year, month) appears only once
    cpi_df = cpi_df.groupby(['year', 'month'], as_index=False).agg({'CPIAUCSL': 'mean'})
    cpi_df.rename(columns={'CPIAUCSL': 'cpi'}, inplace=True)
    return cpi_df


def process_rates(rates_path):
    """
    Reads the interest rates CSV, converts dates (using the given format),
    and aggregates so that each month-year appears only once.
    """
    rates_df = pd.read_csv(rates_path)
    rates_df['date'] = pd.to_datetime(rates_df['date'], format='%d/%m/%Y')
    rates_df['month'] = rates_df['date'].dt.month
    rates_df['year'] = rates_df['date'].dt.year
    # Aggregate rates by taking the mean (you can adjust this as needed)
    rates_df = rates_df.groupby(['year', 'month'], as_index=False).agg({'rates': 'mean'})
    return rates_df


def merge_datasets(dataset_dir):
    # Define file paths
    cpi_path = os.path.join(dataset_dir, 'CPI_2015-2024.csv')
    rates_path = os.path.join(dataset_dir, 'interestrates_2015-2024.csv')
    compiled_path = os.path.join(dataset_dir, 'stock_market.csv')
    output_path = os.path.join(dataset_dir, 'combined_dataset.csv')

    # Read compiled financial data and extract month/year
    finance_df = pd.read_csv(compiled_path)
    finance_df['Date'] = pd.to_datetime(finance_df['Date'], format='%Y-%m-%d')
    finance_df['month'] = finance_df['Date'].dt.month
    finance_df['year'] = finance_df['Date'].dt.year

    # Process CPI and rates to get unique month-year rows
    cpi_df = process_cpi(cpi_path)
    rates_df = process_rates(rates_path)

    # Merge finance data with CPI and rates on year and month
    merged_df = pd.merge(finance_df, cpi_df, on=['year', 'month'], how='left')
    merged_df = pd.merge(merged_df, rates_df, on=['year', 'month'], how='left')

    # Drop helper columns used for merging
    merged_df.drop(['month', 'year'], axis=1, inplace=True)

    # Reorder columns to match desired output:
    # Date, Gold, Silver, Crude Oil, DXY, S&P500, cpi, rates
    merged_df = merged_df[['Date', 'Gold', 'Silver', 'Crude Oil', 'DXY', 'S&P500', 'cpi', 'rates']]

    # Optionally, format Date as DD/MM/YYYY for final output
    merged_df['Date'] = merged_df['Date'].dt.strftime('%d/%m/%Y')

    merged_df.to_csv(output_path, index=False)
    print(f"Merged data saved to {output_path}")


def main():
    # Set the directory where your CSV files reside
    dataset_dir = r'C:\Users\tanwm\Desktop\INF2008_P2_3_ML\dataset'
    compiled_path = os.path.join(dataset_dir, 'stock_market.csv')

    # Define date range (adjust as needed)
    start_date = "2015-01-01"
    end_date = "2025-01-01"

    # Step 1: Generate compiled_data.csv using the new_dataset_gen.py approach
    generate_compiled_data(start_date, end_date, compiled_path)

    # Step 2: Merge compiled_data.csv with BETTERCPI.csv and the rates CSV
    merge_datasets(dataset_dir)


if __name__ == "__main__":
    main()
