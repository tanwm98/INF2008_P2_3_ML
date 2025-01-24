import pandas as pd
from datetime import datetime
import os

# Set paths
base_path = r'C:\Users\tanwm\Desktop\INF2008_P2_3_ML\dataset'
cpi_path = os.path.join(base_path, 'BETTERCPI.csv')
rates_path = os.path.join(base_path, 'interestrates_2020-2024.csv')
gold_path = os.path.join(base_path, 'gold_details.csv')
output_path = os.path.join(base_path, 'combined_data.csv')

# Read files
cpi_df = pd.read_csv(cpi_path)
rates_df = pd.read_csv(rates_path)
gold_df = pd.read_csv(gold_path)

# Convert dates 
cpi_df['observation_date'] = pd.to_datetime(cpi_df['observation_date'])
rates_df['date'] = pd.to_datetime(rates_df['date'], format='%d/%m/%Y')
gold_df['Date'] = pd.to_datetime(gold_df['Date'], format='%Y-%m-%d')

# Extract month and year
rates_df['month'] = rates_df['date'].dt.month
rates_df['year'] = rates_df['date'].dt.year
gold_df['month'] = gold_df['Date'].dt.month
gold_df['year'] = gold_df['Date'].dt.year
cpi_df['month'] = cpi_df['observation_date'].dt.month
cpi_df['year'] = cpi_df['observation_date'].dt.year

# Merge datasets
merged_df = pd.merge(gold_df, cpi_df[['month', 'year', 'CPIAUCSL']], on=['month', 'year'], how='left')
merged_df = pd.merge(merged_df, rates_df[['date', 'rates', 'month', 'year']], on=['month', 'year'], how='left')

# Clean up and format
merged_df = merged_df.sort_values('Date')
merged_df = merged_df.rename(columns={
    'Date': 'date',
    'CPIAUCSL': 'cpi',
    'Gold': 'gold_price',
    'Gold_Close': 'gold_close',
    'Gold_Volume': 'volume',
    'Gold_Day_Low_High_Change': 'day_range',
    'Gold_Open_Close_Change': 'day_change',
    'Gold_Day_Low_High_Change_%': 'day_range_pct',
    'Gold_Open_Close_Change_%': 'day_change_pct',
    'Gold_RSI': 'rsi'
})

# Keep relevant columns
columns_to_keep = ['date', 'gold_price', 'cpi', 'rates', 'gold_close', 'volume', 
                   'day_range', 'day_change', 'day_range_pct', 'day_change_pct', 'rsi']
merged_df = merged_df[columns_to_keep]

# Format date
merged_df['date'] = pd.to_datetime(merged_df['date']).dt.strftime('%d/%m/%Y')

# Save to CSV
merged_df.to_csv(output_path, index=False)

print("Data processing complete. Check combined_data.csv for results.")