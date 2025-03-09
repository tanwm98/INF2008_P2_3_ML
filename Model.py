import warnings
import matplotlib
import torch

matplotlib.use('TkAgg')  # Set the backend to TkAgg
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from sklearn.linear_model import LinearRegression
from torch.utils.data import Dataset, DataLoader

class GoldDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_data(file_path):
    # Read the dataset
    df = pd.read_csv(file_path)

    # Select columns from the new CSV
    selected_columns = ['Date', 'Gold', 'Silver', 'Crude Oil', 'DXY', 'S&P500', 'cpi', 'rates']
    df = df[selected_columns].copy()

    # Convert Date to datetime (DD/MM/YYYY in your CSV)
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

    # Handle missing values
    df.dropna(inplace=True)

    return df


def time_based_split(df, train_ratio=0.7):
    """
    Splits the DataFrame into train/test sets based on chronological order.
    train_ratio: fraction of data used for training.
    """
    df_sorted = df.sort_values('Date')
    cutoff = int(len(df_sorted) * train_ratio)
    train_df = df_sorted.iloc[:cutoff]
    test_df = df_sorted.iloc[cutoff:]
    return train_df, test_df


def prepare_data(df, train_ratio=0.7):
    """Prepare data for linear regression"""
    # Primary features based on correlation
    primary_features = ['Silver', 'S&P500', 'cpi']

    # Add other features if present
    for feature in ['Crude Oil', 'DXY', 'rates']:
        if feature in df.columns:
            primary_features.append(feature)

    # Technical indicators
    df['Gold_MA5'] = df['Gold'].rolling(window=5).mean()
    df['Gold_MA10'] = df['Gold'].rolling(window=10).mean()
    df['RSI'] = calculate_rsi(df['Gold'], periods=14)
    df['Gold_Return'] = df['Gold'].pct_change()
    df['Silver_Return'] = df['Silver'].pct_change()
    df['SP500_Return'] = df['S&P500'].pct_change()
    df['Gold_Vol'] = df['Gold'].rolling(window=10).std()
    df['Gold_Silver_Ratio'] = df['Gold'] / df['Silver']
    df['Gold_EMA5'] = df['Gold'].ewm(span=5, adjust=False).mean()
    df['Gold_EMA10'] = df['Gold'].ewm(span=10, adjust=False).mean()
    df['Gold_ROC'] = df['Gold'].pct_change(periods=5)
    df['Gold_Silver_Change'] = df['Gold_Return'] - df['Silver_Return']
    df['Price_Momentum'] = df['Gold_Return'].rolling(window=5).mean()

    # Drop missing values
    df.dropna(inplace=True)

    # Final features list
    features = primary_features + [
        'Gold_MA5', 'Gold_MA10', 'Gold_EMA5', 'Gold_EMA10', 'RSI',
        'Gold_Return', 'Silver_Return', 'SP500_Return',
        'Gold_Vol', 'Gold_Silver_Ratio', 'Gold_ROC',
        'Gold_Silver_Change', 'Price_Momentum'
    ]

    # Split chronologically to prevent data leakage
    df_sorted = df.sort_values('Date')
    cutoff = int(len(df_sorted) * train_ratio)
    train_df = df_sorted.iloc[:cutoff]
    test_df = df_sorted.iloc[cutoff:]

    # Prepare feature matrices
    X_train = train_df[features].values
    y_train = train_df['Gold'].values
    X_test = test_df[features].values
    y_test = test_df['Gold'].values

    # Scale features
    feature_scaler = StandardScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)

    # Scale target
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).ravel()

    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, feature_scaler, target_scaler, features


def create_correlation_heatmap(df):
    # Create correlation matrix using numeric columns (drop Date)
    correlation = df.drop('Date', axis=1).corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='RdBu_r', vmin=-1, vmax=1, fmt='.2f')
    plt.title('Correlation Heatmap of Gold Price Features')
    plt.tight_layout()
    plt.show()


def plot_predictions_vs_actual(y_test, y_pred, model_name):
    """
    Create a scatter plot comparing actual vs predicted values with a diagonal reference line.
    
    Parameters:
    - y_test: Numpy array of actual gold prices
    - y_pred: Numpy array of predicted gold prices
    - model_name: String name of the model for the plot title
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             'r--', lw=2)
    plt.xlabel('Actual Gold Price')
    plt.ylabel('Predicted Gold Price')
    plt.title(f'{model_name}: Predicted vs Actual Gold Prices')
    plt.tight_layout()
    plt.show()


def evaluate_model(model, X_test, y_test, y_pred, model_name, features):
    """Evaluate model performance and show feature importance"""
    r2 = r2_score(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"\n{model_name} Results:")
    print(f"R-Squared Value: {r2:.5f}")
    print(f"Explained Variance Score: {evs:.5f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")

    # Feature importance for linear regression
    coef = model.coef_
    if len(coef.shape) > 1:
        coef = coef.ravel()

    importances = pd.DataFrame({
        'Feature': features,
        'Importance': np.abs(coef)
    })
    importances.sort_values('Importance', ascending=False, inplace=True)

    print("\nFeature Importance:")
    print(importances)

    # Add the visualization for predicted vs actual values
    plot_predictions_vs_actual(y_test, y_pred, model_name)


def train_and_evaluate_models(X_train, X_test, y_train, y_test, features):
    """Train and evaluate linear regression model"""
    # Create and train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    # Store results
    results = {
        'Linear Regression': {
            'R2': r2,
            'EVS': evs,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse
        }
    }

    # Evaluate and display results
    evaluate_model(model, X_test, y_test, y_pred, 'Linear Regression', features)

    return results

def calculate_rsi(data, periods=14):
    # Calculate price differences
    delta = data.diff()

    # Make two series: one for lower closes and one for higher closes
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()

    # Calculate RSI
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def main():
    # Load data
    print("Loading data...")
    df = load_data('dataset/combined_dataset.csv')
    print(f"\nDataset shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())

    # Correlation heatmap
    print("\nGenerating correlation heatmap...")
    create_correlation_heatmap(df)

    # Prepare data for modeling
    print("\nPreparing data for modeling with a time-based split (70% train / 30% test)...")
    X_train, X_test, y_train, y_test, feature_scaler, target_scaler, features = prepare_data(df, train_ratio=0.7)

    # Train and evaluate model
    print("\nTraining and evaluating linear regression model...")
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test, features)

if __name__ == "__main__":
    main()
