import warnings
import matplotlib
matplotlib.use('TkAgg')  # Set the backend to TkAgg
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from sklearn.linear_model import LinearRegression
import torch
from torch.utils.data import Dataset

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

def prepare_data(df, train_ratio=0.7):
    """Prepare data without using gold-derived features"""
    # First split data chronologically
    df_sorted = df.sort_values('Date')
    cutoff = int(len(df_sorted) * train_ratio)
    train_df = df_sorted.iloc[:cutoff].copy()
    test_df = df_sorted.iloc[cutoff:].copy()

    # Base exogenous features (not derived from Gold)
    base_features = ['Silver', 'S&P500', 'cpi', 'Crude Oil', 'DXY', 'rates']
    base_features = [f for f in base_features if f in df.columns]

    # Feature engineering separately on train and test
    for dataset in [train_df, test_df]:
        # Generate features from exogenous variables only
        dataset['Silver_Return'] = dataset['Silver'].pct_change()
        dataset['Silver_MA5'] = dataset['Silver'].rolling(window=5).mean()
        dataset['Silver_EMA10'] = dataset['Silver'].ewm(span=10, adjust=False).mean()

        dataset['SP500_Return'] = dataset['S&P500'].pct_change()
        dataset['SP500_MA10'] = dataset['S&P500'].rolling(window=10).mean()

        if 'Crude Oil' in dataset.columns:
            dataset['Oil_Return'] = dataset['Crude Oil'].pct_change()

        # Create ratio features (avoiding any that involve Gold)
        if 'Silver' in dataset.columns and 'Crude Oil' in dataset.columns:
            dataset['Silver_Oil_Ratio'] = dataset['Silver'] / dataset['Crude Oil']

    # Drop missing values from each set separately
    train_df.dropna(inplace=True)
    test_df.dropna(inplace=True)

    # Create feature list without any Gold-derived features
    feature_columns = base_features + [
        col for col in train_df.columns
        if col not in ['Gold', 'Date'] and 'Gold' not in col
    ]

    # Ensure all features exist in both datasets
    features = [f for f in feature_columns if f in train_df.columns and f in test_df.columns]

    # Prepare feature matrices and target vectors
    X_train = train_df[features].values
    y_train = train_df['Gold'].values
    X_test = test_df[features].values
    y_test = test_df['Gold'].values

    # Scale features using StandardScaler
    feature_scaler = StandardScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)

    # Scale target values using StandardScaler
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).ravel()

    # Create datasets from scaled data
    train_dataset = GoldDataset(X_train_scaled, y_train_scaled)
    test_dataset = GoldDataset(X_test_scaled, y_test_scaled)

    return train_dataset, test_dataset, feature_scaler, target_scaler


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


def evaluate_model(model, X_test, y_test, y_pred, model_name):
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

    # Feature importance for linear models
    if hasattr(model, 'coef_'):
        coef = model.coef_
        if coef.ndim > 1:
            coef = coef.ravel()

        # Hardcoded list you originally had
        hardcoded_features = ['Silver', 'Crude Oil', 'DXY', 'S&P500', 'cpi', 'rates']

        # Check if the number of coefficients matches the hardcoded feature list
        if len(coef) == len(hardcoded_features):
            feature_names = hardcoded_features
        else:
            # Otherwise, create generic feature names
            feature_names = [f"Feature {i}" for i in range(len(coef))]

        importances = pd.DataFrame({
            'Feature': feature_names,
            'Importance': np.abs(coef)
        })
        importances.sort_values('Importance', ascending=False, inplace=True)

        print("\nFeature Importance:")
        print(importances)

    # Visualize predicted vs actual values
    plot_predictions_vs_actual(y_test, y_pred, model_name)


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        'Linear Regression': LinearRegression(),
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results[name] = {
            'R2': r2_score(y_test, y_pred),
            'EVS': explained_variance_score(y_test, y_pred),
            'MAE': mean_absolute_error(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
        }

        evaluate_model(model, X_test, y_test, y_pred, name)

    return results


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
    train_dataset, test_dataset, feature_scaler, target_scaler = prepare_data(df, train_ratio=0.7)

    # Convert torch tensors to numpy arrays for training
    X_train_np = train_dataset.X.numpy()
    y_train_np = train_dataset.y.numpy().ravel()  # Flatten if needed
    X_test_np = test_dataset.X.numpy()
    y_test_np = test_dataset.y.numpy().ravel()

    # Train and evaluate models using numpy arrays
    print("\nTraining and evaluating models...")
    results = train_and_evaluate_models(X_train_np, X_test_np, y_train_np, y_test_np)


if __name__ == "__main__":
    main()
