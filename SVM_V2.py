import warnings
import matplotlib
matplotlib.use('TkAgg')
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_regression
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

#######################################
# Dataset Class
#######################################
class GoldDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

#######################################
# Data Preparation Functions
#######################################
def load_data(file_path):
    df = pd.read_csv(file_path)
    selected_columns = ['Date', 'Gold', 'Silver', 'Crude Oil', 'DXY', 'S&P500', 'cpi', 'rates']
    df = df[selected_columns].copy()
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df.dropna(inplace=True)
    return df

def calculate_rsi(data, periods=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def process_features(df, remove_gold=False):
    """Refined feature engineering"""
    result = df.copy()
    
    # Technical indicators for Gold
    if 'Gold' in result.columns:
        # RSI with different periods
        result['RSI_14'] = calculate_rsi(result['Gold'], periods=14)
        result['RSI_21'] = calculate_rsi(result['Gold'], periods=21)
        
        # Moving averages and ratios
        for window in [5, 10, 20, 30]:
            result[f'Gold_MA_{window}'] = result['Gold'].rolling(window=window).mean()
            result[f'Gold_MA_Ratio_{window}'] = result['Gold'] / result[f'Gold_MA_{window}']
        
        # Volatility indicators
        result['Gold_Volatility_20'] = result['Gold'].rolling(window=20).std()
        result['Gold_Volatility_30'] = result['Gold'].rolling(window=30).std()
        
        # Price momentum
        for period in [5, 10, 20]:
            result[f'Gold_ROC_{period}'] = result['Gold'].pct_change(period)
    
    # Market indicators
    if 'DXY' in result.columns:
        result['DXY_MA_10'] = result['DXY'].rolling(window=10).mean()
        result['DXY_MA_20'] = result['DXY'].rolling(window=20).mean()
        result['DXY_ROC_10'] = result['DXY'].pct_change(10)
        result['DXY_ROC_20'] = result['DXY'].pct_change(20)
        
    if 'Crude Oil' in result.columns:
        result['Oil_MA_10'] = result['Crude Oil'].rolling(window=10).mean()
        result['Oil_MA_20'] = result['Crude Oil'].rolling(window=20).mean()
        result['Oil_ROC_10'] = result['Crude Oil'].pct_change(10)
        result['Oil_ROC_20'] = result['Crude Oil'].pct_change(20)
    
    # Economic indicators
    if all(col in result.columns for col in ['rates', 'cpi']):
        result['Real_Rate'] = result['rates'] - result['cpi']
        result['Real_Rate_Change'] = result['Real_Rate'].diff()
        result['CPI_Change'] = result['cpi'].pct_change()
    
    # Handle missing values
    result = result.fillna(method='ffill').fillna(method='bfill')
    
    return result

def prepare_data(df, train_ratio=0.8, val_ratio=0.1):
    """Improved data preparation"""
    df_sorted = df.sort_values('Date')
    
    # Process features first
    processed_df = process_features(df_sorted, remove_gold=False)
    processed_df.dropna(inplace=True)
    
    # Split data
    n = len(processed_df)
    train_cutoff = int(n * train_ratio)
    val_cutoff = int(n * (train_ratio + val_ratio))

    train_df = processed_df.iloc[:train_cutoff].copy()
    val_df = processed_df.iloc[train_cutoff:val_cutoff].copy()
    test_df = processed_df.iloc[val_cutoff:].copy()

    # Define features to keep
    feature_columns = [col for col in processed_df.columns 
                      if col not in ['Date', 'Gold', 'Silver', 'S&P500'] 
                      and not col.startswith('Unnamed')]

    # Extract features and target
    X_train = train_df[feature_columns].values
    X_val = val_df[feature_columns].values
    X_test = test_df[feature_columns].values
    
    y_train = train_df['Gold'].values
    y_val = val_df['Gold'].values
    y_test = test_df['Gold'].values

    # Print feature importance if requested
    print("\nFeatures used:")
    for i, feature in enumerate(feature_columns):
        print(f"{i+1}. {feature}")

    # Scale features
    feature_scaler = RobustScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_val_scaled = feature_scaler.transform(X_val)
    X_test_scaled = feature_scaler.transform(X_test)

    # Scale target
    target_scaler = RobustScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1)).ravel()
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).ravel()

    return (X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled,
            X_test_scaled, y_test_scaled, feature_scaler, target_scaler, feature_columns)

def tune_model(X_train, y_train, X_val, y_val):
    """Enhanced model tuning"""
    param_grid = {
        'C': [1.0, 10.0, 100.0, 1000.0],
        'epsilon': [0.001, 0.01, 0.1],
        'gamma': ['scale', 'auto', 0.1, 0.01],
        'kernel': ['rbf'],
    }
    
    model = SVR()
    grid_search = GridSearchCV(
        model, 
        param_grid, 
        cv=5,
        scoring='r2',  # Changed to R2 scoring
        n_jobs=-1, 
        verbose=1
    )
    
    # Combine train and validation data
    X_combined = np.vstack([X_train, X_val])
    y_combined = np.concatenate([y_train, y_val])
    
    grid_search.fit(X_combined, y_combined)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def evaluate_model(model, test_dataset, target_scaler):
    """Enhanced model evaluation"""
    X_test_np = test_dataset.X.numpy()
    y_test_np = test_dataset.y.numpy().reshape(-1)
    
    # Make predictions
    y_pred_scaled = model.predict(X_test_np)
    
    # Inverse transform predictions and actual values
    y_pred_original = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(-1)
    y_test_original = target_scaler.inverse_transform(y_test_np.reshape(-1, 1)).reshape(-1)
    
    # Calculate metrics
    r2 = r2_score(y_test_original, y_pred_original)
    evs = explained_variance_score(y_test_original, y_pred_original)
    mae = mean_absolute_error(y_test_original, y_pred_original)
    mse = mean_squared_error(y_test_original, y_pred_original)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test_original - y_pred_original) / y_test_original)) * 100
    
    print("\nModel Evaluation Results:")
    print(f"R-squared (R²): {r2:.4f}")
    print(f"Explained Variance Score: {evs:.4f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")
    
    # Create visualization
    plt.figure(figsize=(15, 5))
    
    # Subplot 1: Scatter plot
    plt.subplot(1, 2, 1)
    plt.scatter(y_test_original, y_pred_original, alpha=0.5)
    plt.plot([y_test_original.min(), y_test_original.max()],
             [y_test_original.min(), y_test_original.max()],
             'r--', lw=2)
    plt.xlabel('Actual Gold Price')
    plt.ylabel('Predicted Gold Price')
    plt.title('Support Vector Regression: Predicted vs Actual Gold Prices')
    
    # Subplot 2: Time series plot
    plt.subplot(1, 2, 2)
    plt.plot(y_test_original, label='Actual', alpha=0.7)
    plt.plot(y_pred_original, label='Predicted', alpha=0.7)
    plt.title('Price Predictions Over Time')
    plt.xlabel('Time')
    plt.ylabel('Gold Price')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return {
        'R2': r2,
        'EVS': evs,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape
    }

def main():
    print("Loading data...")
    df = load_data('dataset/combined_dataset.csv')
    
    # Remove outliers
    Q1 = df['Gold'].quantile(0.25)
    Q3 = df['Gold'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df['Gold'] < (Q1 - 1.5 * IQR)) | (df['Gold'] > (Q3 + 1.5 * IQR)))]
    
    print("\nPreparing data...")
    (X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled,
     X_test_scaled, y_test_scaled, feature_scaler, target_scaler, features) = prepare_data(df)
    
    # Print shapes for debugging
    print(f"\nData shapes:")
    print(f"X_train: {X_train_scaled.shape}")
    print(f"y_train: {y_train_scaled.shape}")
    print(f"X_val: {X_val_scaled.shape}")
    print(f"y_val: {y_val_scaled.shape}")
    print(f"X_test: {X_test_scaled.shape}")
    print(f"y_test: {y_test_scaled.shape}")
    
    # Create datasets
    train_dataset = GoldDataset(X_train_scaled, y_train_scaled)
    val_dataset = GoldDataset(X_val_scaled, y_val_scaled)
    test_dataset = GoldDataset(X_test_scaled, y_test_scaled)
    
    print("\nTuning model...")
    best_model = tune_model(X_train_scaled, y_train_scaled, 
                          X_val_scaled, y_val_scaled)
    
    print("\nEvaluating final model...")
    results = evaluate_model(best_model, test_dataset, target_scaler)
    
    return best_model, results

if __name__ == "__main__":
    main()