# Grid Search
# R-Squared Value: 0.17907
# Explained Variance Score: 0.40698
# Mean Absolute Error: 142.73
# Mean Squared Error: 69481.11
# Root Mean Squared Error: 263.59
# Cross-validated MSE: 0.12333792931132374

# Manual Tuning
# R-Squared Value: 0.24053
# Explained Variance Score: 0.43725
# Mean Absolute Error: 137.33
# Mean Squared Error: 64279.21
# Root Mean Squared Error: 253.53
# Cross-validated MSE: 0.14923463471346327

import warnings
import matplotlib
matplotlib.use('TkAgg')
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

# --------------------------------------------------------------------
# GoldDataset remains unchanged.
class GoldDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --------------------------------------------------------------------
# Data loading and feature engineering functions remain unchanged.
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

def prepare_data(df, train_ratio=0.7):
    primary_features = ['Silver', 'S&P500', 'cpi']
    
    # Feature Engineering
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
    
    # Fill missing values
    df.fillna(method='ffill', inplace=True)  # Forward fill missing values
    
    # Lag Features
    df['Gold_Lag1'] = df['Gold'].shift(1)
    df['Gold_Lag3'] = df['Gold'].shift(3)
    df['Gold_Lag5'] = df['Gold'].shift(5)
    df['Gold_Lag10'] = df['Gold'].shift(10)
    df['Gold_Lag30'] = df['Gold'].shift(30)
    df['Gold_Lag50'] = df['Gold'].shift(50)

    df['Gold_Log'] = np.log(df['Gold'])
    df['Gold_Silver'] = df['Gold'] * df['Silver']
    df['Gold_Silver_Ratio_SNP200'] = df['Gold_Silver_Ratio'] * df['S&P500']

    df['MACD'] = df['Gold'].ewm(span=12, adjust=False).mean() - df['Gold'].ewm(span=26, adjust=False).mean()
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    features = primary_features + [
        'Gold_MA5', 'Gold_MA10', 'Gold_EMA5', 'Gold_EMA10', 'RSI',
        'Gold_Return', 'Silver_Return', 'SP500_Return',
        'Gold_Vol', 'Gold_Silver_Ratio', 'Gold_ROC',
        'Gold_Silver_Change', 'Price_Momentum'
        # 'Gold_Lag1', 'Gold_Lag3', 'Gold_Lag5', 'Gold_Lag10', 'Gold_Lag30', 'Gold_Lag50',
        # 'Gold_Log', 'Gold_Silver', 'Gold_Silver_Ratio_SNP200',
        # 'MACD', 'MACD_Signal'
    ]
    
    # Drop rows with NaN values after lag
    df.dropna(inplace=True)
    
    df_sorted = df.sort_values('Date')
    cutoff = int(len(df_sorted) * train_ratio)
    train_df = df_sorted.iloc[:cutoff]
    test_df = df_sorted.iloc[cutoff:]
    
    X_train = train_df[features].values
    y_train = train_df['Gold'].values.reshape(-1, 1)
    X_test = test_df[features].values
    y_test = test_df['Gold'].values.reshape(-1, 1)
    
    # Scaling the features and target
    feature_scaler = StandardScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)
    
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train)
    y_test_scaled = target_scaler.transform(y_test)
    
    # Return the datasets
    train_dataset = GoldDataset(X_train_scaled, y_train_scaled)
    test_dataset = GoldDataset(X_test_scaled, y_test_scaled)
    
    return train_dataset, test_dataset, feature_scaler, target_scaler

def create_correlation_heatmap(df):
    correlation = df.drop('Date', axis=1).corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='RdBu_r', vmin=-1, vmax=1, fmt='.2f')
    plt.title('Correlation Heatmap of Gold Price Features')
    plt.tight_layout()
    plt.show()

# --------------------------------------------------------------------
# New function to tune the RF model using Grid Search.
def train_random_forest(train_dataset):
    X_train_np = train_dataset.X.numpy()
    y_train_np = train_dataset.y.numpy().ravel()

    # Create and train Random Forest model
    # rf_model = RandomForestRegressor(
    #     n_estimators=147,       # Number of trees in the forest
    #     max_depth=40,           # Maximum depth of the tree
    #     max_features=None,      # Number of features to consider when looking for the best split
    #     min_samples_split=2,    # Minimum number of samples required to split an internal node
    #     min_samples_leaf=2,     # Minimum number of samples required to be at a leaf node
    #     bootstrap=False,         # Whether bootstrap samples are used when building trees
    #     random_state=42,        # Random seed for reproducibility
    #     n_jobs=-1               # Use all available cores
    # )
    
    # # Train the model
    # rf_model.fit(X_train_np, y_train_np)

    # return rf_model

    # Define the parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [20, 30, 40, None],
        'max_features': ['auto', 'sqrt', 'log2', None],
        'min_samples_split': [2, 10, 20],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False],
        'criterion': ['absolute_error', 'squared_error', 'friedman_mse', 'poisson'],
        'min_weight_fraction_leaf': [0.0, 0.1, 0.2],
        'max_samples': [0.5, 0.7, 1.0],
        'oob_score': [True, False],
        'warm_start': [True, False],
        'max_leaf_nodes': [None, 10, 20, 30]
    }

    rf_model = RandomForestRegressor(random_state=42)

    # grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    # grid_search.fit(X_train_np, y_train_np)

    # # Get the best model
    # best_rf_model = grid_search.best_estimator_

    rand_search = RandomizedSearchCV(rf_model, param_distributions=param_grid, n_iter=100, cv=5, n_jobs=-1, verbose=2)
    rand_search.fit(X_train_np, y_train_np)

    best_rf_model = rand_search.best_estimator_

    return best_rf_model
    

# --------------------------------------------------------------------
# Modified evaluation function for RF.
def evaluate_model(model, test_dataset, target_scaler):
    X_test_np = test_dataset.X.numpy()
    y_test_np = test_dataset.y.numpy()

    y_pred_scaled = model.predict(X_test_np)
    y_pred_original = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
    y_test_original = target_scaler.inverse_transform(y_test_np)

    r2 = r2_score(y_test_original, y_pred_original)
    evs = explained_variance_score(y_test_original, y_pred_original)
    mae = mean_absolute_error(y_test_original, y_pred_original)
    mse = mean_squared_error(y_test_original, y_pred_original)
    rmse = np.sqrt(mse)

    print("RF Regression Results:")
    print(f"R-Squared Value: {r2:.5f}")
    print(f"Explained Variance Score: {evs:.5f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")

    # Cross-validated MSE
    scores = cross_val_score(model, X_test_np, y_test_np, cv=5, scoring='neg_mean_squared_error')
    print(f"Cross-validated MSE: {-scores.mean()}")

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_original, y_pred_original, alpha=0.5)
    plt.plot([y_test_original.min(), y_test_original.max()],
             [y_test_original.min(), y_test_original.max()],
             'r--', lw=2)
    plt.xlabel('Actual Gold Price')
    plt.ylabel('Predicted Gold Price')
    plt.title('Predicted vs Actual Gold Prices (RF)')
    plt.show()

# --------------------------------------------------------------------
# prepare_future_data and predict_future functions remain unchanged.
def prepare_future_data(df, feature_scaler, num_days=7):
    daily_changes = df['Gold'].pct_change().dropna()
    daily_std = daily_changes.std()
    max_daily_change = daily_std * 0.75

    last_data = df.iloc[-1].copy()
    last_price = df['Gold'].iloc[-1]

    market_holidays = ['2025-01-01']
    future_dates = pd.date_range(start=df.iloc[-1]['Date'] + pd.Timedelta(days=1),
                                 periods=num_days, freq='B')
    future_dates = future_dates[~future_dates.strftime('%Y-%m-%d').isin(market_holidays)]

    future_data = []
    previous_pred = last_data.copy()
    historical_prices = list(df['Gold'].tail(10))
    avg_daily_move = abs(daily_changes.tail(30)).mean()

    for i, future_date in enumerate(future_dates):
        new_row = previous_pred.copy()
        new_row['Date'] = future_date

        max_move = min(last_price * max_daily_change, 25.0)
        volatility_adjustment = np.random.normal(0, max_move / 5)

        if i > 0:
            prev_trend = (future_data[-1]['Gold'] - future_data[-2]['Gold']) if i > 1 else 0
            volatility_adjustment += prev_trend * 0.3

        if i == 0:
            new_price = last_price + volatility_adjustment
        else:
            prev_price = future_data[-1]['Gold']
            new_price = prev_price + volatility_adjustment

        max_up = (last_price if i == 0 else prev_price) + max_move
        max_down = (last_price if i == 0 else prev_price) - max_move
        new_price = np.clip(new_price, max_down, max_up)

        new_row['Gold'] = new_price
        historical_prices.append(new_price)
        new_row['Gold_MA5'] = np.mean(historical_prices[-5:])
        new_row['Gold_MA10'] = np.mean(historical_prices[-10:])

        if i == 0:
            new_row['Gold_EMA5'] = df['Gold'].tail(5).ewm(span=5, adjust=False).mean().iloc[-1]
            new_row['Gold_EMA10'] = df['Gold'].tail(10).ewm(span=10, adjust=False).mean().iloc[-1]
        else:
            new_row['Gold_EMA5'] = new_price * 0.333 + future_data[-1]['Gold_EMA5'] * 0.667
            new_row['Gold_EMA10'] = new_price * 0.182 + future_data[-1]['Gold_EMA10'] * 0.818

        prev_gold = last_price if i == 0 else future_data[-1]['Gold']
        new_row['Gold_Return'] = (new_price - prev_gold) / prev_gold
        new_row['Gold_ROC'] = new_row['Gold_Return'] * 100
        new_row['Gold_Vol'] = np.std(historical_prices[-10:])
        avg_ratio = df['Gold_Silver_Ratio'].mean()
        new_row['Silver'] = new_price / avg_ratio

        future_data.append(new_row)
        previous_pred = new_row.copy()

    future_df = pd.DataFrame(future_data)

    for feature in ['S&P500', 'cpi']:
        feature_std = df[feature].std() * 0.05
        future_df[feature] = future_df[feature].mean() + np.random.normal(0, feature_std, size=len(future_df))

    features = ['Silver', 'S&P500', 'cpi', 'Gold_MA5', 'Gold_MA10', 'Gold_EMA5', 'Gold_EMA10',
                'RSI', 'Gold_Return', 'Silver_Return', 'SP500_Return', 'Gold_Vol',
                'Gold_Silver_Ratio', 'Gold_ROC', 'Gold_Silver_Change', 'Price_Momentum']
    X_future = future_df[features].values
    X_future_scaled = feature_scaler.transform(X_future)
    return future_dates, X_future_scaled

def predict_future(model, df, feature_scaler, target_scaler, num_days=7):
    future_dates, X_future_scaled = prepare_future_data(df, feature_scaler, num_days)
    predictions_scaled = model.predict(X_future_scaled)
    predictions = target_scaler.inverse_transform(predictions_scaled.reshape(-1, 1))
    results_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Gold_Price': predictions.flatten()
    })
    return results_df

# --------------------------------------------------------------------
# Main function now uses the tuned RF model.
def main():
    print("Loading data...")
    df = load_data('dataset/combined_dataset.csv')
    print(f"\nDataset shape: {df.shape}")

    print("\nGenerating correlation heatmap...")
    create_correlation_heatmap(df)

    print("\nPreparing data for modeling...")
    train_dataset, test_dataset, feature_scaler, target_scaler = prepare_data(df)

    print("\nTuning RF model...")
    rf_model = train_random_forest(train_dataset)
    print(rf_model.get_params())

    print("\nEvaluating RF model...")
    evaluate_model(rf_model, test_dataset, target_scaler)

    print("\nPredicting future gold prices...")
    future_predictions = predict_future(rf_model, df, feature_scaler, target_scaler, num_days=7)
    print("\nPredicted Gold Prices for next week:")
    print(future_predictions)

    plt.figure(figsize=(10, 6))
    plt.plot(future_predictions['Date'], future_predictions['Predicted_Gold_Price'],
             marker='o', linestyle='-', label='Predicted')
    plt.title('Gold Price Predictions for Next Week (Tuned RF)')
    plt.xlabel('Date')
    plt.ylabel('Gold Price')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
