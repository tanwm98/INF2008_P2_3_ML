import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, cross_val_score
import torch
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')


# --------------------------
# Define a PyTorch dataset for our data
class GoldDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# --------------------------
# Data Loading Function
def load_data(file_path):
    df = pd.read_csv(file_path)
    selected_columns = ['Date', 'Gold', 'Silver', 'Crude Oil', 'DXY', 'S&P500', 'cpi', 'rates']
    df = df[selected_columns].copy()
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df.dropna(inplace=True)
    return df


# --------------------------
# RSI Calculation Function
def calculate_rsi(data, periods=14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0).rolling(window=periods).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=periods).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# --------------------------
# Data Preparation with Extended Lag/Technical Features
def prepare_data_rf_with_lag_rsi(df, train_ratio=0.7):
    """
    Prepare data for RF regression without scaling.
    Adds:
      - Lags (1 to 5 days) for Gold
      - Momentum (5-day and 20-day)
      - Volatility (rolling std over 5 and 20 days)
      - RSI (14-day and 7-day)
      - Moving averages (5-day, 20-day) and their crossover
      - Rolling correlation between Gold and Silver
      - Ratio between Gold and S&P500
    """
    df_sorted = df.sort_values('Date')
    cutoff = int(len(df_sorted) * train_ratio)
    train_df = df_sorted.iloc[:cutoff].copy()
    test_df = df_sorted.iloc[cutoff:].copy()

    # Exogenous base features remain (if needed later)
    base_features = ['Silver', 'S&P500', 'cpi', 'Crude Oil', 'DXY', 'rates']
    base_features = [f for f in base_features if f in df.columns]

    for dataset in [train_df, test_df]:
        # Replace raw lag features with percentage changes
        for lag in range(1, 6):
            dataset[f'Gold_Lag{lag}_Pct'] = dataset['Gold'].pct_change(lag)
            # Keep one raw lag for reference but less emphasis
            if lag == 1:
                dataset[f'Gold_Lag{lag}'] = dataset['Gold'].shift(lag)

        # Other features remain the same
        dataset['Gold_MOM5'] = dataset['Gold'].diff(5)
        dataset['Gold_MOM20'] = dataset['Gold'].diff(20)

        # Volatility measures
        dataset['Gold_Volatility_5'] = dataset['Gold'].rolling(window=5).std()
        dataset['Gold_Volatility_20'] = dataset['Gold'].rolling(window=20).std()

        # RSI indicators with different periods
        dataset['Gold_RSI14'] = calculate_rsi(dataset['Gold'], 14)
        dataset['Gold_RSI7'] = calculate_rsi(dataset['Gold'], 7)

        # Moving averages and their crossover
        dataset['Gold_MA5'] = dataset['Gold'].rolling(window=5).mean()
        dataset['Gold_MA20'] = dataset['Gold'].rolling(window=20).mean()
        dataset['Gold_MA_Cross'] = dataset['Gold_MA5'] - dataset['Gold_MA20']

        # Correlation feature: rolling correlation between Gold and Silver (20-day window)
        dataset['Gold_Silver_Corr'] = dataset['Gold'].rolling(window=20).corr(dataset['Silver'])

        # Ratio feature between Gold and S&P500
        dataset['Gold_SP500_Ratio'] = dataset['Gold'] / dataset['S&P500']
        # Add ratio features
        dataset['Gold_to_Oil_Ratio'] = dataset['Gold'] / dataset['Crude Oil']
        dataset['Gold_to_Silver_Ratio'] = dataset['Gold'] / dataset['Silver']

        # Add rate of change for economic indicators
        dataset['Interest_Rate_Change'] = dataset['rates'].diff()
        dataset['Inflation_Change'] = dataset['cpi'].diff()

        # Add relative strength to economic indicators
        dataset['Gold_to_SP500_Change'] = dataset['Gold'].pct_change() - dataset['S&P500'].pct_change()
        dataset['Gold_Dollar_Strength'] = dataset['Gold'].pct_change() - dataset['DXY'].pct_change()

        # Add volatility ratios
        dataset['Gold_Vol_to_SP500_Vol'] = dataset['Gold'].rolling(20).std() / dataset['S&P500'].rolling(20).std()

        # (Optionally include other exogenous features engineering if desired)

    # Drop rows with missing values due to rolling calculations and lags
    train_df.dropna(inplace=True)
    test_df.dropna(inplace=True)

    # Define feature columns (exclude Date and target Gold)
    feature_columns = [col for col in train_df.columns if col not in ['Date', 'Gold']]
    # Ensure features exist in both train and test sets
    features = [col for col in feature_columns if col in test_df.columns]

    X_train = train_df[features].values
    y_train = train_df['Gold'].values
    X_test = test_df[features].values
    y_test = test_df['Gold'].values

    train_dataset = GoldDataset(X_train, y_train)
    test_dataset = GoldDataset(X_test, y_test)

    return train_dataset, test_dataset, features


def rolling_window_prediction(df, features, target='Gold', window_size=90, test_size=30, step_size=5):
    """
    Implements a rolling window approach to predict gold prices.

    Args:
        df: Dataframe with features and target
        features: List of feature column names
        target: Target column name
        window_size: Size of training window
        test_size: Size of testing/forecasting window
        step_size: Number of steps to move window forward

    Returns:
        DataFrame with actual and predicted values
    """
    df_sorted = df.sort_values('Date').reset_index(drop=True)
    results = []

    # Prepare features
    for lag in range(1, 6):
        df_sorted[f'Gold_Lag{lag}_Pct'] = df_sorted['Gold'].pct_change(lag)

    df_sorted['Gold_to_Oil_Ratio'] = df_sorted['Gold'] / df_sorted['Crude Oil']
    df_sorted['Gold_to_Silver_Ratio'] = df_sorted['Gold'] / df_sorted['Silver']
    # Add other feature engineering as needed

    df_sorted.dropna(inplace=True)

    # Define best parameters from previous tuning
    best_params = {
        'n_estimators': 400,
        'max_depth': 30,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt'
    }

    for i in range(0, len(df_sorted) - window_size - test_size, step_size):
        # Define window
        train_start = i
        train_end = i + window_size
        test_start = train_end
        test_end = test_start + test_size

        # Split data
        train_data = df_sorted.iloc[train_start:train_end].copy()
        test_data = df_sorted.iloc[test_start:test_end].copy()

        # Prepare training and test sets
        X_train = train_data[features].values
        y_train = train_data[target].values
        X_test = test_data[features].values
        y_test = test_data[target].values

        # Train model for this window
        model = RandomForestRegressor(**best_params, random_state=42)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Store results
        window_results = pd.DataFrame({
            'Date': test_data['Date'].values,
            'Actual': y_test,
            'Predicted': y_pred,
            'WindowStart': df_sorted.iloc[train_start]['Date'],
            'WindowEnd': df_sorted.iloc[train_end - 1]['Date']
        })

        results.append(window_results)

    all_results = pd.concat(results)

    # Calculate metrics for the complete set of predictions
    rmse = np.sqrt(mean_squared_error(all_results['Actual'], all_results['Predicted']))
    r2 = r2_score(all_results['Actual'], all_results['Predicted'])
    mae = mean_absolute_error(all_results['Actual'], all_results['Predicted'])

    print(f"Rolling Window Prediction Results:")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.5f}")
    print(f"MAE: {mae:.2f}")

    # Plot actual vs predicted
    plt.figure(figsize=(12, 6))
    plt.plot(all_results['Date'], all_results['Actual'], label='Actual')
    plt.plot(all_results['Date'], all_results['Predicted'], label='Predicted')
    plt.xlabel('Date')
    plt.ylabel('Gold Price')
    plt.title('Rolling Window Forecast of Gold Prices')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return all_results
# --------------------------
# Random Forest Training using RandomizedSearchCV with TimeSeriesSplit
def train_random_forest_rf(train_dataset):
    X_train_np = train_dataset.X.numpy()
    y_train_np = train_dataset.y.numpy().ravel()

    # Focused parameter grid based on previous tuning efforts
    param_grid = {
        'n_estimators': [200, 300, 400, 500],
        'max_depth': [20, 30, 40, 50, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }

    tscv = TimeSeriesSplit(n_splits=5)
    rand_search = RandomizedSearchCV(
        RandomForestRegressor(random_state=42),
        param_distributions=param_grid,
        n_iter=50,
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=2,
        random_state=42
    )

    rand_search.fit(X_train_np, y_train_np)
    print(f"Best parameters: {rand_search.best_params_}")
    return rand_search.best_estimator_


# --------------------------
# Feature Importance and Selection Function
def select_important_features(rf_model, features, X_train):
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("Feature ranking (top 20):")
    for f in range(min(20, len(features))):
        print(f"{f + 1}. {features[indices[f]]} ({importances[indices[f]]:.4f})")

    plt.figure(figsize=(12, 8))
    plt.title("Feature Importances")
    plt.bar(range(min(20, len(features))), importances[indices][:20], align="center")
    plt.xticks(range(min(20, len(features))), [features[i] for i in indices][:20], rotation=90)
    plt.tight_layout()
    plt.show()

    # Select top features that account for 70% cumulative importance
    cumulative_importance = np.cumsum(importances[indices])
    threshold_index = np.where(cumulative_importance >= 0.7)[0][0] + 1
    selected_features_indices = indices[:threshold_index]
    selected_features = [features[i] for i in selected_features_indices]

    print(f"Selected top {threshold_index} features explaining 70% of importance")
    return selected_features, X_train[:, selected_features_indices]


# --------------------------
# Ensemble Training Function
def train_ensemble(X_train, y_train):
    # Train individual models
    rf = RandomForestRegressor(n_estimators=400, max_depth=30, random_state=42)
    rf.fit(X_train, y_train)

    gbr = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
    gbr.fit(X_train, y_train)

    etr = ExtraTreesRegressor(n_estimators=300, max_depth=20, random_state=42)
    etr.fit(X_train, y_train)

    en = ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=42)
    en.fit(X_train, y_train)

    # Define an ensemble prediction (weighted average; weights can be tuned)
    def ensemble_predict(X_test):
        rf_pred = rf.predict(X_test)
        gbr_pred = gbr.predict(X_test)
        etr_pred = etr.predict(X_test)
        en_pred = en.predict(X_test)
        return (0.4 * rf_pred + 0.3 * gbr_pred + 0.2 * etr_pred + 0.1 * en_pred)

    return ensemble_predict


# --------------------------
# (Optional) Data Preparation with Stationary Transforms
def prepare_data_with_stationary_transforms(df, train_ratio=0.7):
    """
    Apply log transformation, differencing, and outlier removal to stabilize the time series.
    """
    df_sorted = df.sort_values('Date')
    cutoff = int(len(df_sorted) * train_ratio)
    train_df = df_sorted.iloc[:cutoff].copy()
    test_df = df_sorted.iloc[cutoff:].copy()

    for dataset in [train_df, test_df]:
        # Log transform
        dataset['Gold_Log'] = np.log(dataset['Gold'])
        dataset['Silver_Log'] = np.log(dataset['Silver'])
        # Difference transforms
        dataset['Gold_Diff'] = dataset['Gold'].diff()
        dataset['Gold_Diff_Log'] = dataset['Gold_Log'].diff()
        # Percentage change (stationary)
        dataset['Gold_Return'] = dataset['Gold'].pct_change()
        # Remove outliers using IQR method on returns
        z_scores = stats.zscore(dataset['Gold_Return'].dropna())
        abs_z_scores = np.abs(z_scores)
        mask = abs_z_scores < 3
        dataset.loc[dataset['Gold_Return'].dropna().index[~mask], 'Gold_Return'] = np.nan

    train_df.dropna(inplace=True)
    test_df.dropna(inplace=True)

    feature_columns = [col for col in train_df.columns if col not in ['Date', 'Gold']]
    features = [col for col in feature_columns if col in test_df.columns]

    X_train = train_df[features].values
    y_train = train_df['Gold'].values
    X_test = test_df[features].values
    y_test = test_df['Gold'].values

    train_dataset = GoldDataset(X_train, y_train)
    test_dataset = GoldDataset(X_test, y_test)
    return train_dataset, test_dataset, features


# --------------------------
# (Optional) Rolling Window Training/Evaluation
def train_test_rolling_window(df, window_size=90, test_size=30):
    """
    Train and test using a rolling window to better capture temporal dynamics.
    Returns a DataFrame with RMSE and R2 for each window.
    """
    results = []
    df_sorted = df.sort_values('Date').reset_index(drop=True)

    for i in range(len(df_sorted) - window_size - test_size):
        train_data = df_sorted.iloc[i: i + window_size]
        test_data = df_sorted.iloc[i + window_size: i + window_size + test_size]

        # Use the lag/RSI based preparation on the window (here we use a simplified version)
        # For a full pipeline, you could call prepare_data_rf_with_lag_rsi on the subset.
        features = [col for col in train_data.columns if col not in ['Date', 'Gold']]
        X_train = train_data[features].values
        y_train = train_data['Gold'].values
        X_test = test_data[features].values
        y_test = test_data['Gold'].values

        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        results.append({'window_start': train_data.iloc[0]['Date'], 'rmse': rmse, 'r2': r2})

    return pd.DataFrame(results)


# --------------------------
# Main function: choose which pipeline to run
def main():
    print("Loading data...")
    df = load_data('dataset/combined_dataset.csv')
    print(f"Dataset shape: {df.shape}")

    # Step 1: First prepare the data with improved features
    print("\nPreparing data with improved features...")
    train_dataset, test_dataset, features = prepare_data_rf_with_lag_rsi(df)

    # Step 2: Run the rolling window prediction
    print("\nImplementing rolling window prediction...")
    # Get features that don't include the target or date
    all_features = [col for col in df.columns if col not in ['Date', 'Gold']]
    # Filter to only include features that have been engineered/prepared
    valid_features = [f for f in all_features if f in df.columns]

    # Run the rolling window prediction
    results = rolling_window_prediction(
        df,
        features=valid_features,
        target='Gold',
        window_size=90,  # 90 days training window
        test_size=30,  # 30 days test window
        step_size=10  # Move window 10 days at a time
    )


if __name__ == "__main__":
    main()
