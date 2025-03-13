import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, GridSearchCV, learning_curve, validation_curve

import random
from torch.utils.data import Dataset

# Set all random seeds for reproducibility
random.seed(42)
np.random.seed(42)
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')


class GoldDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).reshape(-1, 1)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


#######################################
# Data Loading and Feature Engineering
#######################################
def load_data(file_path):
    df = pd.read_csv(file_path)
    selected_columns = ['Date', 'Gold', 'Silver', 'Crude Oil', 'DXY', 'S&P500', 'cpi', 'rates']
    df = df[selected_columns].copy()
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df.dropna(inplace=True)
    return df


def process_features(df):
    """
    Enhanced feature engineering focusing ONLY on external factors - no historical gold price data
    """
    result = df.copy()

    # Create features for external economic indicators only
    numerical_cols = ['Silver', 'DXY', 'Crude Oil', 'rates', 'cpi', 'S&P500']
    for feature in numerical_cols:
        if feature in result.columns:
            # Create lags for external features
            for lag in [1, 3, 6, 12]:
                result[f'{feature}_Lag{lag}'] = result[feature].shift(lag)

            # Create moving averages
            for window in [7, 14, 30, 90]:
                result[f'{feature}_MA{window}'] = result[feature].rolling(window=window).mean()

            # Create momentum indicators
            for period in [3, 6, 12]:
                result[f'{feature}_Momentum{period}'] = result[feature].pct_change(period)

            # Volatility indicators
            for window in [14, 30]:
                result[f'{feature}_Volatility{window}'] = result[feature].pct_change().rolling(window=window).std()

    # Cross-asset relationships are important
    # Gold/Silver ratio is omitted as it contains gold price

    # Economic indicator relationships
    if 'Silver' in result.columns and 'DXY' in result.columns:
        result['Silver_DXY_Ratio'] = result['Silver'] / result['DXY']
        result['Silver_DXY_Ratio_Change'] = result['Silver_DXY_Ratio'].pct_change(5)

    if 'Crude Oil' in result.columns and 'DXY' in result.columns:
        result['Oil_DXY_Ratio'] = result['Crude Oil'] / result['DXY']
        result['Oil_DXY_Ratio_Change'] = result['Oil_DXY_Ratio'].pct_change(5)

    # Real interest rate features - important for precious metals
    if 'rates' in result.columns and 'cpi' in result.columns:
        result['real_rate'] = result['rates'] - result['cpi']
        for window in [7, 30, 90]:
            result[f'real_rate_MA{window}'] = result['real_rate'].rolling(window=window).mean()
        result['real_rate_change'] = result['real_rate'].diff()

    # Add seasonal components from date
    result['month'] = result['Date'].dt.month
    result['quarter'] = result['Date'].dt.quarter

    # Economic cycle indicators
    if 'S&P500' in result.columns and 'rates' in result.columns:
        result['SP500_Rate_Ratio'] = result['S&P500'] / result['rates']

    if 'Crude Oil' in result.columns and 'S&P500' in result.columns:
        result['Oil_SP500_Ratio'] = result['Crude Oil'] / result['S&P500']

    # IMPORTANT: Remove any columns containing gold price information
    gold_cols = [col for col in result.columns if 'Gold' in col]
    result = result.drop(gold_cols, axis=1)

    # Keep the target column for training
    if 'Gold' in df.columns:
        result['Gold'] = df['Gold']

    return result


def prepare_data(df, train_ratio=0.7, val_ratio=0.15):
    """Split the data into train/validation/test sets and process features.
       Returns the NumPy arrays, scalers, and the list of features.
    """
    df_sorted = df.sort_values('Date')
    # Optional: Warn about high correlations
    corr_with_gold = df_sorted.corr()['Gold'].abs().sort_values(ascending=False)
    high_corr_features = [f for f in corr_with_gold[corr_with_gold > 0.9].index.tolist() if f not in ['Gold', 'Date']]
    if high_corr_features:
        print("WARNING: The following features have very high correlation with Gold (potential leakage):")
        for feature in high_corr_features:
            print(f"- {feature}")

    n = len(df_sorted)
    train_cutoff = int(n * train_ratio)
    val_cutoff = int(n * (train_ratio + val_ratio))
    train_df = df_sorted.iloc[:train_cutoff].copy()
    val_df = df_sorted.iloc[train_cutoff:val_cutoff].copy()
    test_df = df_sorted.iloc[val_cutoff:].copy()

    train_df = process_features(train_df)
    val_df = process_features(val_df)
    test_df = process_features(test_df)
    train_df.dropna(inplace=True)
    val_df.dropna(inplace=True)
    test_df.dropna(inplace=True)

    # Build final feature list (exclude Gold and Date)
    features = [col for col in train_df.columns if col not in ['Gold', 'Date'] and 'Gold' not in col]
    print("Final features used in model training:")
    for f in features:
        print(f"- {f}")

    X_train = train_df[features].values
    y_train = train_df['Gold'].values
    X_val = val_df[features].values
    y_val = val_df['Gold'].values
    X_test = test_df[features].values
    y_test = test_df['Gold'].values

    feature_scaler = StandardScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_val_scaled = feature_scaler.transform(X_val)
    X_test_scaled = feature_scaler.transform(X_test)

    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1)).ravel()
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).ravel()

    return X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_test_scaled, y_test_scaled, feature_scaler, target_scaler, features


#######################################
# Two-Stage Hyperparameter Tuning
#######################################
def two_stage_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    """
    Enhanced hyperparameter tuning for gold price prediction using only external factors
    """
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)

    # Broader search for more complex feature relationships
    param_grid_broad = {
        'n_estimators': [100, 300, 500, 1000],
        'max_depth': [None, 20, 40, 60, 80],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 0.7, 0.9],
        'bootstrap': [True, False],
        'criterion': ['squared_error', 'absolute_error', 'friedman_mse']
    }

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import RandomizedSearchCV

    rf = RandomForestRegressor(random_state=42)
    rf_random = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_grid_broad,
        n_iter=50,  # Increased iterations for broader search
        cv=tscv,
        verbose=1,
        random_state=42,
        n_jobs=-1,
        scoring='neg_mean_squared_error'
    )

    rf_random.fit(X_train, y_train)
    best_params_stage1 = rf_random.best_params_
    print("Best parameters from broad search:")
    print(best_params_stage1)

    # Fine-tune around the best parameters
    from sklearn.model_selection import GridSearchCV

    param_grid_fine = {
        'n_estimators': [max(100, best_params_stage1['n_estimators'] - 100),
                         best_params_stage1['n_estimators'],
                         min(1500, best_params_stage1['n_estimators'] + 100)],
        'max_depth': [best_params_stage1['max_depth']] if best_params_stage1['max_depth'] is None
        else [max(10, best_params_stage1['max_depth'] - 10),
              best_params_stage1['max_depth'],
              best_params_stage1['max_depth'] + 10],
        'min_samples_split': [max(2, best_params_stage1['min_samples_split'] - 1),
                              best_params_stage1['min_samples_split'],
                              best_params_stage1['min_samples_split'] + 1],
        'min_samples_leaf': [max(1, best_params_stage1['min_samples_leaf'] - 1),
                             best_params_stage1['min_samples_leaf'],
                             best_params_stage1['min_samples_leaf'] + 1],
        'max_features': [best_params_stage1['max_features']],
        'criterion': [best_params_stage1['criterion']]
    }

    rf_grid = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42, bootstrap=best_params_stage1['bootstrap']),
        param_grid=param_grid_fine,
        cv=tscv,
        verbose=1,
        n_jobs=-1,
        scoring='neg_mean_squared_error'
    )

    rf_grid.fit(X_train, y_train)
    best_params_stage2 = rf_grid.best_params_
    print("\nBest parameters after fine-tuning:")
    print(best_params_stage2)

    # Train the final model with best parameters
    final_params = best_params_stage2.copy()
    final_params['bootstrap'] = best_params_stage1['bootstrap']
    final_model = RandomForestRegressor(**final_params, random_state=42)

    # Train on combined train and validation set for final model
    import numpy as np
    X_combined = np.vstack((X_train, X_val))
    y_combined = np.hstack((y_train, y_val))
    final_model.fit(X_combined, y_combined)

    return final_model, final_params


#######################################
# Model Evaluation
#######################################
def evaluate_model(model, X_test, y_test, target_scaler):
    """Evaluate and visualize model performance."""
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)

    print("\nTest Set Performance Metrics:")
    print(f"R²: {r2:.5f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"Explained Variance Score: {evs:.5f}")

    # Convert predictions back to original scale for visualization
    y_test_orig = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # Plot actual vs predicted values side by side
    plt.figure(figsize=(12, 6))
    plt.title('Actual Price vs Predicted Price')
    x_axis = np.arange(len(y_test_orig))
    width = 0.35  # Width of the bars

    # Interleave the points for better visualization
    plt.plot(x_axis, y_test_orig, 'b-', label='Actual Value', alpha=0.7)

    plt.xlabel('Number of Values')
    plt.ylabel('Gold Price')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return r2, rmse, mae, evs


#######################################
# Main Function
#######################################
def main():
    print("Loading data...")
    df = load_data('dataset/combined_dataset.csv')
    print(f"Dataset shape: {df.shape}")

    # Prepare data (returning NumPy arrays for tuning and evaluation)
    X_train, y_train, X_val, y_val, X_test, y_test, feature_scaler, target_scaler, features = prepare_data(
        df, train_ratio=0.7, val_ratio=0.15)

    print("\nPerforming hyperparameter tuning...")
    best_model, best_params = two_stage_hyperparameter_tuning(X_train, y_train, X_val, y_val)

    # Evaluate final model on test set
    evaluate_model(best_model, X_test, y_test, target_scaler)


if __name__ == "__main__":
    main()
