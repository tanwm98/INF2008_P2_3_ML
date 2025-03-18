# Grid Search
# R-Squared Value: 0.17128
# Explained Variance Score: 0.40739
# Mean Absolute Error: 147.00
# Mean Squared Error: 70139.96
# Root Mean Squared Error: 264.84
# Cross-validated MSE: 0.13779729287659623

import warnings
import matplotlib

matplotlib.use('TkAgg')
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.tree import DecisionTreeRegressor

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
# Data loading function remains unchanged.
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


# --------------------------------------------------------------------
# Enhanced feature engineering function to create more external predictors
def engineer_features(dataset):
    """Create features from external factors only (no gold-derived features)"""
    result = dataset.copy()

    # Technical indicators for Silver
    if 'Silver' in result.columns:
        result['Silver_Return'] = result['Silver'].pct_change()
        result['Silver_MA5'] = result['Silver'].rolling(window=5).mean()
        result['Silver_MA20'] = result['Silver'].rolling(window=20).mean()
        result['Silver_EMA10'] = result['Silver'].ewm(span=10, adjust=False).mean()
        result['Silver_Vol'] = result['Silver'].pct_change().rolling(window=20).std()
        result['Silver_RSI'] = calculate_rsi(result['Silver'])

    # S&P 500 indicators
    if 'S&P500' in result.columns:
        result['SP500_Return'] = result['S&P500'].pct_change()
        result['SP500_MA10'] = result['S&P500'].rolling(window=10).mean()
        result['SP500_MA50'] = result['S&P500'].rolling(window=50).mean()
        result['SP500_Vol'] = result['S&P500'].pct_change().rolling(window=20).std()

    # Oil indicators
    if 'Crude Oil' in result.columns:
        result['Oil_Return'] = result['Crude Oil'].pct_change()
        result['Oil_MA10'] = result['Crude Oil'].rolling(window=10).mean()
        result['Oil_Vol'] = result['Crude Oil'].pct_change().rolling(window=15).std()

    # Economic indicators
    if 'cpi' in result.columns:
        result['CPI_Change'] = result['cpi'].pct_change()
        result['CPI_MA3'] = result['cpi'].rolling(window=3).mean()

    if 'rates' in result.columns:
        result['Rates_Change'] = result['rates'].diff()
        result['Rates_MA3'] = result['rates'].rolling(window=3).mean()

    if 'DXY' in result.columns:
        result['DXY_Return'] = result['DXY'].pct_change()
        result['DXY_MA10'] = result['DXY'].rolling(window=10).mean()
        result['DXY_RSI'] = calculate_rsi(result['DXY'])

    # Create ratio features (avoiding any that involve Gold)
    if 'Silver' in result.columns and 'Crude Oil' in result.columns:
        result['Silver_Oil_Ratio'] = result['Silver'] / result['Crude Oil']

    if 'DXY' in result.columns and 'Silver' in result.columns:
        result['Silver_DXY_Ratio'] = result['Silver'] / result['DXY']

    if 'cpi' in result.columns and 'rates' in result.columns:
        result['Real_Rate'] = result['rates'] - result['cpi']

    # Lag features for important variables
    for col in ['Silver', 'DXY', 'Crude Oil', 'rates', 'cpi']:
        if col in result.columns:
            for lag in [1, 3, 5]:
                result[f'{col}_Lag{lag}'] = result[col].shift(lag)

    return result


# --------------------------------------------------------------------
# Feature selection function to identify the most important external factors
def select_features(X, y, feature_names, k=10, method='combined'):
    """Select top k features using mutual information or f_regression"""
    # Check for high correlations among features
    X_df = pd.DataFrame(X, columns=feature_names)
    corr_matrix = X_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_pairs = [(corr_matrix.columns[i], corr_matrix.columns[j], upper.iloc[i, j])
                       for i in range(len(upper.columns))
                       for j in range(i + 1, len(upper.columns))
                       if upper.iloc[i, j] > 0.95]

    if high_corr_pairs:
        print("Warning: High correlation detected between features:")
        for feat1, feat2, corr in high_corr_pairs:
            print(f"  - {feat1} and {feat2}: {corr:.3f}")

    # Select features
    if method == 'mutual_info':
        selector = SelectKBest(mutual_info_regression, k=k)
    elif method == 'f_regression':
        selector = SelectKBest(f_regression, k=k)
    else:  # combined approach
        # First select with mutual info
        mi_selector = SelectKBest(mutual_info_regression, k=min(k * 2, len(feature_names)))
        X_mi = mi_selector.fit_transform(X, y)
        mi_mask = mi_selector.get_support()
        mi_features = [feature_names[i] for i in range(len(feature_names)) if mi_mask[i]]

        # Then refine with f_regression
        X_mi_df = pd.DataFrame(X_mi, columns=mi_features)
        fr_selector = SelectKBest(f_regression, k=k)
        X_selected = fr_selector.fit_transform(X_mi_df, y)
        fr_mask = fr_selector.get_support()
        selected_features = [mi_features[i] for i in range(len(mi_features)) if fr_mask[i]]

        # Create feature mask for original X
        feature_mask = np.zeros(len(feature_names), dtype=bool)
        for feat in selected_features:
            feature_mask[feature_names.index(feat)] = True

        return selected_features, X_selected, feature_mask

    # If using a single method
    X_selected = selector.fit_transform(X, y)
    feature_mask = selector.get_support()
    selected_features = [feature_names[i] for i in range(len(feature_names)) if feature_mask[i]]

    # Print feature importance scores
    scores = selector.scores_
    feature_scores = [(feature_names[i], scores[i]) for i in range(len(feature_names))]
    feature_scores.sort(key=lambda x: x[1], reverse=True)

    print(f"\nTop {k} features selected:")
    for feature, score in feature_scores[:k]:
        if feature in selected_features:
            print(f"  - {feature}: {score:.4f}")

    return selected_features, X_selected, feature_mask


# --------------------------------------------------------------------
# Improved prepare_data function with feature selection
def prepare_data(df, train_ratio=0.7, feature_selection=True, k_features=10):
    """Prepare data without using gold-derived features, with optional feature selection"""
    # First split data chronologically
    df_sorted = df.sort_values('Date')
    cutoff = int(len(df_sorted) * train_ratio)
    train_df = df_sorted.iloc[:cutoff].copy()
    test_df = df_sorted.iloc[cutoff:].copy()

    # Apply feature engineering
    train_df = engineer_features(train_df)
    test_df = engineer_features(test_df)

    # Drop missing values from each set separately
    train_df.dropna(inplace=True)
    test_df.dropna(inplace=True)

    # Create feature list without any Gold-derived features
    feature_columns = [col for col in train_df.columns
                       if col not in ['Gold', 'Date'] and 'Gold' not in col]

    # Ensure all features exist in both datasets
    features = [f for f in feature_columns if f in train_df.columns and f in test_df.columns]

    # Prepare feature matrices and target vectors
    X_train = train_df[features].values
    y_train = train_df['Gold'].values
    X_test = test_df[features].values
    y_test = test_df['Gold'].values

    # Perform feature selection if requested
    if feature_selection:
        print(f"Performing feature selection to select top {k_features} features...")
        selected_features, X_train_selected, feature_mask = select_features(
            X_train, y_train, features, k=k_features, method='combined')
        X_test_selected = X_test[:, feature_mask]

        # Update X_train and X_test
        X_train = X_train_selected
        X_test = X_test_selected

        # Update features list
        features = selected_features

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

    return train_dataset, test_dataset, feature_scaler, target_scaler, features


# --------------------------------------------------------------------
# Time Series Cross-Validation for AdaBoost
def time_series_cv_adaboost(df, n_splits=5, test_size=50):
    """Evaluate AdaBoost model using Time Series Cross-Validation"""
    df_sorted = df.sort_values('Date').reset_index(drop=True)
    df_processed = engineer_features(df_sorted)
    df_processed.dropna(inplace=True)

    # Define features (excluding Gold and Date)
    features = [col for col in df_processed.columns
                if col not in ['Gold', 'Date'] and 'Gold' not in col]

    cv_results = {
        'train_r2': [], 'test_r2': [],
        'train_rmse': [], 'test_rmse': [],
        'selected_features': [],
        'best_params': []
    }

    n_samples = len(df_processed)
    indices = []

    # Create time series splits
    for i in range(n_splits):
        test_end = n_samples - i * test_size
        test_start = test_end - test_size
        if test_start <= 0:
            continue
        indices.append((0, test_start, test_start, test_end))

    indices.reverse()  # Start with earliest test set

    for i, (train_start, train_end, test_start, test_end) in enumerate(indices):
        print(f"\nFold {i + 1}/{len(indices)}")
        print(f"Train: samples {train_start} to {train_end - 1}")
        print(f"Test: samples {test_start} to {test_end - 1}")

        train_df = df_processed.iloc[train_start:train_end]
        test_df = df_processed.iloc[test_start:test_end]

        X_train = train_df[features].values
        y_train = train_df['Gold'].values
        X_test = test_df[features].values
        y_test = test_df['Gold'].values

        # Feature selection
        k = min(10, len(features))
        selected_features, X_train_selected, feature_mask = select_features(
            X_train, y_train, features, k=k)
        X_test_selected = X_test[:, feature_mask]

        # Scaling
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train_selected)
        X_test_scaled = scaler_X.transform(X_test_selected)

        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

        # Define base estimator
        base_estimator = DecisionTreeRegressor(
            max_depth=3,
            min_samples_split=5,
            min_samples_leaf=2
        )

        # Create AdaBoost model with the custom base estimator
        adaboost = AdaBoostRegressor(
            estimator=base_estimator,
            n_estimators=100,
            learning_rate=0.05,
            loss='linear',
            random_state=42
        )

        # Define parameter grid - corrected to match estimator parameters
        param_grid = {
            'estimator__max_depth': [2, 3, 4],
            'estimator__min_samples_split': [3, 5, 10],
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'loss': ['linear', 'square']
        }

        grid_search = GridSearchCV(
            adaboost,
            param_grid,
            cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train_scaled, y_train_scaled)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        # Predictions
        y_train_pred = best_model.predict(X_train_scaled)
        y_test_pred = best_model.predict(X_test_scaled)

        # Transform back to original scale for metrics
        y_train_original = scaler_y.inverse_transform(y_train_scaled.reshape(-1, 1)).ravel()
        y_test_original = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).ravel()
        y_train_pred_original = scaler_y.inverse_transform(y_train_pred.reshape(-1, 1)).ravel()
        y_test_pred_original = scaler_y.inverse_transform(y_test_pred.reshape(-1, 1)).ravel()

        # Calculate metrics
        train_r2 = r2_score(y_train_original, y_train_pred_original)
        test_r2 = r2_score(y_test_original, y_test_pred_original)
        train_rmse = np.sqrt(mean_squared_error(y_train_original, y_train_pred_original))
        test_rmse = np.sqrt(mean_squared_error(y_test_original, y_test_pred_original))

        cv_results['train_r2'].append(train_r2)
        cv_results['test_r2'].append(test_r2)
        cv_results['train_rmse'].append(train_rmse)
        cv_results['test_rmse'].append(test_rmse)
        cv_results['selected_features'].append(selected_features)
        cv_results['best_params'].append(best_params)

        print(f"Best parameters: {best_params}")
        print(f"Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
        print(f"Train RMSE: {train_rmse:.2f}, Test RMSE: {test_rmse:.2f}")

    print("\nFeature Selection Frequency:")
    feature_counts = {}
    for f_list in cv_results['selected_features']:
        for feat in f_list:
            feature_counts[feat] = feature_counts.get(feat, 0) + 1

    for feat, count in sorted(feature_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {feat}: {count}/{len(indices)} folds")

    all_param_keys = set()
    for params in cv_results['best_params']:
        all_param_keys.update(params.keys())

    param_counts = {param: {} for param in all_param_keys}
    for params in cv_results['best_params']:
        for param, value in params.items():
            param_counts[param][value] = param_counts[param].get(value, 0) + 1

    for param, counts in param_counts.items():
        print(f"\n{param}:")
        for value, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {value}: {count}/{len(indices)} folds")

    return cv_results


# --------------------------------------------------------------------
# Train the final AdaBoost model with the best features and parameters
def train_adaboost_final(df, cv_results):
    """Train final AdaBoost model using insights from cross-validation"""
    # Get most frequently selected features
    feature_counts = {}
    for f_list in cv_results['selected_features']:
        for feat in f_list:
            feature_counts[feat] = feature_counts.get(feat, 0) + 1

    top_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
    selected_features = [feat for feat, count in top_features if count >= len(cv_results['selected_features']) / 2]

    if len(selected_features) < 5:  # Ensure we have at least 5 features
        selected_features = [feat for feat, _ in top_features[:5]]

    # Determine most common best parameters, with a guard in case CV didn't return any.
    if not cv_results['best_params']:
        print("No best parameters found in cross-validation; using default parameters.")
        best_n_estimators = 100
        best_learning_rate = 0.05
        best_loss = 'linear'
    else:
        param_counts = {'n_estimators': {}, 'learning_rate': {}, 'loss': {}}
        for params in cv_results['best_params']:
            for param, value in params.items():
                param_counts[param][value] = param_counts[param].get(value, 0) + 1

        best_n_estimators = max(param_counts['n_estimators'].items(), key=lambda x: x[1])[0]
        best_learning_rate = max(param_counts['learning_rate'].items(), key=lambda x: x[1])[0]
        best_loss = max(param_counts['loss'].items(), key=lambda x: x[1])[0]

    # Prepare the data
    df_processed = engineer_features(df)
    df_processed.dropna(inplace=True)
    df_sorted = df_processed.sort_values('Date')
    train_ratio = 0.8
    cutoff = int(len(df_sorted) * train_ratio)
    train_df = df_sorted.iloc[:cutoff]
    test_df = df_sorted.iloc[cutoff:]

    # Get selected features only
    X_train = train_df[selected_features].values
    y_train = train_df['Gold'].values
    X_test = test_df[selected_features].values
    y_test = test_df['Gold'].values

    # Scale the data
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    # Train final model with best parameters
    final_model = AdaBoostRegressor(
        n_estimators=best_n_estimators,
        learning_rate=best_learning_rate,
        loss=best_loss,
        random_state=42
    )
    final_model.fit(X_train_scaled, y_train_scaled)

    # Evaluate final model
    y_train_pred = final_model.predict(X_train_scaled)
    y_test_pred = final_model.predict(X_test_scaled)

    y_train_original = scaler_y.inverse_transform(y_train_scaled.reshape(-1, 1)).ravel()
    y_test_original = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).ravel()
    y_train_pred_original = scaler_y.inverse_transform(y_train_pred.reshape(-1, 1)).ravel()
    y_test_pred_original = scaler_y.inverse_transform(y_test_pred.reshape(-1, 1)).ravel()

    train_r2 = r2_score(y_train_original, y_train_pred_original)
    test_r2 = r2_score(y_test_original, y_test_pred_original)
    train_rmse = np.sqrt(mean_squared_error(y_train_original, y_train_pred_original))
    test_rmse = np.sqrt(mean_squared_error(y_test_original, y_test_pred_original))

    print("\nFinal AdaBoost Model Results:")
    print(f"Selected Features: {selected_features}")
    print(f"Best Parameters: n_estimators={best_n_estimators}, learning_rate={best_learning_rate}, loss={best_loss}")
    print(f"R-Squared Value: {test_r2:.5f}")
    print(f"Explained Variance Score: {explained_variance_score(y_test_original, y_test_pred_original):.5f}")
    print(f"Mean Absolute Error: {mean_absolute_error(y_test_original, y_test_pred_original):.2f}")
    print(f"Mean Squared Error: {mean_squared_error(y_test_original, y_test_pred_original):.2f}")
    print(f"Root Mean Squared Error: {test_rmse:.2f}")

    # Feature importance
    feature_importance = final_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': selected_features,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)

    print("\nFeature Importance:")
    for idx, row in feature_importance_df.iterrows():
        print(f"  - {row['Feature']}: {row['Importance']:.4f}")

    # Create visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_original, y_test_pred_original, alpha=0.5)
    plt.plot([y_test_original.min(), y_test_original.max()],
             [y_test_original.min(), y_test_original.max()],
             'r--', lw=2)
    plt.xlabel('Actual Gold Price')
    plt.ylabel('Predicted Gold Price')
    plt.title('Predicted vs Actual Gold Prices (AdaBoost)')
    plt.tight_layout()
    plt.show()

    return final_model, scaler_X, scaler_y, selected_features


# --------------------------------------------------------------------
# Main function
def main():
    print("Loading data...")
    df = load_data('../dataset/combined_dataset.csv')
    print(f"\nDataset shape: {df.shape}")

    print("\nGenerating correlation heatmap...")
    correlation = df.drop('Date', axis=1).corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='RdBu_r', vmin=-1, vmax=1, fmt='.2f')
    plt.title('Correlation Heatmap of Gold Price Features')
    plt.tight_layout()
    plt.show()

    print("\nRunning Time Series Cross-Validation for AdaBoost...")
    cv_results = time_series_cv_adaboost(df, n_splits=5, test_size=50)

    print("\nTraining final AdaBoost model with insights from cross-validation...")
    final_model, feature_scaler, target_scaler, selected_features = train_adaboost_final(df, cv_results)

    # Additional visualization: Actual vs Predicted over time
    df_sorted = df.sort_values('Date')
    train_ratio = 0.8
    cutoff = int(len(df_sorted) * train_ratio)
    test_df = df_sorted.iloc[cutoff:].copy()

    test_df = engineer_features(test_df)
    test_df.dropna(inplace=True)

    X_test = test_df[selected_features].values
    y_test = test_df['Gold'].values

    X_test_scaled = feature_scaler.transform(X_test)
    y_test_pred = final_model.predict(X_test_scaled)
    y_test_pred_original = target_scaler.inverse_transform(y_test_pred.reshape(-1, 1)).ravel()

    plt.figure(figsize=(12, 6))
    plt.plot(test_df['Date'], test_df['Gold'], label='Actual Gold Price', color='blue')
    plt.plot(test_df['Date'], y_test_pred_original, label='Predicted Gold Price', color='red', linestyle='--')
    plt.title('Actual vs Predicted Gold Prices Over Time (AdaBoost)')
    plt.xlabel('Date')
    plt.ylabel('Gold Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
