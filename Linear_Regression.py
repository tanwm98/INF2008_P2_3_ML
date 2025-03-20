import warnings
import matplotlib

matplotlib.use('TkAgg')
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression


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


def process_features(df):
    """Create lag-based features to avoid leakage, including for Gold."""
    result = df.copy()

    # For all numerical columns, create percentage change features
    numerical_cols = ['Silver', 'S&P500', 'Crude Oil', 'DXY', 'rates', 'cpi']
    for feature in numerical_cols:
        if feature in result.columns:
            # Create percentage changes at different lags
            for lag in [1, 3, 6, 12]:
                result[f'{feature}_Pct_{lag}'] = result[feature].pct_change(periods=lag)
                if lag < 12:  # Only for smaller lags
                    result[f'{feature}_Accel_{lag}'] = result[feature].pct_change().diff(lag)
            # Rolling volatility
            for window in [5, 10, 20]:
                result[f'{feature}_Vol_{window}'] = result[feature].pct_change().rolling(window=window).std()
            # Moving average ratio
            for window in [10, 20]:
                ma = result[feature].rolling(window=window).mean()
                result[f'{feature}_MA_Ratio_{window}'] = result[feature] / ma - 1

    # Add lag-based features for Gold itself to capture time series patterns
    if 'Gold' in result.columns:
        for lag in [1, 3, 6, 12]:
            result[f'Gold_Lag_{lag}'] = result['Gold'].shift(lag)
        # Add rolling statistics for Gold
        for window in [3, 6, 12]:
            result[f'Gold_Rolling_Mean_{window}'] = result['Gold'].rolling(window=window).mean()
            result[f'Gold_Rolling_Std_{window}'] = result['Gold'].rolling(window=window).std()

    # Create cross-asset ratios
    if 'Silver' in result.columns and 'Gold' in result.columns:
        result['Gold_Silver_Ratio'] = result['Gold'] / result['Silver']
        result['Gold_Silver_Ratio_Pct'] = result['Gold_Silver_Ratio'].pct_change()
    if 'Crude Oil' in result.columns and 'Gold' in result.columns:
        result['Gold_Oil_Ratio'] = result['Gold'] / result['Crude Oil']
        result['Gold_Oil_Ratio_Pct'] = result['Gold_Oil_Ratio'].pct_change()
    if 'DXY' in result.columns and 'Gold' in result.columns:
        result['Gold_DXY_Ratio'] = result['Gold'] / result['DXY']
        result['Gold_DXY_Ratio_Pct'] = result['Gold_DXY_Ratio'].pct_change()

    # Drop original features that might cause leakage
    leakage_cols = ['Silver', 'S&P500']
    result = result.drop(leakage_cols, axis=1)

    return result


def prepare_data(df, train_ratio=0.7, val_ratio=0.15):
    """Split the data into train/validation/test sets and process features"""
    df_sorted = df.sort_values('Date')

    # Print warnings for high correlations
    corr_with_gold = df_sorted.corr()['Gold'].abs().sort_values(ascending=False)
    high_corr_features = [f for f in corr_with_gold[corr_with_gold > 0.9].index.tolist() if f not in ['Gold', 'Date']]
    if high_corr_features:
        print("WARNING: The following features have very high correlation with Gold (potential leakage):")
        for feature in high_corr_features:
            print(f"- {feature}")

    # Split data by time
    n = len(df_sorted)
    train_cutoff = int(n * train_ratio)
    val_cutoff = int(n * (train_ratio + val_ratio))

    train_df = df_sorted.iloc[:train_cutoff].copy()
    val_df = df_sorted.iloc[train_cutoff:val_cutoff].copy()
    test_df = df_sorted.iloc[val_cutoff:].copy()

    # Process features for each split
    train_df = process_features(train_df)
    val_df = process_features(val_df)
    test_df = process_features(test_df)

    # Drop NaN values
    train_df.dropna(inplace=True)
    val_df.dropna(inplace=True)
    test_df.dropna(inplace=True)

    # Build feature list (exclude Gold and Date and gold-derived features)
    feature_columns = [col for col in train_df.columns if col not in ['Gold', 'Date'] and 'Gold' not in col]
    features = list(dict.fromkeys(feature_columns))  # Remove duplicates

    print("Final features used in model training:")
    for f in features:
        print(f"- {f}")

    # Prepare data arrays
    X_train = train_df[features].values
    y_train = train_df['Gold'].values
    X_val = val_df[features].values
    y_val = val_df['Gold'].values
    X_test = test_df[features].values
    y_test = test_df['Gold'].values

    # Scale features and target
    feature_scaler = StandardScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_val_scaled = feature_scaler.transform(X_val)
    X_test_scaled = feature_scaler.transform(X_test)

    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1)).ravel()
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).ravel()

    return (X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled,
            X_test_scaled, y_test_scaled, feature_scaler, target_scaler, features)


def select_top_features(X_train, y_train, features, k=10):
    """Select the most important features using F-regression and mutual information"""
    # F-regression
    f_selector = SelectKBest(f_regression, k=k)
    f_selector.fit(X_train, y_train)
    f_scores = f_selector.scores_

    # Mutual information
    mi_selector = SelectKBest(mutual_info_regression, k=k)
    mi_selector.fit(X_train, y_train)
    mi_scores = mi_selector.scores_

    # Combine scores (normalize first)
    f_scores_norm = f_scores / np.max(f_scores)
    mi_scores_norm = mi_scores / np.max(mi_scores)
    combined_scores = (f_scores_norm + mi_scores_norm) / 2

    # Create a dataframe for easy sorting
    importance_df = pd.DataFrame({
        'Feature': features,
        'F_Score': f_scores,
        'MI_Score': mi_scores,
        'Combined_Score': combined_scores
    })

    # Sort by combined score
    importance_df = importance_df.sort_values('Combined_Score', ascending=False)

    # Select top k features
    top_features = importance_df.head(k)['Feature'].tolist()

    print(f"\nTop {k} features selected:")
    for i, (feature, score) in enumerate(zip(top_features, importance_df.head(k)['Combined_Score'])):
        print(f"{i + 1}. {feature}: {score:.4f}")

    # Get indices of top features
    top_indices = [features.index(feature) for feature in top_features]

    # Filter X_train to only include top features
    X_train_selected = X_train[:, top_indices]

    return top_features, top_indices, X_train_selected


def time_series_cv(df, features, target='Gold', n_splits=5, test_size=50):
    """Perform time series cross-validation"""
    df_sorted = df.sort_values('Date').reset_index(drop=True)
    df_processed = process_features(df_sorted)
    df_processed.dropna(inplace=True)

    cv_results = {
        'train_r2': [], 'test_r2': [],
        'train_rmse': [], 'test_rmse': [],
        'selected_features': []
    }

    # Create time-based folds
    n_samples = len(df_processed)
    indices = []

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

        # Split data
        train_df = df_processed.iloc[train_start:train_end]
        test_df = df_processed.iloc[test_start:test_end]

        # Prepare features and target
        X_train = train_df[features].values
        y_train = train_df[target].values
        X_test = test_df[features].values
        y_test = test_df[target].values

        # Select top features
        k = min(12, len(features))
        top_features, top_indices, X_train_selected = select_top_features(X_train, y_train, features, k=k)
        X_test_selected = X_test[:, top_indices]

        # Scale data
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train_selected)
        X_test_scaled = scaler_X.transform(X_test_selected)

        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

        # Train model
        model = LinearRegression()
        model.fit(X_train_scaled, y_train_scaled)

        # Make predictions
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)

        # Calculate metrics
        train_r2 = r2_score(y_train_scaled, y_train_pred)
        test_r2 = r2_score(y_test_scaled, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train_scaled, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test_scaled, y_test_pred))

        # Store results
        cv_results['train_r2'].append(train_r2)
        cv_results['test_r2'].append(test_r2)
        cv_results['train_rmse'].append(train_rmse)
        cv_results['test_rmse'].append(test_rmse)
        cv_results['selected_features'].append(top_features)

        print(f"Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
        print(f"Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")

    # Print average results
    avg_train_r2 = np.mean(cv_results['train_r2'])
    avg_test_r2 = np.mean(cv_results['test_r2'])
    avg_train_rmse = np.mean(cv_results['train_rmse'])
    avg_test_rmse = np.mean(cv_results['test_rmse'])

    print("\nAverage Cross-Validation Results:")
    print(f"Train R²: {avg_train_r2:.4f}, Test R²: {avg_test_r2:.4f}")
    print(f"Train RMSE: {avg_train_rmse:.4f}, Test RMSE: {avg_test_rmse:.4f}")

    # Count feature selection frequency
    print("\nFeature Selection Frequency:")
    feature_counts = {}
    for f_list in cv_results['selected_features']:
        for feat in f_list:
            feature_counts[feat] = feature_counts.get(feat, 0) + 1

    for feat, count in sorted(feature_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"- {feat}: {count}/{len(indices)} folds")

    return cv_results


def create_correlation_heatmap(df):
    """Create correlation heatmap of features"""
    # Create correlation matrix using numeric columns (drop Date)
    correlation = df.drop('Date', axis=1).corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='RdBu_r', vmin=-1, vmax=1, fmt='.2f')
    plt.title('Correlation Heatmap of Gold Price Features')
    plt.tight_layout()
    plt.show()


def plot_predictions_vs_actual(y_test, y_pred, model_name):
    """Create a scatter plot comparing actual vs predicted values"""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)

    # Add diagonal line for reference
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             'r--', lw=2)

    plt.xlabel('Actual Gold Price')
    plt.ylabel('Predicted Gold Price')
    plt.title(f'{model_name}: Predicted vs Actual Gold Prices')
    plt.tight_layout()
    plt.show()


def evaluate_model(model, X_test, y_test, features):
    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    print(f"\nLinear Regression Results:")
    print(f"R-Squared Value: {r2:.5f}")
    print(f"Explained Variance Score: {evs:.5f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")

    # (Optional) Display feature importance if available
    if hasattr(model, 'coef_'):
        coef = model.coef_
        importances = pd.DataFrame({
            'Feature': features,
            'Coefficient': coef,
            'Absolute Importance': np.abs(coef)
        }).sort_values('Absolute Importance', ascending=False)
        print("\nFeature Importance:")
        print(importances)
        plt.figure(figsize=(10, 6))
        plt.barh(importances['Feature'][:10], importances['Absolute Importance'][:10])
        plt.xlabel('Absolute Coefficient Value')
        plt.title('Top 10 Features by Importance')
        plt.tight_layout()
        plt.show()

    # (Optional) Visualize predictions vs actual values
    plot_predictions_vs_actual(y_test, y_pred, "Linear Regression")

    return {
        'R2': r2,
        'Explained_Variance': evs,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'predictions': y_pred
    }


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
    print("\nPreparing data for modeling with train/validation/test split...")
    X_train, y_train, X_val, y_val, X_test, y_test, feature_scaler, target_scaler, features = prepare_data(
        df, train_ratio=0.7, val_ratio=0.15)

    # Perform time series cross-validation
    print("\nRunning time series cross-validation with feature selection...")
    cv_results = time_series_cv(df, features, n_splits=5, test_size=50)

    # Identify most important features across CV folds
    feature_counts = {}
    for feat_list in cv_results['selected_features']:
        for feat in feat_list:
            feature_counts[feat] = feature_counts.get(feat, 0) + 1

    top_features = [f for f, count in sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:10]]
    print("\nTop features across CV folds:")
    for feature in top_features:
        print(f"- {feature}")

    # Filter to top features for final model
    top_indices = [features.index(feature) for feature in top_features if feature in features]
    X_train_selected = X_train[:, top_indices]
    X_test_selected = X_test[:, top_indices]

    # Train final model
    print("\nTraining final model with selected features...")
    final_model = LinearRegression()
    final_model.fit(X_train_selected, y_train)

    # Evaluate final model
    print("\nEvaluating final model...")
    results = evaluate_model(final_model, X_test_selected, y_test, top_features)

    # Convert predictions back to original scale for interpretation
    y_test_original = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_original = target_scaler.inverse_transform(results['predictions'].reshape(-1, 1)).flatten()
    # Transform predictions and true values back to original scale

    # Compute metrics on original scale
    r2_orig = r2_score(y_test_original, y_pred_original)
    mae_orig = mean_absolute_error(y_test_original, y_pred_original)
    mse_orig = mean_squared_error(y_test_original, y_pred_original)
    rmse_orig = np.sqrt(mse_orig)
    mape_orig = np.mean(np.abs((y_test_original - y_pred_original) / y_test_original)) * 100

    print("Metrics on Original Scale:")
    print(f"R²: {r2_orig:.5f}")
    print(f"MAE: {mae_orig:.2f}")
    print(f"MSE: {mse_orig:.2f}")
    print(f"RMSE: {rmse_orig:.2f}")
    print(f"EVS: {explained_variance_score(y_test_original, y_pred_original):.5f}")
    # Plot original scale predictions
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_original, label='Actual')
    plt.plot(y_pred_original, label='Predicted')
    plt.title('Gold Price: Actual vs Predicted')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()