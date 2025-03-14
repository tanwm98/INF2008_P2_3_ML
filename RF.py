import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import random

random.seed(42)
np.random.seed(42)
warnings.filterwarnings('ignore')
plt.style.use('ggplot')


#######################################
# Data Loading
#######################################
def load_data(file_path):
    df = pd.read_csv(file_path)
    selected_columns = ['Date', 'Gold', 'Silver', 'Crude Oil', 'DXY', 'S&P500', 'cpi', 'rates']
    df = df[selected_columns].copy()
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df.dropna(inplace=True)
    return df


#######################################
# Enhanced Feature Engineering
#######################################
def enhanced_feature_engineering(df):
    """Enhanced feature engineering addressing the identified issues"""
    result = df.copy()

    # 1. Log transform price levels for better statistical properties
    for col in ['Gold', 'Silver', 'Crude Oil', 'DXY', 'S&P500']:
        if col in result.columns:
            result[f'{col}_Log'] = np.log(result[col])

    # 2. Add trend features to capture directionality
    result['Time_Index'] = np.arange(len(result))
    result['Time_Index_Scaled'] = result['Time_Index'] / len(result)

    # 3. External predictors
    external_cols = ['Silver', 'DXY', 'Crude Oil', 'rates', 'cpi', 'S&P500']

    for col in external_cols:
        if col in result.columns:
            # Return-based features (better statistical properties than changes)
            for period in [1, 5, 10, 20]:
                result[f'{col}_Return_{period}d'] = result[col].pct_change(periods=period)

            # Volatility indicators
            for window in [10, 20, 30]:
                result[f'{col}_Volatility_{window}d'] = result[col].pct_change().rolling(window=window).std()

            # Momentum indicators
            for period in [10, 20, 50]:
                result[f'{col}_Momentum_{period}d'] = (
                                                              result[col] - result[col].shift(period)
                                                      ) / result[col].shift(period)

    # 4. Important economic indicators
    if 'rates' in result.columns and 'cpi' in result.columns:
        result['Real_Rate'] = result['rates'] - result['cpi']
        result['Real_Rate_Change'] = result['Real_Rate'].diff()
        # Add non-linear transformations for real rates
        result['Real_Rate_Squared'] = result['Real_Rate'] ** 2

    # 5. Market regime indicators
    if 'S&P500' in result.columns:
        # Bull/bear market indicator
        result['Market_Trend'] = result['S&P500'].pct_change(20).apply(
            lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
        )

    # 6. Cross-asset relationships (avoiding using Gold in calculations)
    if 'Silver' in result.columns and 'DXY' in result.columns:
        result['Silver_DXY_Ratio'] = result['Silver'] / result['DXY']
        result['Silver_DXY_Ratio_Change'] = result['Silver_DXY_Ratio'].pct_change()

    if 'Silver' in result.columns and 'Crude Oil' in result.columns:
        result['Silver_Oil_Ratio'] = result['Silver'] / result['Crude Oil']
        result['Silver_Oil_Ratio_Change'] = result['Silver_Oil_Ratio'].pct_change()

    # 7. Seasonality (avoid using explicit year to prevent capturing a simple trend)
    result['Month'] = result['Date'].dt.month
    result['Quarter'] = result['Date'].dt.quarter
    # Convert to cyclical features
    result['Month_Sin'] = np.sin(2 * np.pi * result['Month'] / 12)
    result['Month_Cos'] = np.cos(2 * np.pi * result['Month'] / 12)

    # 8. Calendar effects for financial markets
    result['Day_Of_Week'] = result['Date'].dt.dayofweek

    # Remove columns with NaN values
    result.dropna(inplace=True)

    return result


#######################################
# Bias-Corrected Prediction with Recalibration
#######################################
class GoldPricePredictor:
    def __init__(self, recalibration_window=20, use_log_transform=True):
        self.rf_model = None
        self.feature_scaler = None
        self.target_scaler = None
        self.features = None
        self.recalibration_window = recalibration_window
        self.use_log_transform = use_log_transform
        self.bias_correction = 0
        self.bias_history = []

    def fit(self, X_train, y_train, features):
        """Fit the model on training data"""
        self.features = features

        # Apply log transformation if enabled
        if self.use_log_transform:
            y_train = np.log(y_train)

        # Scale features
        self.feature_scaler = RobustScaler()  # More robust to outliers
        X_train_scaled = self.feature_scaler.fit_transform(X_train)

        # Scale target
        self.target_scaler = StandardScaler()
        y_train_scaled = self.target_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

        # Define model with optimal parameters
        self.rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features=0.7,
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )

        # Fit model
        self.rf_model.fit(X_train_scaled, y_train_scaled)

        # Calculate initial bias on training data
        y_train_pred = self.predict_without_correction(X_train)
        self.bias_correction = np.mean(y_train - y_train_pred)
        print(f"Initial bias correction: {self.bias_correction:.4f}")

        return self

    def predict_without_correction(self, X):
        """Make predictions without bias correction"""
        X_scaled = self.feature_scaler.transform(X)
        y_pred_scaled = self.rf_model.predict(X_scaled)
        y_pred = self.target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

        # Inverse log transform if enabled
        if self.use_log_transform:
            y_pred = np.exp(y_pred)

        return y_pred

    def predict(self, X, recalibrate=False, actual_values=None):
        """Make predictions with bias correction"""
        # Base prediction
        y_pred = self.predict_without_correction(X)

        # Apply bias correction
        y_pred_corrected = y_pred + self.bias_correction

        # Recalibrate if requested and actual values are provided
        if recalibrate and actual_values is not None:
            # Update bias correction based on recent observations
            recent_errors = actual_values - y_pred[:len(actual_values)]
            recent_bias = np.mean(recent_errors[-self.recalibration_window:])
            self.bias_history.append(recent_bias)

            # Exponential smoothing for bias correction
            alpha = 0.3  # Smoothing factor
            self.bias_correction = alpha * recent_bias + (1 - alpha) * self.bias_correction

            # Apply updated correction
            y_pred_corrected = y_pred + self.bias_correction
            print(f"Recalibrated bias correction: {self.bias_correction:.4f}")

        return y_pred_corrected

    def evaluate(self, X_test, y_test, recalibrate=True):
        """Evaluate model performance with detailed metrics"""
        # Initial prediction without recalibration
        y_pred_initial = self.predict(X_test)

        # Calculate metrics
        initial_r2 = r2_score(y_test, y_pred_initial)
        initial_rmse = np.sqrt(mean_squared_error(y_test, y_pred_initial))
        initial_mae = mean_absolute_error(y_test, y_pred_initial)

        print("\nInitial Prediction Metrics:")
        print(f"R²: {initial_r2:.5f}")
        print(f"RMSE: {initial_rmse:.2f}")
        print(f"MAE: {initial_mae:.2f}")

        # If recalibration is enabled, perform recalibration prediction
        if recalibrate:
            # Split test data for stepwise recalibration
            y_recalib = []

            for i in range(0, len(X_test), self.recalibration_window):
                # Get the current batch
                end_idx = min(i + self.recalibration_window, len(X_test))
                X_batch = X_test[i:end_idx]

                # Predict with current calibration
                y_batch_pred = self.predict(X_batch)
                y_recalib.extend(y_batch_pred)

                # Recalibrate using actual values if we have more batches to process
                if end_idx < len(X_test):
                    self.predict(X_batch, recalibrate=True,
                                 actual_values=y_test[i:end_idx])

            # Calculate metrics after recalibration
            recalib_r2 = r2_score(y_test, y_recalib)
            recalib_rmse = np.sqrt(mean_squared_error(y_test, y_recalib))
            recalib_mae = mean_absolute_error(y_test, y_recalib)

            print("\nRecalibrated Prediction Metrics:")
            print(f"R²: {recalib_r2:.5f}")
            print(f"RMSE: {recalib_rmse:.2f}")
            print(f"MAE: {recalib_mae:.2f}")

            # Plot results
            plt.figure(figsize=(14, 7))
            plt.title('Gold Price: Actual vs Predicted (with Recalibration)')
            plt.plot(y_test, 'b-', label='Actual Price')
            plt.plot(y_pred_initial, 'g-', label='Initial Prediction')
            plt.plot(y_recalib, 'r-', label='Recalibrated Prediction')
            plt.xlabel('Time')
            plt.ylabel('Gold Price')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

            return {
                'initial_metrics': {
                    'r2': initial_r2,
                    'rmse': initial_rmse,
                    'mae': initial_mae
                },
                'recalibrated_metrics': {
                    'r2': recalib_r2,
                    'rmse': recalib_rmse,
                    'mae': recalib_mae
                },
                'predictions': {
                    'initial': y_pred_initial,
                    'recalibrated': y_recalib
                }
            }

        return {
            'initial_metrics': {
                'r2': initial_r2,
                'rmse': initial_rmse,
                'mae': initial_mae
            },
            'predictions': {
                'initial': y_pred_initial
            }
        }

    def plot_feature_importance(self, n=15):
        """Plot feature importances from the model"""
        if self.rf_model is None or self.features is None:
            print("Model not trained yet!")
            return

        importances = self.rf_model.feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': self.features,
            'Importance': importances
        }).sort_values('Importance', ascending=False)

        print(f"\nTop {n} Most Important Features:")
        print(feature_importance.head(n))

        plt.figure(figsize=(12, 8))
        plt.barh(feature_importance['Feature'].head(n), feature_importance['Importance'].head(n))
        plt.title(f'Top {n} Features for Predicting Gold Price')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()

        return feature_importance


#######################################
# Main Function
#######################################
def main():
    print("Loading data...")
    df = load_data('dataset/combined_dataset.csv')
    print(f"Dataset shape: {df.shape}")

    # Process features with enhanced engineering
    print("\nApplying enhanced feature engineering...")
    processed_df = enhanced_feature_engineering(df)
    processed_df.dropna(inplace=True)

    # Split data
    n = len(processed_df)
    train_ratio = 0.7
    val_ratio = 0.15
    train_cutoff = int(n * train_ratio)
    val_cutoff = int(n * (train_ratio + val_ratio))

    train_df = processed_df.iloc[:train_cutoff].copy()
    val_df = processed_df.iloc[train_cutoff:val_cutoff].copy()
    test_df = processed_df.iloc[val_cutoff:].copy()

    # Remove features that directly contain Gold or are time-based leakage
    leakage_features = ['Gold', 'Gold_Log', 'Date', 'Time_Index']
    features = [col for col in train_df.columns if col not in leakage_features]

    print(f"\nUsing {len(features)} features for prediction")

    # Prepare data
    X_train = train_df[features].values
    y_train = train_df['Gold'].values
    X_val = val_df[features].values
    y_val = val_df['Gold'].values
    X_test = test_df[features].values
    y_test = test_df['Gold'].values

    # Train and evaluate the model
    print("\nTraining gold price predictor with bias correction...")
    predictor = GoldPricePredictor(recalibration_window=10, use_log_transform=True)
    predictor.fit(X_train, y_train, features)

    # Validate on validation set
    print("\nEvaluating on validation set...")
    val_results = predictor.evaluate(X_val, y_val, recalibrate=True)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = predictor.evaluate(X_test, y_test, recalibrate=True)

    # Plot feature importance
    predictor.plot_feature_importance(n=15)

    return predictor, test_results


if __name__ == "__main__":
    main()