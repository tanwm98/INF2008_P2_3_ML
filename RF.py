import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from skopt import BayesSearchCV
from skopt.space import Integer, Real
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
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

def calculate_rsi(data, periods=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

#######################################
# Enhanced Feature Engineering
#######################################
def enhanced_feature_engineering(df):
    """Enhanced feature engineering addressing the identified issues"""
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

#######################################
# Bias-Corrected Prediction with Recalibration
#######################################
class GoldPricePredictor:
    def __init__(self, recalibration_window=20, use_log_transform=True):
        self.xgb_model = None
        self.feature_scaler = None
        self.target_scaler = None
        self.features = None
        self.recalibration_window = recalibration_window
        self.use_log_transform = use_log_transform
        self.bias_correction = 0
        self.bias_history = []

    def fit(self, X_train, y_train, features):
        """Fit the model using Bayesian Optimization"""
        self.features = features

        print("\nFeatures used for training:")
        for feature in self.features:
            print(f"- {feature}")

        # Apply log transformation if enabled
        if self.use_log_transform:
            y_train = np.log(y_train)

        # Scale features
        self.feature_scaler = RobustScaler()  # More robust to outliers
        X_train_scaled = self.feature_scaler.fit_transform(X_train)

        # Scale target
        self.target_scaler = StandardScaler()
        y_train_scaled = self.target_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

        # Define Bayesian search space for XGBoost
        search_space = {
            'n_estimators': Integer(50, 300),       # Number of trees
            'max_depth': Integer(3, 15),            # Depth of trees
            'learning_rate': Real(0.01, 0.3, 'log-uniform'),  # Learning rate (log scale for better optimization)
            'subsample': Real(0.6, 1.0),            # Row sampling
            'colsample_bytree': Real(0.6, 1.0),     # Feature sampling
            'reg_alpha': Real(0, 10.0),             # L1 Regularization
            'reg_lambda': Real(0, 10.0),            # L2 Regularization
        }

        # Initialize XGBoost model
        xgb_model = XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
        )

        # Bayesian Optimization using BayesSearchCV
        opt = BayesSearchCV(
            estimator=xgb_model,
            search_spaces=search_space,
            n_iter=20,   # Number of parameter combinations to try
            cv=3,        # Cross-validation folds
            n_jobs=-1,   # Use all CPUs
            random_state=42,
            verbose=2
        )

        # Fit the model with optimized hyperparameters
        opt.fit(X_train_scaled, y_train_scaled)

        # Save the best model
        self.xgb_model = opt.best_estimator_
        print(f"\nBest Bayesian Optimized Parameters: {opt.best_params_}")

        self.print_feature_importance()

        # Calculate initial bias correction
        y_train_pred = self.predict_without_correction(X_train)
        self.bias_correction = np.mean(y_train - y_train_pred)
        print(f"Initial bias correction: {self.bias_correction:.4f}")

        return self.xgb_model

    def predict_without_correction(self, X):
        """Make predictions without bias correction"""
        X_scaled = self.feature_scaler.transform(X)
        y_pred_scaled = self.xgb_model.predict(X_scaled)
        y_pred = self.target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

        # Inverse log transform if enabled
        if self.use_log_transform:
            y_pred = np.exp(y_pred)

        return y_pred
    
    def print_feature_importance(self, top_n=15):
        """Prints and plots the feature importance from the trained model"""
        if self.xgb_model is None:
            print("Model not trained yet!")
            return

        feature_importance = pd.DataFrame({
            'Feature': self.features,
            'Importance': self.xgb_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        print("\nFeature Importances:")
        for i, row in feature_importance.iterrows():
            print(f"{row['Feature']}: {row['Importance']:.5f}")

        # Plot feature importance
        plt.figure(figsize=(12, 6))
        sns.barplot(y=feature_importance['Feature'][:top_n], x=feature_importance['Importance'][:top_n], orient='h')
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.title("Top Feature Importances")
        plt.show()

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

        # Calculate metrics for initial predictions
        initial_r2 = r2_score(y_test, y_pred_initial)
        initial_explained_variance = explained_variance_score(y_test, y_pred_initial)
        initial_mae = mean_absolute_error(y_test, y_pred_initial)
        initial_mse = mean_squared_error(y_test, y_pred_initial)
        initial_rmse = np.sqrt(initial_mse)
        initial_mape = np.mean(np.abs((y_test - y_pred_initial) / y_test)) * 100

        print("\nInitial Prediction Metrics:")
        print(f"R²: {initial_r2:.5f}")
        print(f"Explained Variance Score: {initial_explained_variance:.5f}")
        print(f"MAE: {initial_mae:.2f}")
        print(f"MSE: {initial_mse:.2f}")
        print(f"RMSE: {initial_rmse:.2f}")
        print(f"MAPE: {initial_mape:.2f}%")

        # If recalibration is enabled, perform recalibration prediction
        if recalibrate:
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
                    self.predict(X_batch, recalibrate=True, actual_values=y_test[i:end_idx])

            # Calculate metrics for recalibrated predictions
            recalib_r2 = r2_score(y_test, y_recalib)
            recalib_explained_variance = explained_variance_score(y_test, y_recalib)
            recalib_mae = mean_absolute_error(y_test, y_recalib)
            recalib_mse = mean_squared_error(y_test, y_recalib)
            recalib_rmse = np.sqrt(recalib_mse)
            recalib_mape = np.mean(np.abs((y_test - y_recalib) / y_test)) * 100

            print("\nRecalibrated Prediction Metrics:")
            print(f"R²: {recalib_r2:.5f}")
            print(f"Explained Variance Score: {recalib_explained_variance:.5f}")
            print(f"MAE: {recalib_mae:.2f}")
            print(f"MSE: {recalib_mse:.2f}")
            print(f"RMSE: {recalib_rmse:.2f}")
            print(f"MAPE: {recalib_mape:.2f}%")

            return {
                'initial_metrics': {
                    'r2': initial_r2,
                    'explained_variance': initial_explained_variance,
                    'mae': initial_mae,
                    'mse': initial_mse,
                    'rmse': initial_rmse,
                    'mape': initial_mape
                },
                'recalibrated_metrics': {
                    'r2': recalib_r2,
                    'explained_variance': recalib_explained_variance,
                    'mae': recalib_mae,
                    'mse': recalib_mse,
                    'rmse': recalib_rmse,
                    'mape': recalib_mape
                },
                'predictions': {
                    'initial': y_pred_initial,
                    'recalibrated': y_recalib
                }
            }

        return {
            'initial_metrics': {
                'r2': initial_r2,
                'explained_variance': initial_explained_variance,
                'mae': initial_mae,
                'mse': initial_mse,
                'rmse': initial_rmse,
                'mape': initial_mape
            },
            'predictions': {
                'initial': y_pred_initial
            }
        }

    def plot_feature_importance(self, n=15):
        """Plot feature importances from the model"""
        if self.xgb_model is None or self.features is None:
            print("Model not trained yet!")
            return

        importances = self.xgb_model.feature_importances_
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
def plot_predictions_vs_actual(y_test, y_pred, model_name="XGBoost Model"):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    # Plot the diagonal line for reference
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', lw=2)
    plt.xlabel("Actual Gold Price")
    plt.ylabel("Predicted Gold Price")
    plt.title(f"{model_name}: Predicted vs Actual Gold Prices")
    plt.tight_layout()
    plt.show()

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
    leakage_features = ['Date', 'Gold', 'Silver', 'S&P500'] 
    features = [col for col in train_df.columns if col not in leakage_features and not col.startswith('Unnamed')]

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

    # Plot predictions vs actual
    plot_predictions_vs_actual(y_test, test_results['predictions']['recalibrated'], model_name="XGBoost Model")

    return predictor, test_results


if __name__ == "__main__":
    main()