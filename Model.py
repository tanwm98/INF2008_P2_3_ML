import warnings
import matplotlib
matplotlib.use('TkAgg')  # Set the backend to Agg
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import SVR


def load_data(file_path):
    # Read the dataset
    df = pd.read_csv(file_path)

    # Select only the specified columns
    selected_columns = ['Date', 'Gold','Silver','S&P500','cpi','rates']
    df = df[selected_columns].copy()

    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Handle any missing values
    df = df.dropna()

    return df


def prepare_data(df):
    # Create features (X) and target (y)
    # Using today's values of Silver, Crude Oil, DXY, S&P500 to predict Gold
    X = df[['Silver','S&P500','cpi','rates']].values
    y = df['Gold'].values

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def create_correlation_heatmap(df):
    # Create correlation matrix
    correlation = df.drop('Date', axis=1).corr()

    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='RdBu_r', vmin=-1, vmax=1, fmt='.2f')
    plt.title('Correlation Heatmap of Gold Price Features')
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
        feature_names = ['Silver','S&P500','cpi','rates']
        coef = model.coef_
        if len(coef.shape) > 1:
            coef = coef.ravel()

        importances = pd.DataFrame({
            'Feature': feature_names,
            'Importance': np.abs(coef)
        })
        importances = importances.sort_values('Importance', ascending=False)
        print("\nFeature Importance:")
        print(importances)

        # Visualize feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importances)
        plt.title(f'Feature Importance - {model_name}')
        plt.tight_layout()
        plt.show()

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    # Dictionary to store models
    models = {
        'Linear Regression': LinearRegression(),
    }

    # Dictionary to store results
    results = {}

    # Train and evaluate each model
    for name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate and store results
        results[name] = {
            'R2': r2_score(y_test, y_pred),
            'EVS': explained_variance_score(y_test, y_pred),
            'MAE': mean_absolute_error(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
        }

        # Print detailed results
        evaluate_model(model, X_test, y_test, y_pred, name)

    return results


def main():
    # Load and prepare data
    print("Loading data...")
    df = load_data('dataset/new_gold_details.csv')

    print("\nDataset shape:", df.shape)
    print("\nFirst few rows of selected columns:")
    print(df.head())

    # Create correlation heatmap
    print("\nGenerating correlation heatmap...")
    create_correlation_heatmap(df)

    # Prepare data for modeling
    print("\nPreparing data for modeling...")
    X_train, X_test, y_train, y_test = prepare_data(df)

    # Train and evaluate models
    print("\nTraining and evaluating models...")
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()