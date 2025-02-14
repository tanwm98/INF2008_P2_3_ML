import warnings
import matplotlib
matplotlib.use('TkAgg')  # Set the backend to TkAgg
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from sklearn.linear_model import LinearRegression


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

    # Optional: If you want to predict today's Gold using *yesterday's* features, you can shift:
    # df['Silver'] = df['Silver'].shift(1)
    # df['Crude Oil'] = df['Crude Oil'].shift(1)
    # df['DXY'] = df['DXY'].shift(1)
    # df['S&P500'] = df['S&P500'].shift(1)
    # df['cpi'] = df['cpi'].shift(1)
    # df['rates'] = df['rates'].shift(1)
    # df.dropna(inplace=True)

    return df


def time_based_split(df, train_ratio=0.7):
    """
    Splits the DataFrame into train/test sets based on chronological order.
    train_ratio: fraction of data used for training.
    """
    df_sorted = df.sort_values('Date')
    cutoff = int(len(df_sorted) * train_ratio)
    train_df = df_sorted.iloc[:cutoff]
    test_df = df_sorted.iloc[cutoff:]
    return train_df, test_df


def prepare_data(df, train_ratio=0.7):
    """
    Performs a time-based split, then separates features (X) and target (y).
    """
    # Split into train/test by date
    train_df, test_df = time_based_split(df, train_ratio=train_ratio)

    # Features and target
    features = ['Silver', 'Crude Oil', 'DXY', 'S&P500', 'cpi', 'rates']
    target = 'Gold'

    X_train = train_df[features].values
    y_train = train_df[target].values

    X_test = test_df[features].values
    y_test = test_df[target].values

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def create_correlation_heatmap(df):
    # Create correlation matrix using numeric columns (drop Date)
    correlation = df.drop('Date', axis=1).corr()

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
        feature_names = ['Silver', 'Crude Oil', 'DXY', 'S&P500', 'cpi', 'rates']
        coef = model.coef_
        if len(coef.shape) > 1:
            coef = coef.ravel()

        importances = pd.DataFrame({
            'Feature': feature_names,
            'Importance': np.abs(coef)
        })
        importances.sort_values('Importance', ascending=False, inplace=True)

        print("\nFeature Importance:")
        print(importances)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importances)
        plt.title(f'Feature Importance - {model_name}')
        plt.tight_layout()
        plt.show()


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        'Linear Regression': LinearRegression(),
        # You could add more models here, e.g.:
        # 'Lasso': Lasso(alpha=0.1),
        # 'Ridge': Ridge(alpha=1.0),
        # 'ElasticNet': ElasticNet(alpha=0.1),
        # 'SVR': SVR(kernel='rbf')
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
    df = load_data('dataset/combined_data_v2.csv')
    print(f"\nDataset shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())

    # Correlation heatmap
    print("\nGenerating correlation heatmap...")
    create_correlation_heatmap(df)

    # Prepare data for modeling
    print("\nPreparing data for modeling with a time-based split (70% train / 30% test)...")
    X_train, X_test, y_train, y_test = prepare_data(df, train_ratio=0.7)

    # Train and evaluate models
    print("\nTraining and evaluating models...")
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
