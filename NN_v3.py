import warnings
import matplotlib
import random
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

# Set all random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

#######################################
# Neural Network Model Definition with BatchNorm and Dropout
#######################################
class GoldPriceNN(nn.Module):
    def __init__(self, input_size):
        super(GoldPriceNN, self).__init__()
        # Minimal architecture
        self.layer1 = nn.Linear(input_size, 4)  # Only 4 hidden units
        self.layer2 = nn.Linear(4, 1)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x
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
# Feature Engineering
#######################################
def process_features(df):
    """More conservative feature engineering approach"""
    result = df.copy()

    # Only create the most important features
    numerical_cols = ['Silver', 'DXY', 'Crude Oil', 'rates', 'cpi']
    for feature in numerical_cols:
        if feature in result.columns:
            # Just simple lags and percent changes
            for lag in [1, 3]:
                result[f'{feature}_Lag_{lag}'] = result[feature].shift(lag)
                result[f'{feature}_Pct_{lag}'] = result[feature].pct_change(periods=lag)
    result['Gold_Price_Trend'] = result['Gold'].diff(20).apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

    # Rate of change features for key variables
    for feature in ['DXY', 'rates', 'cpi']:
        if feature in result.columns:
            result[f'{feature}_ROC_5'] = result[feature].pct_change(5) * 100
    # Just a few technical features for Gold
    if 'Gold' in result.columns:
        # Moving averages (avoid complex derivatives)
        for window in [20, 50]:
            result[f'Gold_MA_{window}'] = result['Gold'].rolling(window=window).mean()

        # Simple volatility measure
        result['Gold_Vol_20d'] = result['Gold'].pct_change().rolling(window=20).std()

    # A couple of key ratio features
    if 'DXY' in result.columns and 'Gold' in result.columns:
        result['Gold_DXY_Ratio'] = result['Gold'] / result['DXY']

    if 'cpi' in result.columns and 'rates' in result.columns:
        result['real_rate'] = result['rates'] - result['cpi']

    # Remove potential leakage columns
    leakage_cols = ['Silver', 'S&P500']
    result = result.drop(leakage_cols, axis=1)

    return result

#######################################
# Data Preparation
#######################################
def prepare_data(df, train_ratio=0.7, val_ratio=0.15):
    """Split the data into train/validation/test sets and process features"""
    df_sorted = df.sort_values('Date')
    # (Optional) print warnings for very high correlations
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
    feature_columns = [col for col in train_df.columns if col not in ['Gold', 'Date'] and 'Gold' not in col]
    features = list(dict.fromkeys(feature_columns))  # Remove duplicates
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
    train_dataset = GoldDataset(X_train_scaled, y_train_scaled)
    val_dataset = GoldDataset(X_val_scaled, y_val_scaled)
    test_dataset = GoldDataset(X_test_scaled, y_test_scaled)
    return train_dataset, val_dataset, test_dataset, feature_scaler, target_scaler, features

def load_data(file_path):
    df = pd.read_csv(file_path)
    selected_columns = ['Date', 'Gold', 'Silver', 'Crude Oil', 'DXY', 'S&P500', 'cpi', 'rates']
    df = df[selected_columns].copy()
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df.dropna(inplace=True)
    return df

#######################################
# Feature Selection using SelectKBest
#######################################
def select_top_features(X_train, y_train, features, k=8, method='combined'):
    if method == 'mutual_info':
        selector = SelectKBest(mutual_info_regression, k=k)
    else:
        selector = SelectKBest(f_regression, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    feature_mask = selector.get_support()
    selected_features = [features[i] for i in range(len(features)) if feature_mask[i]]
    scores = selector.scores_
    feature_scores = [(features[i], scores[i]) for i in range(len(features))]
    feature_scores.sort(key=lambda x: x[1], reverse=True)
    print(f"\nTop {k} features selected:")
    for feature, score in feature_scores[:k]:
        print(f"- {feature}: {score:.4f}")
    return selected_features, X_train_selected, feature_mask

#######################################
# Time Series Cross-Validation with Feature Selection
#######################################
def time_series_cv(df, features, target='Gold', n_splits=5, test_size=50):
    df_sorted = df.sort_values('Date').reset_index(drop=True)
    df_processed = process_features(df_sorted)
    df_processed.dropna(inplace=True)
    cv_results = {'train_r2': [], 'test_r2': [], 'train_rmse': [], 'test_rmse': [], 'selected_features': []}
    n_samples = len(df_processed)
    indices = []
    for i in range(n_splits):
        test_end = n_samples - i * test_size
        test_start = test_end - test_size
        if test_start <= 0:
            continue
        indices.append((0, test_start, test_start, test_end))
    indices.reverse()
    for i, (train_start, train_end, test_start, test_end) in enumerate(indices):
        print(f"\nFold {i + 1}/{len(indices)}")
        print(f"Train: samples {train_start} to {train_end - 1}")
        print(f"Test: samples {test_start} to {test_end - 1}")
        train_df = df_processed.iloc[train_start:train_end]
        test_df = df_processed.iloc[test_start:test_end]
        X_train = train_df[features].values
        y_train = train_df[target].values
        X_test = test_df[features].values
        y_test = test_df[target].values
        k = min(8, len(features))
        selected_features, X_train_selected, feature_mask = select_top_features(X_train, y_train, features, k=k)
        X_test_selected = X_test[:, feature_mask]
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train_selected)
        X_test_scaled = scaler_X.transform(X_test_selected)
        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()
        train_dataset_cv = GoldDataset(X_train_scaled, y_train_scaled)
        test_dataset_cv = GoldDataset(X_test_scaled, y_test_scaled)
        input_size = X_train_selected.shape[1]
        model, _, _ = train_neural_network(train_dataset_cv, test_dataset_cv, input_size, epochs=100)
        model.eval()

        with torch.no_grad():
            y_train_pred = model(torch.FloatTensor(X_train_scaled)).numpy().flatten()
            y_test_pred = model(torch.FloatTensor(X_test_scaled)).numpy().flatten()
        train_r2 = r2_score(y_train_scaled, y_train_pred)
        test_r2 = r2_score(y_test_scaled, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train_scaled, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test_scaled, y_test_pred))
        cv_results['train_r2'].append(train_r2)
        cv_results['test_r2'].append(test_r2)
        cv_results['train_rmse'].append(train_rmse)
        cv_results['test_rmse'].append(test_rmse)
        cv_results['selected_features'].append(selected_features)
        print(f"Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
        print(f"Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")

    avg_train_r2 = np.mean(cv_results['train_r2'])
    avg_test_r2 = np.mean(cv_results['test_r2'])
    avg_train_rmse = np.mean(cv_results['train_rmse'])
    avg_test_rmse = np.mean(cv_results['test_rmse'])
    print("\nAverage Cross-Validation Results:")
    print(f"Train R²: {avg_train_r2:.4f}, Test R²: {avg_test_r2:.4f}")
    print(f"Train RMSE: {avg_train_rmse:.4f}, Test RMSE: {avg_test_rmse:.4f}")
    print("\nFeature Selection Frequency:")
    feature_counts = {}
    for f_list in cv_results['selected_features']:
        for feat in f_list:
            feature_counts[feat] = feature_counts.get(feat, 0) + 1
    for feat, count in sorted(feature_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"- {feat}: {count}/{len(indices)} folds")
    return cv_results

#######################################
# Neural Network Training with L1 Regularization and Adam Optimizer
#######################################

def train_neural_network(train_dataset, val_dataset, input_size, epochs=200):
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0,drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), num_workers=0)
    model = GoldPriceNN(input_size)
    # Use Adam optimizer with higher weight decay for stronger regularization
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    criterion = nn.MSELoss()
    l1_lambda = 0.1
    train_losses = []
    val_losses = []
    best_loss = float('inf')
    patience = 40
    patience_counter = 0
    best_model = None
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.reshape(-1, 1))
            l1_norm = sum(torch.sum(torch.abs(param)) for param in model.parameters())
            loss = loss + l1_lambda * l1_norm
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        model.eval()
        with torch.no_grad():
            X_val, y_val = next(iter(val_loader))
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val.reshape(-1, 1))
        train_losses.append(epoch_loss / len(train_loader))
        val_losses.append(val_loss.item())
        scheduler.step(val_loss)
        if epoch % 20 == 0:
            print(f'Epoch [{epoch}/{epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_model = model.state_dict().copy()
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    if best_model is not None:
        model.load_state_dict(best_model)
    return model, train_losses, val_losses


#######################################
# Main Function
#######################################
def main():
    print("Loading data...")
    df = load_data('dataset/combined_dataset.csv')
    print(f"\nDataset shape: {df.shape}")
    print("\nPreparing data for modeling with train/validation/test split...")
    train_dataset, val_dataset, test_dataset, feature_scaler, target_scaler, features = prepare_data(
        df, train_ratio=0.7, val_ratio=0.2)
    print("\nRunning time series cross-validation with feature selection...")
    cv_results = time_series_cv(df, features, n_splits=5, test_size=50)
    feature_counts = {}
    for feat_list in cv_results['selected_features']:
        for feat in feat_list:
            feature_counts[feat] = feature_counts.get(feat, 0) + 1
    top_features = [f for f, count in sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:8]]
    print("\nTop features across CV folds:")
    for feature in top_features:
        print(f"- {feature}")
    print("\nTraining final model with selected features...")
    X_train = train_dataset.X.numpy()
    y_train = train_dataset.y.numpy()
    X_test = test_dataset.X.numpy()
    y_test = test_dataset.y.numpy()
    selected_indices = [features.index(f) for f in top_features if f in features]
    X_train_selected = X_train[:, selected_indices]
    X_test_selected = X_test[:, selected_indices]
    train_dataset_selected = GoldDataset(X_train_selected, y_train)
    test_dataset_selected = GoldDataset(X_test_selected, y_test)
    input_size = len(top_features)
    model, train_losses, val_losses = train_neural_network(train_dataset_selected, test_dataset_selected, input_size, epochs = 50)
    model.eval()
    with torch.no_grad():
        y_train_pred = model(torch.FloatTensor(X_train_selected)).numpy().flatten()
        y_test_pred = model(torch.FloatTensor(X_test_selected)).numpy().flatten()
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    print("\nFinal Model Results (Selected Features):")
    print(f"R-Squared Value: {test_r2:.5f}")
    print(f"Mean Squared Error: {mean_squared_error(y_test, y_test_pred):.2f}")
    print(f"Root Mean Squared Error: {test_rmse:.2f}")
    print(f"Training R²: {train_r2:.5f}  |  Test R²: {test_r2:.5f}")
    print(f"Training RMSE: {train_rmse:.2f}  |  Test RMSE: {test_rmse:.2f}")
    y_test_original = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_test_pred_original = target_scaler.inverse_transform(y_test_pred.reshape(-1, 1)).flatten()
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_original, y_test_pred_original, alpha=0.5)
    plt.plot([y_test_original.min(), y_test_original.max()],
             [y_test_original.min(), y_test_original.max()],
             'r--', lw=2)
    plt.xlabel('Actual Gold Price')
    plt.ylabel('Predicted Gold Price')
    plt.title('Neural Network: Predicted vs Actual Gold Prices')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()