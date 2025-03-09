import warnings
import matplotlib

matplotlib.use('TkAgg')
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn


# Neural Network Model Definition
class GoldPriceNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Dropout(0.7)
        )
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        x = self.fc(x)
        return self.output(x)

class GoldDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class WeightedHuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(WeightedHuberLoss, self).__init__()
        self.huber = nn.HuberLoss(delta=delta)

    def forward(self, pred, target):
        # Add more weight to high-value predictions
        weights = torch.sqrt(torch.abs(target))
        return self.huber(pred * weights, target * weights)


def augment_data(X, y, noise_level=0.03):
    # Add small random noise to features
    noise = torch.randn_like(X) * noise_level
    X_aug = X + noise
    return torch.cat([X, X_aug]), torch.cat([y, y])


def load_data(file_path):
    df = pd.read_csv(file_path)
    selected_columns = ['Date', 'Gold', 'Silver', 'Crude Oil', 'DXY', 'S&P500', 'cpi', 'rates']
    df = df[selected_columns].copy()
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df.dropna(inplace=True)
    return df


def calculate_rsi(data, periods=14):
    # Calculate price differences
    delta = data.diff()

    # Make two series: one for lower closes and one for higher closes
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()

    # Calculate RSI
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def prepare_data(df, train_ratio=0.7):
    """Prepare data without using gold-derived features"""
    # First split data chronologically
    df_sorted = df.sort_values('Date')
    cutoff = int(len(df_sorted) * train_ratio)
    train_df = df_sorted.iloc[:cutoff].copy()
    test_df = df_sorted.iloc[cutoff:].copy()

    # Base exogenous features (not derived from Gold)
    base_features = ['Silver', 'S&P500', 'cpi', 'Crude Oil', 'DXY', 'rates']
    base_features = [f for f in base_features if f in df.columns]

    # Feature engineering separately on train and test
    for dataset in [train_df, test_df]:
        # Generate features from exogenous variables only
        dataset['Silver_Return'] = dataset['Silver'].pct_change()
        dataset['Silver_MA5'] = dataset['Silver'].rolling(window=5).mean()
        dataset['Silver_EMA10'] = dataset['Silver'].ewm(span=10, adjust=False).mean()

        dataset['SP500_Return'] = dataset['S&P500'].pct_change()
        dataset['SP500_MA10'] = dataset['S&P500'].rolling(window=10).mean()

        if 'Crude Oil' in dataset.columns:
            dataset['Oil_Return'] = dataset['Crude Oil'].pct_change()

        # Create ratio features (avoiding any that involve Gold)
        if 'Silver' in dataset.columns and 'Crude Oil' in dataset.columns:
            dataset['Silver_Oil_Ratio'] = dataset['Silver'] / dataset['Crude Oil']

    # Drop missing values from each set separately
    train_df.dropna(inplace=True)
    test_df.dropna(inplace=True)

    # Create feature list without any Gold-derived features
    feature_columns = base_features + [
        col for col in train_df.columns
        if col not in ['Gold', 'Date'] and 'Gold' not in col
    ]

    # Ensure all features exist in both datasets
    features = [f for f in feature_columns if f in train_df.columns and f in test_df.columns]

    # Prepare feature matrices and target vectors
    X_train = train_df[features].values
    y_train = train_df['Gold'].values
    X_test = test_df[features].values
    y_test = test_df['Gold'].values

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

    return train_dataset, test_dataset, feature_scaler, target_scaler



def create_correlation_heatmap(df):
    # Keep your existing create_correlation_heatmap function unchanged
    correlation = df.drop('Date', axis=1).corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='RdBu_r', vmin=-1, vmax=1, fmt='.2f')
    plt.title('Correlation Heatmap of Gold Price Features')
    plt.tight_layout()
    plt.show()


def train_neural_network(train_dataset, test_dataset, input_size):
    torch.manual_seed(42)
    np.random.seed(42)

    # Hyperparameters
    batch_size = 23
    initial_lr = 0.001
    epochs = 500
    patience = 10

    # Augmented data
    X_train, y_train = train_dataset.X, train_dataset.y
    X_train_aug, y_train_aug = augment_data(X_train, y_train)

    from torch.utils.data import TensorDataset
    augmented_train_dataset = TensorDataset(X_train_aug, y_train_aug)

    train_loader = DataLoader(augmented_train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

    model = GoldPriceNN(input_size)

    # AdamW optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-2)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15, verbose=True, min_lr=1e-6
    )

    # Custom loss function - combination of MSE and Huber
    mse_criterion = nn.MSELoss()
    huber_criterion = nn.HuberLoss(delta=1.0)

    train_losses = []
    test_losses = []
    best_model = None
    best_loss = float('inf')
    no_improve_count = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)

            # Combined loss
            mse_loss = mse_criterion(outputs, batch_y)
            huber_loss = huber_criterion(outputs, batch_y)
            loss = 0.6 * mse_loss + 0.4 * huber_loss

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            epoch_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            X_test, y_test = next(iter(test_loader))
            test_outputs = model(X_test)
            test_loss = mse_criterion(test_outputs, y_test)

        # Update learning rate based on validation performance
        scheduler.step(test_loss)

        train_losses.append(epoch_loss / len(train_loader))
        test_losses.append(test_loss.item())

        # Save best model
        if test_loss < best_loss:
            best_loss = test_loss
            best_model = model.state_dict().copy()
            no_improve_count = 0
        else:
            no_improve_count += 1

        # Early stopping
        if no_improve_count >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        if epoch % 20 == 0:
            print(f'Epoch [{epoch}/{epochs}], Train Loss: {epoch_loss / len(train_loader):.4f}, '
                  f'Test Loss: {test_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')

    # Load best model
    model.load_state_dict(best_model)
    return model, train_losses, test_losses


def evaluate_model(model, test_dataset, train_losses, test_losses, target_scaler):
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
    X_test, y_test = next(iter(test_loader))

    with torch.no_grad():
        y_pred = model(X_test)

    # Convert to numpy and inverse transform predictions
    y_test_np = y_test.numpy()
    y_pred_np = y_pred.numpy()

    # Inverse transform the scaled values
    y_test_original = target_scaler.inverse_transform(y_test_np)
    y_pred_original = target_scaler.inverse_transform(y_pred_np)

    # Calculate metrics on original scale
    r2 = r2_score(y_test_original, y_pred_original)
    evs = explained_variance_score(y_test_original, y_pred_original)
    mae = mean_absolute_error(y_test_original, y_pred_original)
    mse = mean_squared_error(y_test_original, y_pred_original)
    rmse = np.sqrt(mse)

    print("\nNeural Network Results:")
    print(f"R-Squared Value: {r2:.5f}")
    print(f"Explained Variance Score: {evs:.5f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")

    # Plot training progress
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Training and Test Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plot predictions vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_original, y_pred_original, alpha=0.5)
    plt.plot([y_test_original.min(), y_test_original.max()],
             [y_test_original.min(), y_test_original.max()],
             'r--', lw=2)
    plt.xlabel('Actual Gold Price')
    plt.ylabel('Predicted Gold Price')
    plt.title('Predicted vs Actual Gold Prices')
    plt.show()

def main():
    # Load data
    print("Loading data...")
    df = load_data('dataset/combined_dataset.csv')
    print(f"\nDataset shape: {df.shape}")

    # Correlation heatmap
    print("\nGenerating correlation heatmap...")
    create_correlation_heatmap(df)

    # Prepare data
    print("\nPreparing data for modeling...")
    train_dataset, test_dataset, feature_scaler, target_scaler = prepare_data(df)

    # Train neural network
    print("\nTraining neural network...")
    input_size = train_dataset.X.shape[1]  # This gets the actual number of features
    model, train_losses, test_losses = train_neural_network(train_dataset, test_dataset, input_size)

    # Evaluate model
    print("\nEvaluating model...")
    evaluate_model(model, test_dataset, train_losses, test_losses, target_scaler)



if __name__ == "__main__":
    main()
