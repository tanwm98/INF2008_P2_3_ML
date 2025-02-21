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
        super(GoldPriceNN, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 16)
        self.layer4 = nn.Linear(16, 1)
        self.relu = nn.LeakyReLU(0.01)  # Use LeakyReLU instead
        self.dropout = nn.Dropout(0.1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(16)

    def forward(self, x):
        x = self.bn1(self.relu(self.layer1(x)))
        x = self.dropout(x)
        x = self.bn2(self.relu(self.layer2(x)))
        x = self.dropout(x)
        x = self.bn3(self.relu(self.layer3(x)))
        x = self.layer4(x)
        return x
class GoldDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_data(file_path):
    df = pd.read_csv(file_path)
    selected_columns = ['Date', 'Gold', 'Silver', 'Crude Oil', 'DXY', 'S&P500', 'cpi', 'rates']
    df = df[selected_columns].copy()
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df.dropna(inplace=True)
    return df


def prepare_data(df, train_ratio=0.7):
    df_sorted = df.sort_values('Date')
    cutoff = int(len(df_sorted) * train_ratio)
    train_df = df_sorted.iloc[:cutoff]
    test_df = df_sorted.iloc[cutoff:]

    features = ['Silver', 'Crude Oil', 'DXY', 'S&P500', 'cpi', 'rates']
    target = 'Gold'

    # Separate features and target
    X_train = train_df[features].values
    y_train = train_df[target].values.reshape(-1, 1)  # Reshape for scaling
    X_test = test_df[features].values
    y_test = test_df[target].values.reshape(-1, 1)

    # Scale features
    feature_scaler = MinMaxScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)

    # Scale target (this is crucial!)
    target_scaler = MinMaxScaler()
    y_train_scaled = target_scaler.fit_transform(y_train)
    y_test_scaled = target_scaler.transform(y_test)

    # Create datasets with scaled data
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


def train_neural_network(train_dataset, test_dataset, input_size, epochs=200, batch_size=32, learning_rate=0.001):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

    model = GoldPriceNN(input_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # Added L2 regularization

    train_losses = []
    test_losses = []
    best_loss = float('inf')
    patience = 20  # Increased patience
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            X_test, y_test = next(iter(test_loader))
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)

        train_losses.append(epoch_loss / len(train_loader))
        test_losses.append(test_loss.item())

        if test_loss < best_loss:
            best_loss = test_loss
            patience_counter = 0
            best_model = model.state_dict()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{epochs}], Train Loss: {epoch_loss/len(train_loader):.4f}, Test Loss: {test_loss.item():.4f}')

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
    train_dataset, test_dataset, feature_scaler, target_scaler = prepare_data(df)  # Modified this line

    # Train neural network
    print("\nTraining neural network...")
    input_size = 6  # number of features
    model, train_losses, test_losses = train_neural_network(train_dataset, test_dataset, input_size)

    # Evaluate model
    print("\nEvaluating model...")
    evaluate_model(model, test_dataset, train_losses, test_losses, target_scaler)  # Added target_scaler

if __name__ == "__main__":
    main()
