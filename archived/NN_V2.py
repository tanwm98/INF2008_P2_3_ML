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

            self.fc1 = nn.Linear(input_size, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 1)

            self.bn1 = nn.BatchNorm1d(64, momentum=0.2)
            self.bn2 = nn.BatchNorm1d(32, momentum=0.2)
            self.bn3 = nn.BatchNorm1d(1, momentum=0.2)
            self.dropout1 = nn.Dropout(0.1)
            self.dropout2 = nn.Dropout(0.1)

            self.activation = nn.LeakyReLU(0.1)

        def forward(self, x):
            x = self.fc1(x)
            x = self.bn1(x)
            x = self.activation(x)
            x = self.dropout1(x)

            x = self.fc2(x)
            x = self.bn2(x)
            x = self.activation(x)
            x = self.dropout2(x)

            x = self.fc3(x)
            x = self.bn3(x)

            return x

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


    def augment_data(X, y, noise_level=0.02):
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
        # Primary features based on correlation
        primary_features = ['Silver', 'S&P500', 'cpi']  # Highest correlations: 0.92, 0.94, 0.89

        # Technical indicators
        df['Gold_MA5'] = df['Gold'].rolling(window=5).mean()
        df['Gold_MA10'] = df['Gold'].rolling(window=10).mean()
        df['RSI'] = calculate_rsi(df['Gold'], periods=14)

        # Price changes and momentum
        df['Gold_Return'] = df['Gold'].pct_change()
        df['Silver_Return'] = df['Silver'].pct_change()
        df['SP500_Return'] = df['S&P500'].pct_change()

        # Volatility
        df['Gold_Vol'] = df['Gold'].rolling(window=10).std()

        # Ratios
        df['Gold_Silver_Ratio'] = df['Gold'] / df['Silver']


        # Add EMA
        df['Gold_EMA5'] = df['Gold'].ewm(span=5, adjust=False).mean()
        df['Gold_EMA10'] = df['Gold'].ewm(span=10, adjust=False).mean()

        # Add ROC
        df['Gold_ROC'] = df['Gold'].pct_change(periods=5)

        # Add more interaction features
        df['Gold_Silver_Change'] = df['Gold_Return'] - df['Silver_Return']
        df['Price_Momentum'] = df['Gold_Return'].rolling(window=5).mean()
        df.dropna(inplace=True)

        # Update your features list
        features = primary_features + [
            'Gold_MA5', 'Gold_MA10', 'Gold_EMA5', 'Gold_EMA10', 'RSI',
            'Gold_Return', 'Silver_Return', 'SP500_Return',
            'Gold_Vol', 'Gold_Silver_Ratio', 'Gold_ROC',
            'Gold_Silver_Change', 'Price_Momentum'
        ]

        # Sort and split data
        df_sorted = df.sort_values('Date')
        cutoff = int(len(df_sorted) * train_ratio)
        train_df = df_sorted.iloc[:cutoff]
        test_df = df_sorted.iloc[cutoff:]

        # Prepare feature matrices
        X_train = train_df[features].values
        y_train = train_df['Gold'].values.reshape(-1, 1)
        X_test = test_df[features].values
        y_test = test_df['Gold'].values.reshape(-1, 1)

        # Scale features
        feature_scaler = MinMaxScaler()
        X_train_scaled = feature_scaler.fit_transform(X_train)
        X_test_scaled = feature_scaler.transform(X_test)

        # Scale target
        target_scaler = MinMaxScaler()
        y_train_scaled = target_scaler.fit_transform(y_train)
        y_test_scaled = target_scaler.transform(y_test)

        # Create datasets
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
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)

        batch_size = 64
        learning_rate = 0.001
        epochs = 300
        patience = 20

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

        model = GoldPriceNN(input_size)

        # Add weight initialization
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        # Apply weight initialization
        model.apply(init_weights)

        mse_criterion = nn.MSELoss()
        huber_criterion = nn.HuberLoss(delta=1.0)

        def combined_loss(pred, target):
            return 0.7 * huber_criterion(pred, target) + 0.3 * mse_criterion(pred, target)

        criterion = combined_loss

        # More sophisticated optimizer setup
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        train_losses = []
        test_losses = []
        best_loss = float('inf')
        patience_counter = 0
        best_model = None

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            batch_losses = []

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()

                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                batch_losses.append(loss.item())
                epoch_loss += loss.item()

            # Calculate average losses properly
            avg_train_loss = np.mean(batch_losses)

            # Validation phase
            model.eval()
            with torch.no_grad():
                X_test, y_test = next(iter(test_loader))
                test_outputs = model(X_test)
                test_loss = criterion(test_outputs, y_test)

            train_losses.append(avg_train_loss)
            test_losses.append(test_loss.item())

            # Update learning rate
            scheduler.step(test_loss)

            # Early stopping check
            if test_loss < best_loss:
                best_loss = test_loss
                patience_counter = 0
                best_model = model.state_dict().copy()  # Ensure proper copy
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

            if epoch % 10 == 0:
                print(f'Epoch [{epoch}/{epochs}], Train Loss: {avg_train_loss:.4f}, '
                      f'Test Loss: {test_loss.item():.4f}, '
                      f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

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


    def prepare_future_data(df, feature_scaler, num_days=7):
        # Calculate typical daily movement limits from historical data
        daily_changes = df['Gold'].pct_change().dropna()
        daily_std = daily_changes.std()
        max_daily_change = daily_std * 0.75  # Reduced from 1.5 to 0.75 for more realistic movements

        last_rows = df.tail(30)
        last_data = df.iloc[-1].copy()
        last_price = df['Gold'].iloc[-1]

        market_holidays = ['2025-01-01']
        future_dates = pd.date_range(start=df.iloc[-1]['Date'] + pd.Timedelta(days=1),
                                     periods=num_days, freq='B')
        future_dates = future_dates[~future_dates.strftime('%Y-%m-%d').isin(market_holidays)]

        future_data = []
        previous_pred = last_data.copy()
        historical_prices = list(df['Gold'].tail(10))

        # Calculate average daily move from recent history
        avg_daily_move = abs(daily_changes.tail(30)).mean()

        for i, future_date in enumerate(future_dates):
            new_row = previous_pred.copy()
            new_row['Date'] = future_date

            # Calculate allowed price movement range with tighter constraints
            max_move = min(last_price * max_daily_change, 25.0)  # Cap absolute move at $25
            volatility_adjustment = np.random.normal(0, max_move / 5)  # Even smaller volatility

            # Add trend persistence (momentum)
            if i > 0:
                prev_trend = (future_data[-1]['Gold'] - future_data[-2]['Gold']) if i > 1 else 0
                trend_factor = prev_trend * 0.3  # 30% trend persistence
                volatility_adjustment += trend_factor

            # Constrain the movement
            if i == 0:
                new_price = last_price + volatility_adjustment
            else:
                prev_price = future_data[-1]['Gold']
                new_price = prev_price + volatility_adjustment

            # Ensure movement isn't too extreme
            max_up = (last_price if i == 0 else prev_price) + max_move
            max_down = (last_price if i == 0 else prev_price) - max_move
            new_price = np.clip(new_price, max_down, max_up)

            new_row['Gold'] = new_price
            historical_prices.append(new_price)

            # Update technical indicators using the rolling window
            new_row['Gold_MA5'] = np.mean(historical_prices[-5:])
            new_row['Gold_MA10'] = np.mean(historical_prices[-10:])

            # Calculate EMAs
            if i == 0:
                new_row['Gold_EMA5'] = df['Gold'].tail(5).ewm(span=5, adjust=False).mean().iloc[-1]
                new_row['Gold_EMA10'] = df['Gold'].tail(10).ewm(span=10, adjust=False).mean().iloc[-1]
            else:
                new_row['Gold_EMA5'] = new_price * 0.333 + future_data[-1]['Gold_EMA5'] * 0.667
                new_row['Gold_EMA10'] = new_price * 0.182 + future_data[-1]['Gold_EMA10'] * 0.818

            # Calculate returns with proper reference
            prev_gold = last_price if i == 0 else future_data[-1]['Gold']
            new_row['Gold_Return'] = (new_price - prev_gold) / prev_gold
            new_row['Gold_ROC'] = new_row['Gold_Return'] * 100

            # Update volatility
            new_row['Gold_Vol'] = np.std(historical_prices[-10:])

            # Update Silver price based on historical ratio
            avg_ratio = df['Gold_Silver_Ratio'].mean()
            new_row['Silver'] = new_price / avg_ratio

            future_data.append(new_row)
            previous_pred = new_row.copy()

        future_df = pd.DataFrame(future_data)

        # Add controlled noise to other features
        for feature in ['S&P500', 'cpi']:
            feature_std = df[feature].std() * 0.05  # Reduced noise
            future_df[feature] = future_df[feature].mean() + np.random.normal(0, feature_std, size=len(future_df))

        features = ['Silver', 'S&P500', 'cpi', 'Gold_MA5', 'Gold_MA10', 'Gold_EMA5', 'Gold_EMA10',
                    'RSI', 'Gold_Return', 'Silver_Return', 'SP500_Return', 'Gold_Vol',
                    'Gold_Silver_Ratio', 'Gold_ROC', 'Gold_Silver_Change', 'Price_Momentum']

        X_future = future_df[features].values
        X_future_scaled = feature_scaler.transform(X_future)

        return future_dates, X_future_scaled


    def predict_future(model, df, feature_scaler, target_scaler, num_days=7):
        model.eval()
        future_dates, X_future_scaled = prepare_future_data(df, feature_scaler, num_days)

        # Convert to tensor
        X_future_tensor = torch.FloatTensor(X_future_scaled)

        # Make predictions
        with torch.no_grad():
            predictions_scaled = model(X_future_tensor)

        # Convert predictions back to original scale
        predictions = target_scaler.inverse_transform(predictions_scaled.numpy())

        # Create results dataframe
        results_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Gold_Price': predictions.flatten()
        })

        return results_df


    def main():
        # Load data
        print("Loading data...")
        df = load_data('../dataset/combined_dataset.csv')
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

        # After training and evaluating the model
        print("\nPredicting future gold prices...")
        future_predictions = predict_future(model, df, feature_scaler, target_scaler, num_days=7)
        print("\nPredicted Gold Prices for next week:")
        print(future_predictions)

        # Plot the predictions
        plt.figure(figsize=(10, 6))
        plt.plot(future_predictions['Date'], future_predictions['Predicted_Gold_Price'],
                 marker='o', linestyle='-', label='Predicted')
        plt.title('Gold Price Predictions for Next Week')
        plt.xlabel('Date')
        plt.ylabel('Gold Price')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()

    if __name__ == "__main__":
        main()
