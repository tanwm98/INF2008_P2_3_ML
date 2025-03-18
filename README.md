# Gold Price Prediction Project

## Overview

This project focuses on predicting gold prices using various machine learning algorithms. The implementation includes multiple approaches such as Linear Regression, Neural Networks, Random Forest, XGBoost, and Support Vector Machines (SVM). Each model employs different feature engineering techniques and evaluation methods to accurately predict gold prices based on financial market data.

## Table of Contents

- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Models Implemented](#models-implemented)
- [Feature Engineering](#feature-engineering)
- [Key Results](#key-results)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)
- [Future Work](#future-work)

## Project Structure

The project consists of the following Python scripts:

- `compiled_dataset.py`: Combines financial data from different sources into a unified dataset
- `gold_detail.py`: Fetches gold-specific financial data from Yahoo Finance
- `Linear_Regression.py`: Implements linear regression model for gold price prediction
- `NN_v3.py`: Implements a neural network model with enhanced architecture
- `RF.py`: Implements Random Forest and XGBoost models with advanced features
- `SVM_V2.py`: Implements Support Vector Machine model for gold price prediction

## Dataset

The dataset combines several financial indicators:

- Gold prices
- Silver prices
- Crude Oil prices
- US Dollar Index (DXY)
- S&P 500
- Consumer Price Index (CPI)
- Interest rates

This comprehensive dataset allows the models to capture various economic factors that influence gold prices, including market trends, inflation indicators, and currency strength.

## Models Implemented

### Linear Regression

A baseline model that uses feature selection to identify the most relevant predictors.

### Neural Network

A minimalist architecture with leaky ReLU activation functions, designed to avoid overfitting on the relatively small dataset.

### Random Forest & XGBoost

Tree-based ensemble methods with hyperparameter optimization using Bayesian search. The implementation includes a recalibration mechanism to adjust for prediction bias.

### Support Vector Machine

An SVM regressor with grid search for hyperparameter tuning.

## Feature Engineering

Each model employs sophisticated feature engineering techniques:

- Technical indicators (RSI, moving averages)
- Price momentum features (rate of change)
- Volatility measures
- Economic indicators (real interest rates)
- Market ratios (Gold/DXY, Gold/Silver)
- Lagged features to capture time-series patterns

## Key Results

All models demonstrate reasonable predictive performance, with the XGBoost model typically achieving the best results:

- **R² scores**: 0.70-0.95 (depending on the model and test period)
- **RMSE**: Varies by model but shows significant improvement over baseline predictions
- **Feature importance**: DXY (US Dollar Index), interest rates, and crude oil prices consistently rank among the most important features

## How to Run

1. Ensure all dependencies are installed
2. Place the dataset files in the `dataset` directory:
    - `combined_dataset.csv` (or run `compiled_dataset.py` to generate it)
    - `CPI_2015-2024.csv`
    - `interestrates_2015-2024.csv`
3. Run any of the model files:
    
    Copy
    
    `python Linear_Regression.py python NN_v3.py python RF.py python SVM_V2.py`
    

## Dependencies

- pandas
- numpy
- matplotlib
- scikit-learn
- torch (PyTorch)
- xgboost
- seaborn
- yfinance (for data acquisition)
- scikit-optimize (for Bayesian optimization)

## Future Work

- Implement ensemble techniques to combine predictions from multiple models
- Explore deep learning models like LSTM for better time-series modeling
- Add macroeconomic indicators such as LSE/EURO STOXX 50 and geopolitical risk indices
- Create a web-based dashboard for real-time predictions

---

_Note: This project is for educational purposes. Financial markets are inherently unpredictable, and no model can guarantee accurate predictions._
