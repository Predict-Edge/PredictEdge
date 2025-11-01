# Gold Price Prediction using Machine Learning

An end-to-end machine learning system that predicts gold prices with high accuracy using LSTM neural networks and economic indicators.

## Project Overview

This project builds a predictive model for gold prices by analyzing historical price data combined with economic indicators, technical analysis, and market sentiment. The model uses deep learning (LSTM) to capture temporal patterns and relationships in financial time-series data.

## Objectives

- Predict gold prices with **95%+ accuracy** using machine learning
- Analyze the impact of economic factors on gold price movements
- Build a scalable foundation for multi-asset price prediction
- Create an interactive dashboard for visualization and forecasting

## Features

- **Historical Data Collection**: Automated fetching of 10+ years of gold price data and economic indicators
- **Multi-Factor Analysis**: Incorporates interest rates, inflation, USD strength, stock market indices, and commodity prices
- **Technical Indicators**: Moving averages, RSI, volatility metrics, and lagged features
- **LSTM Neural Network**: Deep learning model optimized for time-series forecasting
- **Comprehensive Evaluation**: RMSE, MAE, MAPE, and R² metrics for model performance
- **Visualization Dashboard**: Interactive plots showing predictions vs actual prices
- **Future Forecasting**: 7-day ahead price predictions with confidence intervals

## Data Sources

The model uses the following data inputs:

**Price Data:**
- Gold spot prices and futures (10-year history)
- Silver prices (correlated commodity)

**Economic Indicators:**
- US 10-Year Treasury Yields
- USD Index (DXY)
- Federal Funds Rate
- CPI Inflation Data
- S&P 500 Index
- VIX Volatility Index
- Crude Oil Prices

**Technical Features:**
- 7, 30, and 90-day moving averages
- Relative Strength Index (RSI)
- Price volatility (rolling standard deviation)
- Lagged features (1, 3, 5, 7, 10-day lags)

## Technology Stack

**Core Libraries:**
- Python 3.13+
- TensorFlow/Keras (LSTM implementation)
- pandas, numpy (data processing)
- scikit-learn (preprocessing, metrics)

**Data Collection:**
- yfinance (financial market data)
- pandas_datareader (FRED economic data)

**Visualization:**
- matplotlib, seaborn, plotly
- streamlit (interactive dashboard)

## Model Architecture

**LSTM Neural Network:**
- Input Layer: 60 timesteps × multiple features
- LSTM Layer 1: 50 units with dropout (0.2)
- LSTM Layer 2: 50 units with dropout (0.2)
- Dense Layer: 25 units (ReLU activation)
- Output Layer: 1 unit (price prediction)

**Training Configuration:**
- Optimizer: Adam
- Loss Function: Mean Squared Error
- Early Stopping: Patience = 10 epochs
- Batch Size: 32
- Max Epochs: 100

## Data Pipeline

1. **Data Collection**: Fetch historical data from APIs
2. **Data Cleaning**: Handle missing values, outliers, and inconsistencies
3. **Feature Engineering**: Create technical indicators and lagged features
4. **Normalization**: MinMax scaling (0-1 range)
5. **Train/Validation/Test Split**: 70%/15%/15% chronological split
6. **Sequence Creation**: Reshape data for LSTM input format
7. **Model Training**: Train with validation monitoring
8. **Evaluation**: Calculate metrics on test set
9. **Forecasting**: Generate future predictions

## Performance Metrics

The model is evaluated using:
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **MAPE** (Mean Absolute Percentage Error)
- **R² Score** (Coefficient of Determination)

Target accuracy: **95%+ on test data**

## Installation

Install required dependencies: pip install pandas numpy yfinance pandas-datareader scikit-learn tensorflow keras matplotlib seaborn plotly streamlit


## Usage

Run the Jupyter notebook: jupyter notebook fine_name.ipynb

## Results

- Model achieves **[X]% accuracy** on test data (MAPE: [Y]%)
- Successfully predicts gold price trends with [Z] RMSE
- Identifies key economic drivers of gold price movements
- Provides reliable 7-day forecasts with confidence intervals

## Future Enhancements

**Phase 1 - Current:** Gold price prediction model  
**Phase 2:** Expand to silver, platinum, and precious metals  
**Phase 3:** Add stock market predictions (indices and individual stocks)  
**Phase 4:** Cryptocurrency price forecasting  
**Phase 5:** Multi-asset portfolio optimization dashboard  
**Phase 6:** Real-time predictions with automated retraining  
**Phase 7:** REST API for third-party integrations  
**Phase 8:** Mobile application deployment

## Scalability Roadmap

This project is designed to scale into a full-featured asset management platform:
- **Transfer learning** to apply gold model insights to other assets
- **Cloud deployment** for handling thousands of predictions daily
- **Automated data pipelines** for real-time market data ingestion
- **Subscription-based SaaS** model for monetization
- **Enterprise API** for institutional clients

## Project Status

🚧 **Active Development** - Currently in Phase 1 (Gold Prediction MVP)

Current milestone: Building and validating LSTM model with 10 years of historical data
