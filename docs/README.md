# 🧾 Technical Documentation — Gold Price Prediction

Comprehensive guide for setup, data sources, model architecture, training, and troubleshooting.

---

## 📋 Table of Contents

1. [Environment Setup](#environment-setup)
2. [Data Sources](#data-sources)
3. [Data Pipeline](#data-pipeline)
4. [Model Architecture](#model-architecture)
5. [Training Configuration](#training-configuration)
6. [Performance Metrics](#performance-metrics)
7. [Usage Guide](#usage-guide)
8. [API Reference](#api-reference)
9. [Troubleshooting](#troubleshooting)
10. [Changelog](#changelog)

---

## ⚙️ Environment Setup

### 1️⃣ Create and activate a virtual environment

```bash
python -m venv .venv

# On macOS/Linux:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

### 2️⃣ Install dependencies

Install all dependencies:

```bash
pip install -r requirements.txt
```

### 3️⃣ Verify installation

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
python -c "import yfinance as yf; print('yfinance installed')"
```

---

### 4️⃣ Linting and Formatting

```bash
make lint
make format
```

**Install pre-commit hooks** (one-time setup): `pre-commit install`

---

## 💾 Data Sources

| Asset | Ticker | Source | Description |
|-------|---------|---------|-------------|
| Gold | `GC=F` | Yahoo Finance | Gold Futures |
| Silver | `SI=F` | Yahoo Finance | Silver Futures |

### Economic Indicators

| Indicator | Ticker | Source | Description |
|------------|---------|---------|-------------|
| 10-Year Treasury Yield | `^TNX` | Yahoo Finance | Interest rate benchmark |
| USD Index | `DX-Y.NYB` | Yahoo Finance | Dollar strength |
| S&P 500 | `^GSPC` | Yahoo Finance | Stock market index |
| VIX | `^VIX` | Yahoo Finance | Volatility index |
| Crude Oil | `CL=F` | Yahoo Finance | Commodity influence |
| CPI Inflation | `CPIAUCSL` | FRED | Consumer Price Index |
| Federal Funds Rate | `FEDFUNDS` | FRED | US interest rate |

---

## 🔄 Data Pipeline

1. **Collection** – Fetch data via APIs (yfinance, FRED)  
2. **Cleaning** – Handle missing values, remove outliers, align dates  
3. **Feature Engineering** – Moving averages, RSI, volatility, lag features  
4. **Normalization** – Scale features (MinMaxScaler)  
5. **Splitting** – 70% train / 15% validation / 15% test  
6. **Sequence Creation** – Reshape into `(samples, 60 timesteps, n_features)`

---

## 🧠 Model Architecture

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(60, n_features)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
```

---

## ⚙️ Training Configuration

```python
EPOCHS = 100
BATCH_SIZE = 32
SEQUENCE_LENGTH = 60
LEARNING_RATE = 0.001
```

---

## 📈 Performance Metrics

| Metric | Formula | Target |
|--------|----------|--------|
| **RMSE** | √MSE | < $50 |
| **MAE** | mean(|y_true - y_pred|) | — |
| **MAPE** | mean(|(y_true - y_pred)/y_true|)*100 | < 3% |
| **R²** | Coefficient of determination | > 0.90 |

---

## 🧭 Usage Guide

```bash
jupyter notebook gold_price_prediction_lstm.ipynb
```

### Making Predictions

```python
from tensorflow.keras.models import load_model
model = load_model('gold_price_lstm_model.h5')
```

---

## 🧩 API Reference

```python
def fetch_gold_data(start_date, end_date):
    return yf.download('GC=F', start=start_date, end=end_date)
```

---

## 🛠️ Troubleshooting

| Issue | Solution |
|--------|-----------|
| yfinance fails | `pip install --upgrade yfinance` |
| GPU not detected | `pip install tensorflow[and-cuda]` |
| Overfitting | Increase dropout or add early stopping |

---

## 🧱 Changelog

- **v1.0.0:** Initial LSTM model with 95%+ accuracy
- **v1.1.0:** Streamlit dashboard (planned)
- **v2.0.0:** Multi-asset support
