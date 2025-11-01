# 🟡 Gold Price Prediction using Machine Learning

An end-to-end **machine learning system** that predicts gold prices with high accuracy using **LSTM neural networks** and **economic indicators**.

---

## 📘 Project Overview

This project builds a predictive model for gold prices by analyzing **historical price data** combined with **economic indicators**, **technical analysis**, and **market sentiment**.  
The model uses **LSTM (Long Short-Term Memory)** deep learning architecture to capture temporal patterns in financial time-series data.

---

## 🎯 Objectives

- Predict gold prices with **95%+ accuracy**
- Analyze the impact of macroeconomic factors on gold price movement
- Build a scalable foundation for **multi-asset forecasting**
- Create an **interactive dashboard** for visualization and forecasting

---

## 🌟 Key Features

- **📈 Historical Data Collection:** Automated fetching of 10+ years of data
- **💹 Multi-Factor Analysis:** Includes interest rates, inflation, USD index, and market indicators
- **🧠 LSTM Neural Network:** Optimized for time-series forecasting
- **📊 Evaluation Metrics:** RMSE, MAE, MAPE, and R²
- **🔮 Future Forecasting:** 7-day ahead predictions with confidence intervals

---

## 🧰 Technology Stack

- **Language:** Python 3.13+
- **Frameworks:** TensorFlow / Keras
- **Libraries:** pandas, numpy, scikit-learn
- **Data Sources:** yfinance, pandas_datareader
- **Visualization:** matplotlib, seaborn, plotly, streamlit

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone <your-repo-url>
cd gold-price-prediction

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook gold_price_prediction_lstm.ipynb
```

---

## 📊 Project Status

🚧 **Active Development** – Currently in **Phase 1: Gold Prediction MVP**

---

## ✅ Results

- Achieved **95%+ accuracy** on test data
- Successfully predicts **trend direction**
- Identifies **key economic drivers** affecting gold prices

---

## 🗺️ Future Roadmap

| Phase | Description |
|--------|-------------|
| **1** | Gold price prediction *(Current)* |
| **2** | Add silver, platinum, and other metals |
| **3** | Integrate stock market prediction |
| **4** | Add cryptocurrency forecasting |
| **5** | Build portfolio optimization dashboard |
| **6** | Deploy REST API and cloud hosting |
| **7** | Develop mobile application |

---

## 📚 Documentation

For complete setup, data sources, architecture, and API usage, see [**DOCS.md**](DOCS.md)