# windAI_rug 🌬️⚡

An end-to-end **machine learning system for short-term wind power forecasting**, developed for the **Nordic Wind Power Forecasting Contest (Norway)**.

The project predicts **hourly wind power up to 48 hours ahead** across multiple Norwegian bidding zones by integrating real-world weather and production data, advanced feature engineering, and multiple forecasting models.

---

## 🔍 Project Overview

Accurate short-term wind power forecasts are critical for:
- energy market bidding,
- grid stability,
- and renewable integration.

This project implements a **production-style forecasting pipeline** that:
- preprocesses raw weather and power data,
- trains and compares multiple forecasting models,
- and serves predictions through an API-ready architecture.

---

## 🧠 Methodology

### Data & Feature Engineering
- Cyclical time encodings (hour, day, seasonality)
- Wind vector decomposition (direction & magnitude)
- Statistical features (lags, rolling windows)
- Multi-zone aggregation for regional forecasting

### Models Implemented
- **ARIMA / SARIMA** (statistical baselines)
- **GRU**
- **LSTM**
- **Transformer-based models**

Models are evaluated comparatively to assess performance across regions and horizons.

---

## 🚀 Pipeline Structure

1. **Preprocessing**
   - Data cleaning and alignment
   - Feature engineering
   - Windowed time-series construction

2. **Model Training**
   - Classical statistical models
   - Deep learning sequence models

3. **Forecasting & Evaluation**
   - 48-hour ahead hourly predictions
   - Error analysis across bidding zones

4. **Deployment**
   - **FastAPI** prediction endpoint
   - **Dockerized** pipeline for reproducible deployment

---

## 📦 Repository Structure

```text
.
├─ data/                 # Raw and processed datasets (not included)
├─ notebooks/            # Exploration and experiments
├─ src/                  # Core preprocessing, models, and training logic
├─ main.py               # Pipeline orchestration
├─ requirements.txt      # Python dependencies
└─ README.md
