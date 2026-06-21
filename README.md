# Market Trend Analysis & Stock Price Prediction — RELIANCE.NS

A comparative study of statistical, machine learning, and ensemble approaches to equity
price forecasting and volatility modeling, using 7 years of daily OHLCV data for Reliance
Industries Ltd. (NSE: RELIANCE).

## What's inside

- **Data:** Daily OHLCV for RELIANCE.NS, Jan 2018 – Jan 2025, pulled live via `yfinance`.
- **Feature engineering:** SMA/EMA, Wilder's RSI, MACD, Bollinger Bands, realized
  volatility, and multi-day lag features — all causal (no look-ahead leakage).
- **Models compared (same feature set, same split):**
  - Classical time series — ARIMA(5,1,0), Prophet
  - Machine learning — Linear Regression, Random Forest, XGBoost, LightGBM
  - XGBoost and LightGBM are further tuned via `RandomizedSearchCV` with
    `TimeSeriesSplit` cross-validation
- **Evaluation:** RMSE, MAE, MAPE, R², and Directional Accuracy — not just RMSE, since
  price-level RMSE alone is a misleading scoreboard for this task (see the notebook's
  "Methodology" section for why).
- **Volatility modeling:** EWMA and GARCH(1,1), compared and visualized.
- **Robust validation:** Walk-forward (rolling-window) backtesting of the best model,
  rather than a single static train/test split.
- **Interactive visualization:** Plotly chart overlaying all model predictions and GARCH
  volatility.

## Key methodological note

Daily closing prices are close to a random walk, so any model with yesterday's close as a
feature will post a deceptively low RMSE / high R² — that's largely the random-walk
property of price levels, not genuine predictive skill. This notebook is explicit about
that throughout, and uses Directional Accuracy plus walk-forward stability (not single-split
RMSE) as the more honest measures of whether a model is adding real signal. See Section 5
and Section 15 of the notebook for the full discussion.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook MARKET_TREND_ANALYSIS_RELIANCE.ipynb
```

Run all cells top to bottom — the data download cell requires internet access to Yahoo
Finance. Total runtime is a few minutes, dominated by the hyperparameter search cells.

## Project structure

```
.
├── MARKET_TREND_ANALYSIS_RELIANCE.ipynb   # main notebook
├── requirements.txt
└── README.md
```

## Tech stack

`yfinance` · `pandas` / `numpy` · `scikit-learn` · `statsmodels` · `prophet` ·
`xgboost` · `lightgbm` · `arch` (GARCH) · `matplotlib` / `seaborn` · `plotly`

## Limitations & future work

No transaction costs/slippage are modeled, only a single stock/regime is covered, and no
exogenous (macro, sentiment, fundamental) features are included. See the notebook's final
section for a fuller discussion and concrete next steps (return-based reframing, rule-based
strategy backtesting with costs, EGARCH for asymmetric volatility, SHAP explainability,
multi-stock generalization).
