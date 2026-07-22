# Household Power Consumption Forecasting

Time series forecasting on the [UCI Household Power Consumption dataset](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) — comparing ARIMA, SARIMA, and Holt-Winters Exponential Smoothing to forecast daily household energy usage, deployed as an interactive Streamlit app.

## Problem

Given ~4 years of minute-level power readings from a single household (Dec 2006 – Nov 2010), forecast future daily average power consumption (`Global_active_power`, in kW).

## Approach

**1. Data cleaning & leakage analysis**
The raw dataset includes `Voltage`, `Global_intensity`, and three `Sub_metering` columns. All were excluded from modeling — `Voltage` and `Global_intensity` are related to the target through the physics identity *Power = Voltage × Current*, and the sub-metering columns are components that sum into the target. Using either would let the model "predict" the target algebraically rather than learn real consumption behavior, and neither would be available at genuine forecast time anyway.

**2. Resampling & missing data**
Minute-level data was resampled to daily mean (not sum — mean is robust to partial-day sensor gaps, since sum would understate usage on any day with missing readings). After resampling, 8 of 1,442 days were still missing (4 isolated days, plus a 4-day cluster in Aug 2010) and were linearly interpolated, since the gaps were short and didn't span a full weekly cycle.

**3. EDA**
Confirmed via a 365-day rolling average that the series has **no significant long-term trend** — variation is dominated by strong **yearly seasonality** (winter highs from heating, summer lows). Confirmed via ADF test (p = 0.021) that the series is stationary, requiring no differencing (`d = 0`).

**4. Train/test split**
Chronological split (no shuffling — standard time series requirement) anchored to a full calendar year: train on everything through 2009-11-25, test on the final complete year (2009-11-26 → 2010-11-26), so the test set contains one full seasonal cycle (both a winter and a summer) rather than an arbitrary percentage-based cut that might miss a season entirely.

**5. Modeling**
- **ARIMA(1,0,0)** — order chosen from ACF/PACF: PACF showed a dominant lag-1 spike with a clean cutoff; ACF decayed gradually rather than cutting off, the classic signature of an AR-dominated process.
- **SARIMA(1,0,0)(1,0,0,7)** — added a weekly seasonal term after PACF also showed a secondary, isolated spike at lag-7 (a same-day-last-week effect). The seasonal coefficient was statistically significant (p < 0.001).
- **Holt-Winters (additive, seasonal_periods=7)** — a structurally different approach (level/trend/seasonal decomposition, no autoregression) for a genuine methodology comparison rather than three variants of the same family.

**6. Evaluation**

| Model | Test RMSE (kW) |
|---|---|
| **ARIMA(1,0,0)** | **0.347** |
| Holt-Winters | 0.474 |
| SARIMA(1,0,0)(1,0,0,7) | 1.106 |

**Key finding:** despite SARIMA's seasonal term being statistically significant on training data (and Ljung-Box diagnostics confirming plain ARIMA left real weekly structure unexplained in-sample), SARIMA's out-of-sample forecast was 3x worse than plain ARIMA over the 359-day horizon — a concrete example of in-sample fit quality not guaranteeing forecast generalization, likely from compounding seasonal extrapolation error at long horizons. Simple **ARIMA(1,0,0)** generalized best and was selected for deployment.

**Known limitation:** as an AR(1) model, forecasts decay toward the series' long-run mean within roughly 7–10 days — useful for short-horizon forecasts, less informative further out.

## Deployment

A Streamlit app loads the saved ARIMA model and lets users select a forecast horizon (in days), returning a chart and table of predicted daily power consumption.

```
streamlit run app.py
```

## Tech stack

`pandas` · `numpy` · `matplotlib` · `statsmodels` (ADF test, ACF/PACF, ARIMA, SARIMAX, Holt-Winters, Ljung-Box) · `scikit-learn` (RMSE) · `streamlit`

## Repo structure

```
├── household_consumption.ipynb   # full analysis: EDA → stationarity → modeling → evaluation
├── app.py                        # Streamlit forecasting app
├── arima_model.pkl               # saved fitted ARIMA(1,0,0) model
└── README.md
```

## Author

Shreyansh Pathak — [GitHub](https://github.com/shreyanshp132)
