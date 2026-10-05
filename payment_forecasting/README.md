# Payment Forecasting

This project forecasts global payment authorisation volumes using hierarchical time-series forecasting. It reconciles forecasts across regions, payment methods, and channels, then ranks channels by growth gaps, market-share erosion, and optimisation priority.

## Structure

```text
payment_forecasting/
├── data/
│   └── BIS_CPMI_CASHLESS_csv_flat.csv
├── notebooks/
│   └── Payment_forecasting.ipynb
└── README.md
```

## Workflow

The notebook loads the BIS cashless-payments dataset, prepares the regional hierarchy, generates forecasts with Prophet and Nixtla tooling, applies MinTrace reconciliation, and produces diagnostic and optimisation visualisations.

The raw-data loading cell currently expects the CSV path to be entered in `csv_path`. Set it to `data/BIS_CPMI_CASHLESS_csv_flat.csv` when running from this project directory. Install the notebook dependencies, including Pandas, NumPy, StatsForecast, Prophet, Plotly, and related scientific Python packages.