# Credit Spread Forecasting

This project compares two LSTM architectures for forecasting changes in US High Yield credit spreads. The Champion/Challenger design contrasts a level-based baseline with a directional model that uses first-difference targets and a custom directional penalty loss.

## Structure

```text
credit_spread_forecasting/
├── data/
│   ├── BoE_IUDBEDR_Jan10_Dec25.csv
│   ├── BoE_IUDMNPY_Jan10_Dec25.csv
│   ├── BoE_IUDMNZC_Jan10_Dec25.csv
│   ├── BoE_IUQABEDR_Jan10_Dec25.csv
│   └── FTSE_100_Jan10_Dec25.csv
├── docs/
│   └── Credit_Spread_Forecasting_technical_guide.md
├── notebooks/
│   └── credit_spread_forecasting.ipynb
└── README.md
```

## Workflow

The notebook loads official and local market data, creates regime-independent features, trains both LSTM variants, evaluates directional accuracy, and runs a backtest. The current documented result for the directional model is 57.7% directional accuracy and positive backtested P&L.

Before running the notebook, create `data_paths.txt` in this project directory with the local paths for the Bank of England and FTSE 100 CSV files, plus the FRED and CBOE source URLs. The notebook reads this file from its current working directory.

The technical guide in `docs/` contains the methodology, validation approach, results, and production considerations.