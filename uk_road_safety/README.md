# UK Road Safety

This project classifies UK road-accident severity into slight, serious, and fatal outcomes. It combines accident, casualty, and vehicle records from the UK Department for Transport and evaluates model stability with out-of-time testing.

## Structure

```text
uk_road_safety/
├── docs/
│   └── Understanding-historical-road-safety-data.docx
├── notebooks/
│   └── UK Road Safety.ipynb
└── README.md
```

## Workflow

The notebook downloads the latest five-year accident, casualty, and vehicle CSV files, merges the datasets, explores class imbalance, trains classification models, and evaluates performance. The out-of-time section uses provisional 2025 data to test generalisation beyond the 2020-2024 training period.

The notebook uses Python packages including Pandas, Scikit-learn, XGBoost, LightGBM, imbalanced-learn, Matplotlib, and Seaborn. The historical data guide in `docs/` provides additional context for interpreting the source data.