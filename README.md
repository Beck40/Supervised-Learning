# Supervised and Deep Learning

## Overview
This repository documents my engineering journey in Supervised and Deep Learning.

It will contain production-ready pipelines for Regression, Classification, and Time-Series Forecasting.

My focus is on Production-Grade Modelling: building systems that handle non-stationary data, enforce strict temporal validation (Out-of-Time testing), and optimise for financial P&L rather than just MSE/Accuracy.

---

## Project List

| Project | Type | Tech Stack | Description |
|---------|------|------------|-------------|
| **[Credit Spread Forecasting](./credit_spread_forecasting)** | Time series | PyTorch, LSTM, Attention, Pandas | A Champion/Challenger framework predicting directional changes in US High Yield spreads with a custom directional penalty loss and financial backtesting. |
| **[UK Road Safety](./uk_road_safety)** | Classification | LightGBM, XGBoost, Scikit-learn | A severity classification pipeline trained on UK government road-safety records with out-of-time validation and class-imbalance handling. |
| **[Payment Forecasting](./payment_forecasting)** | Hierarchical forecasting | Prophet, Nixtla, Plotly | A hierarchical forecasting pipeline for global authorisation volumes using MinTrace reconciliation and payment-channel optimisation analysis. |

---

## Technical Concepts & Mathematical Foundations

### 1. The Core Objective: Function Approximation
At its most fundamental level, supervised learning is the process of inferring a function from labelled training data. The goal is not to memorise the data, but to approximate the underlying ground truth function:

$$Y = f(X) + \epsilon$$

**The Intuition:**
- $f(X)$: The hidden pattern we are trying to find (e.g., "How do interest rates affect credit spreads?").
- $\epsilon$ (Noise): The random chaos in the real world that cannot be predicted.

**The Engineer's Job:** To build a model that captures $f(X)$ without capturing $\epsilon$ (Overfitting).

### 2. Deep Learning vs. Classical ML (Feature Abstraction)
This repository utilises both classical algorithms (Gradient Boosting) and Deep Neural Networks (LSTMs). The choice depends on the nature of the features.

- **Classical ML (LightGBM/XGBoost):** Best for structured, tabular data where features are distinct (e.g., Road_Type, Speed_Limit). The model learns decision boundaries to slice the data.
- **Deep Learning (Neural Networks):** Best for unstructured or sequential data (e.g., Time-Series). The network uses hidden layers to perform automatic feature extraction, transforming raw noisy inputs into abstract representations (e.g., converting daily price fluctuations into a market trend signal).

### 3. Generalisation & Temporal Stability
In academic settings, models are often tested using random splits (K-Fold Cross-Validation). In production engineering, this is dangerous because it ignores Time.

**The Production Standard (Out-of-Time Validation):**
- Instead of asking "Did the model memorise the past?", we ask "Can the model predict the future?"
- **In-Time (Training):** The model learns the rules of the world from 2020–2024.
- **Out-of-Time (Testing):** We test if those rules still apply in 2025.

**Why this matters:** If the distribution of data changes (e.g., a new regulation or market crash), a production model must be robust enough to handle the "Drift."

---

## Repository Structure

```text
├── credit_spread_forecasting/
│   ├── data/                         # Local market and rate series
│   ├── docs/                         # Technical guide
│   ├── notebooks/                    # Forecasting notebook
│   └── README.md
├── payment_forecasting/
│   ├── data/                         # Cashless payments dataset
│   ├── notebooks/                    # Forecasting notebook
│   └── README.md
├── uk_road_safety/
│   ├── docs/                         # Historical road-safety data guide
│   ├── notebooks/                    # Classification notebook
│   └── README.md
├── LICENSE
└── README.md
```

> **Note:** This repository is intended for technical demonstration. All implementations focus on transparency, interpretability, and statistical rigour.
