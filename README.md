# Supervised and Deep Learning

## Overview
This repository documents my engineering journey in Supervised and Deep Learning.

It will contain production-ready pipelines for Regression, Classification, and Time-Series Forecasting.

My focus is on Production-Grade Modelling: building systems that handle non-stationary data, enforce strict temporal validation (Out-of-Time testing), and optimise for financial P&L rather than just MSE/Accuracy.

---

## Project List

| Project                        | Type         | Tech Stack                        | Description |
|--------------------------------|--------------|-----------------------------------|-------------|
|**[Credit Spread Forecasting](./code/credit_spread_forecasting.ipynb)**  | 📈 Time-Series | PyTorch (LSTM) Attention Pandas   | A Champion/Challenger framework predicting directional changes in US High Yield spreads. Features a custom Directional Penalty Loss and Attention Mechanism, achieving 57.7% directional accuracy and £230k P&L in backtesting |
| **[UK Road Saftey](./code/)**        | 🚦 Classification | LightGBM XGBoost Sklearn         | A severity classification pipeline trained on 1.2M+ government records. Implements strict Temporal Validation (OOT) (training on 2020-2024, testing on 2025) to test stability against regime changes and handles extreme class imbalance |
| **[Payment Forecasting](./code/Payment_forecasting.ipynb)**        | 💳 Payments (HTS) | Prophet, Nixtla, Plotly    | A hierarchical (HTS) pipeline forecasting global authorisation volumes. Implements MinTrace reconciliation and a prescriptive optimisation score to identify market share leakage and systemic volume contraction|

---

## Technical Concepts & Mathematical Foundations

### 1. The Core Objective: Function Approximation
At its most fundamental level, supervised learning is the process of inferring a function from labelled training data. The goal is not to memorise the data, but to approximate the underlying ground truth function:

$$Y = f(X) + \epsilon$$

**💡 The Intuition:**
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
- **code/**: Modular python scripts and jupyter notebooks.
- **docs/**: Technical whitepapers, backtest results, and architectural diagrams.
- **data/**: Sample datasets in CSV format.

> **Note:** This repository is intended for technical demonstration. All implementations focus on transparency, interpretability, and statistical rigour.
