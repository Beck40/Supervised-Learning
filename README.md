# Supervised Learning Explorations

This repository contains a series of technical explorations into Supervised Learning—a subset of machine learning where models are trained on labelled datasets to map input variables to specific outputs for both classification and regression challenges.

The focus of this repository is on implementing rigorous statistical frameworks, feature engineering, and high-standard validation techniques required for production-grade predictive modelling.

## 📖 Core Definition

Supervised learning involves training an algorithm on a labelled dataset, where each training example is paired with an output label. The model learns to approximate the mapping function:

$$Y = f(X)$$

This allows the model to accurately predict labels for new, unseen data, facilitating data-driven decision-making in complex environments.

## 📂 Repository Structure

To maintain professional engineering standards, this repository is organised as follows:

- **code/**: Contains modular Python scripts and Jupyter Notebooks for data pre-processing, model training, and evaluation.
- **documentation/**: Technical reports, model cards, and validation summaries detailing the statistical rigour and methodology.

## 📊 Primary Project: UK Road Safety Analysis

This project implements a supervised learning pipeline to predict and classify road safety incidents using the UK Government's official dataset.

### 🏗️ Data Strategy & Validation

To ensure model robustness and prevent overfitting, a strict temporal validation strategy has been implemented:

- **Training & Testing (In-Time):** Data from 2020 – 2024 is used for initial model development, cross-validation, and hyperparameter optimisation.
- **Out-of-Time (OOT) Validation:** Data from 2025 is reserved as a final hold-out set. This simulates a real-world production environment to test how well the model generalises to future data points (Temporal Stability).

### 🛠️ Key Methodologies

- **Classification & Regression:** Exploring various architectures (e.g., Logistic Regression, Random Forests, XGBoost) to predict incident severity.
- **Feature Engineering:** Handling geospatial variables, categorical encoding, and time-series components.
- **Performance Metrics:** Evaluation based on Precision-Recall curves, F1-score (for class imbalance), and Calibration plots.

## 📚 References

- **Dataset Source:** [UK Government: Road Accidents & Safety Data](https://www.data.gov.uk/dataset/cb7ae6f0-4be6-4935-9277-47e5ce24a11f/road-accidents-safety-data)
- **Methodology:** Standardised CRISP-DM workflow for supervised machine learning.

---

*Note: This repository is intended for technical demonstration and academic exploration. All implementations focus on transparency, interpretability, and statistical rigour.*
