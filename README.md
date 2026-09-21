# 📈 Andaza Engine™

> **Replacing manual guesswork with calibrated algorithmic precision.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-F7931E.svg)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.0%2B-3F4F75.svg)](https://plotly.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Overview

**Andaza Engine™** is an enterprise-grade, theme-adaptive time-series analytics and predictive forecasting platform[cite: 2]. Designed to replace spreadsheet-based estimations with automated machine learning workflows, the engine ingests historical tabular data, executes dynamic feature engineering, and generates 7-to-180-day forecast trajectories[cite: 2].

Equipped with an automated CSV validation suite and a theme-responsive UI, the studio provides operations, finance, and supply chain teams with instant predictive intelligence[cite: 2].

---

## ✨ Key Features

- 🛡️ **Automated Data Validation Engine**: Runs a 5-point structural inspection on uploaded files (CSV validity, minimum row density, parseable datetime columns, numeric metrics, and null value ratios) before initiating training.
- ⚙️ **Autonomous Feature Pipeline**: Converts raw time-series timestamps into high-dimensional temporal features (Day of Week, Month, Quarter, Day of Year) alongside rolling lag statistics ($Lag_1$, $Lag_7$, $Lag_{30}$, $Rolling\_Mean$).
- 🤖 **Recursive Random Forest Predictor**: Trains multi-tree regressors on historical baselines and executes iterative recursive forecasting for forward horizons[cite: 2].
- 📊 **Metric Telemetry & Diagnostics**: Computes real-time evaluation metrics—**MAE**, **RMSE**, and **$R^2$ Score**—alongside feature impact ranking charts[cite: 2].
- 🌓 **Dynamic Theme Engine**: Built using native CSS variables (`var(--background-color)`, `var(--text-color)`) and transparent Plotly layers to transition smoothly between Dark and Light modes without UI glitches.
- 📦 **Data Export Suite**: Generates downloadable projections in standard CSV formats for downstream integration into business intelligence tools[cite: 2].

---

## 🏗️ Architecture & Stack

| Layer | Technology / Library |
| :--- | :--- |
| **Frontend Framework** | [Streamlit](https://streamlit.io/) |
| **Machine Learning** | [scikit-learn](https://scikit-learn.org/) (RandomForestRegressor) |
| **Data Manipulation** | [pandas](https://pandas.pydata.org/), [numpy](https://numpy.org/) |
| **Interactive Plotting** | [Plotly](https://plotly.com/python/) (Graph Objects & Express) |
| **Styling & UI** | Custom CSS Variables, Google Fonts (*Plus Jakarta Sans*) |

---

## 🚀 Quick Start

### Prerequisites

Ensure you have Python 3.9+ installed on your system.

### 1. Clone the Repository

```bash
git clone [https://github.com/your-username/andaza-engine.git](https://github.com/your-username/andaza-engine.git)
cd andaza-engine
