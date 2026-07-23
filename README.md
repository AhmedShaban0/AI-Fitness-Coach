# 🏋️ NTI Fitness Coach AI & Business Intelligence Analytics Platform

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-2196F3?style=for-the-badge&logo=xgboost&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.18+-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

An enterprise-grade **Machine Learning & Executive Business Intelligence Suite** developed for physical exercise metabolic planning, caloric expenditure prediction, and nutritional target estimation. Built for the **National Telecommunications Institute (NTI) AI & Data Science Division**.

---

## 🌟 Key Features

- **🎯 Dual-Target Predictive AI Engine**:
  - **Active Caloric Expenditure Target**: Estimates active workout burn rate (`calories_burned`) using user physical profiles, heart rate dynamics, and workout parameters.
  - **Daily Maintenance Target (TDEE / BMR)**: Calculates basal metabolic rate and daily required caloric intake (`required_calories`) based on individual health goals.
- **⚡ High-Precision Machine Learning Architecture**:
  - Multi-estimator benchmark comparing **Linear Regression**, **Random Forest Regressor**, and **XGBoost Regressor**.
  - Achieves **$R^2 = 99.97\%$** accuracy and **$MAE = 1.18\text{ kcal}$** mean error margin.
- **📈 Executive Business Intelligence Suite**:
  - **Power BI / Tableau-grade** executive scorecard featuring interactive global filters, population demographics, 2D density heatmaps, and top activity efficiency yield cards.
- **⚡ Real-Time AI Prediction Workbench**:
  - Interactive profile input wizard with calculated BMI badges, gauge meters, confidence intervals ($\pm\text{MAE}$), and automated metabolic health recommendations.
- **📂 Bulk CSV Batch Inference Engine**:
  - Process thousands of fitness activity records via bulk CSV upload with progress tracking and instant CSV download exports.
- **🧠 Model Evaluation Deep Dive & Feature Attribution**:
  - Residual error distribution plots, actual vs. predicted regression scatter plots, and Gini feature importance ranking.

---

## 💻 Technology Stack

| Category | Technologies |
| :--- | :--- |
| **Language** | Python 3.11+ |
| **Machine Learning** | Scikit-Learn, XGBoost, Joblib |
| **Data Processing** | Pandas, NumPy, SciPy |
| **Data Visualization** | Plotly Express, Plotly Graph Objects |
| **Web Framework & UI** | Streamlit, Vanilla CSS Glassmorphism Design System |

---

## 📁 Repository Structure

```text
NTI Proj/
├── dashboard/
│   ├── app.py                      # Main Streamlit Navigation & Entry Point
│   ├── components/
│   │   ├── cards.py                # Glassmorphic KPI & Executive Summary Cards
│   │   └── charts.py               # Enterprise Plotly Charting Engine & Density Heatmaps
│   ├── models/
│   │   └── train.py                # ML Model Training Pipeline & Metrics Serializer
│   ├── pages/
│   │   ├── 1_🏠_Home.py            # Platform Landing Page & Architecture Overview
│   │   ├── 2_📊_Dataset_Explorer.py # Interactive Dataset Inspection & Column Metadata
│   │   ├── 3_📈_Analytics_Dashboard.py # Executive BI Suite (7 Analytical Modules)
│   │   ├── 4_⚙️_Machine_Learning.py  # Model Benchmark Comparison & Residual Diagnostics
│   │   ├── 5_🎯_Prediction.py       # Real-Time AI Inference Console (68% / 32% Layout)
│   │   ├── 6_📂_Batch_Prediction.py # High-Speed Bulk CSV Batch Processing Engine
│   │   ├── 7_🧠_Model_Insights.py   # Estimator Specifications & SLA Benchmarks
│   │   └── 8_ℹ️_About.py            # Project Credits & Institution Info
│   ├── styles/
│   │   └── custom.css              # Custom CSS Tokens & Glassmorphism Theme System
│   └── utils/
│       ├── data_loader.py          # Data Ingestion, MET Calculation & Goal Engineering
│       ├── preprocessing.py         # Categorical LabelEncoders & StandardScaler Utilities
│       └── self_test.py            # Pre-Flight Diagnostic Test Suite
├── fitness_coach_final.ipynb       # Exploratory Data Analysis & Notebook Model Benchmarking
├── my_dataset.csv                  # Core Fitness Activity Dataset
├── pyproject.toml                  # Project Metadata & Dependencies Specification
├── requirements.txt                # Pip Dependencies File
└── README.md                       # Documentation
```

---

## 🚀 Quick Start & Local Setup

### 1. Prerequisites
Ensure you have **Python 3.11 or higher** installed on your system.

### 2. Clone Repository
```bash
git clone https://github.com/your-username/nti-fitness-coach-ai.git
cd nti-fitness-coach-ai
```

### 3. Create & Activate Virtual Environment
- **Windows (PowerShell)**:
  ```powershell
  python -m venv .venv
  \.venv\Scripts\Activate.ps1
  ```
- **macOS / Linux**:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Launch Application
```bash
streamlit run dashboard/app.py
```
Open your browser at `http://localhost:8501`.

---

## 🖼️ Application Modules

- **🏠 Home & System Architecture**: Landing page with multi-stage machine learning workflow.
- **📊 Dataset Explorer**: Interactive data table search, column data types, missing value audit, and summary statistics.
- **📈 Executive Analytics Dashboard**: 7 executive BI modules including caloric burn efficiency, density heatmaps, demographic donut charts, and business recommendations.
- **⚙️ Machine Learning Benchmarks**: Multi-algorithm metric comparison table, residual plots, and top predictive feature attribution.
- **🎯 Real-Time AI Prediction Console**: 2-column input wizard (68% left / 32% right) with calculated BMI badges and gauge meters.
- **📂 Batch Prediction Engine**: High-speed batch processing for bulk CSV files.

---

## 📜 License & Accreditation

Developed under the accreditation of the **National Telecommunications Institute (NTI)** — AI & Data Science Division.
Distributed under the **MIT License**.
