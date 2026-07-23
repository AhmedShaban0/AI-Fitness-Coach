# 🏋️ NTI Fitness Coach AI & Business Intelligence Analytics Platform

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Random Forest Production](https://img.shields.io/badge/Production_Model-Random_Forest-2E7D32?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Multi-Model ML](https://img.shields.io/badge/Multi--Model_ML-Benchmarking-0073EC?style=for-the-badge&logo=cpu&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.18+-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

An enterprise-grade **Multi-Model Machine Learning Benchmarking & Business Intelligence Analytics Suite** developed for physical exercise metabolic expenditure prediction, caloric burn rate modeling, and nutritional intake estimation. Developed under the accreditation of the **National Telecommunications Institute (NTI) — AI & Data Science Division**.

---

## 🌟 Executive Summary & ML Architecture

The platform serves as a complete end-to-end Machine Learning Lifecycle ecosystem. It dynamic benchmarks multiple regressor architectures to identify the optimal statistical model for production deployment.

### 🏆 Production Model Selection: Random Forest Regressor
Through automated cross-model metric benchmarking across **Linear Regression**, **Decision Tree Regressor**, **Random Forest Regressor**, and **XGBoost Regressor**, **Random Forest** achieved the highest coefficient of determination ($R^2 = 99.97\%$) and lowest absolute error ($MAE = 1.18\text{ kcal}$, $RMSE = 1.54\text{ kcal}$). Consequently, Random Forest is dynamically assigned as the **Production Winning Estimator** serving real-time single-profile inference and high-throughput CSV batch predictions.

---

## 🔥 Key Platform Capabilities

- **🧪 Multi-Model Regressor Benchmarking**:
  - Automated comparative training matrix evaluating **Linear Regression**, **Decision Tree**, **Random Forest**, and **XGBoost**.
  - Dynamic metric ranking ($R^2$, $MAE$, $RMSE$, training execution time) with zero data leakage.
- **🎯 Dual-Target Metabolic Prediction Engine**:
  - **Active Caloric Expenditure Target**: Predicts active workout burn rate (`calories_burned`) using user physical profiles, heart rate dynamics, step counts, and workout intensity.
  - **Daily Maintenance Target (TDEE / BMR)**: Calculates basal metabolic rate (Harris-Benedict equation) and daily required caloric intake (`required_calories`) for weight goals.
- **🏆 Random Forest Production Inference**:
  - Real-time single-profile inference console (68% left input wizard / 32% right live panel) featuring calculated BMI category badges, interactive Plotly gauge meters, confidence intervals ($\pm\text{MAE}$), and health advisories.
- **📂 High-Speed CSV Batch Prediction Engine**:
  - High-throughput bulk vector processing for uploaded CSV activity logs with animated progress tracking and instant CSV results export.
- **📈 Power BI / Tableau-Grade Executive BI Dashboard**:
  - 7 executive analytical modules featuring global interactive filters, demographic donut charts, 2D density heatmaps (Heart Rate vs. Burn Rate), caloric burn efficiency rates (`kcal/min`), and actionable business recommendations.
- **📊 Interactive Dataset Explorer**:
  - Full tabular search, column data types, missing value audit, and summary statistics over 687,000+ fitness activity records.
- **🧠 Model Diagnostics & Feature Attribution**:
  - Residual error distribution plots, actual vs. predicted regression scatter plots, and Gini feature importance ranking.

---

## 💻 Technology Stack

| Domain | Technologies & Libraries |
| :--- | :--- |
| **Language** | Python 3.11+ |
| **Production ML Model** | **Random Forest Regressor** (Scikit-Learn) |
| **ML Benchmarking Suite** | Linear Regression, Decision Tree, Random Forest, XGBoost, Joblib |
| **Data Processing & Analytics** | Pandas, NumPy, SciPy |
| **Data Visualization** | Plotly Express, Plotly Graph Objects |
| **Web Framework & UI** | Streamlit, Vanilla CSS Glassmorphism Design System |

---

## 📊 Benchmark Model Comparison Matrix

| Model Architecture | Role / Status | Test $R^2$ Score | Test MAE (kcal) | Test RMSE (kcal) | Generalization Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Random Forest Regressor** | **🏆 Production Winner** | **0.9997** | **1.18** | **1.54** | Optimal Generalization |
| **XGBoost Regressor** | Benchmark Candidate | 0.9989 | 1.84 | 2.31 | High Accuracy |
| **Decision Tree Regressor** | Benchmark Candidate | 0.9975 | 2.65 | 3.42 | Baseline Tree |
| **Linear Regression** | Baseline Regressor | 0.9240 | 14.85 | 18.20 | Underfitting Baseline |

---

## 📁 Repository Architecture

```text
AI-Fitness-Coach/
├── dashboard/
│   ├── app.py                      # Main Streamlit Multi-Page Navigation Entrypoint
│   ├── components/
│   │   ├── cards.py                # Standardized Glassmorphic KPI & Summary Cards (22px radius, gradient border)
│   │   └── charts.py               # Enterprise Plotly Engine & Density Heatmap Generators
│   ├── models/
│   │   └── train.py                # Multi-Model Benchmark Pipeline & Metrics Serializer
│   ├── pages/
│   │   ├── 1_🏠_Home.py            # Platform Landing Page & System Workflow Diagram
│   │   ├── 2_📊_Dataset_Explorer.py # Tabular Inspection, Missing Values Audit & CSV Export
│   │   ├── 3_📈_Analytics_Dashboard.py # Executive BI Suite (7 Analytical Modules, 68/32 Layout)
│   │   ├── 4_⚙️_Machine_Learning.py  # Benchmark Comparison Matrix, Residuals & Feature Attribution
│   │   ├── 5_🎯_Prediction.py       # Real-Time AI Prediction Console (Random Forest Production Engine)
│   │   ├── 6_📂_Batch_Prediction.py # High-Throughput Bulk CSV Batch Processing Engine
│   │   ├── 7_🧠_Model_Insights.py   # Estimator Specifications, ONNX Compression & SLA Latency
│   │   └── 8_ℹ️_About.py            # Project Credits & Accreditation Metadata
│   ├── styles/
│   │   └── custom.css              # Glassmorphic Theme System, CSS Variables & Hover Animations
│   └── utils/
│       ├── data_loader.py          # Data Ingestion, MET Calculation & Goal Engineering
│       ├── preprocessing.py         # Saved LabelEncoders & StandardScaler Safety Pipeline
│       └── self_test.py            # Pre-Flight Diagnostic Testing Suite
├── fitness_coach_final.ipynb       # Exploratory Notebook & Initial Model Development
├── my_dataset.csv                  # Primary Fitness Activity Dataset (687K Records)
├── pyproject.toml                  # Metadata & Dependency Specifications
├── requirements.txt                # Pip Dependencies File
└── README.md                       # Documentation
```

---

## 🚀 Local Setup & Installation

### 1. Prerequisites
Ensure **Python 3.11 or higher** is installed on your system.

### 2. Clone Repository
```bash
git clone https://github.com/AhmedShaban0/AI-Fitness-Coach.git
cd AI-Fitness-Coach
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

### 5. Launch Enterprise Web Platform
```bash
streamlit run dashboard/app.py
```
Open your browser at `http://localhost:8501`.

---

## 🏛️ Accreditation & License

Developed under the accreditation of the **National Telecommunications Institute (NTI)** — AI & Data Science Division.
Distributed under the **MIT License**.
