# 🏎️ F1 Pit Stop Prediction — Race Strategy Analysis

A machine learning project that analyzes Formula 1 race telemetry data to predict pit stop events using lap-by-lap driver and tyre performance metrics. Built with a focus on data preprocessing, exploratory analysis, feature engineering, and classification modeling.

---

## 📌 Project Overview

Pit stop strategy is one of the most critical decisions in Formula 1 racing. This project uses historical F1 lap data to build a **Random Forest Classifier** that predicts whether a driver will make a pit stop on a given lap, based on tyre behaviour, race progress, and lap time trends.

---

## 📁 Repository Structure
F1-Pitstop-Time-Analysis/
│
├── F1_Pitstop-Time-Analysis.ipynb # Main Jupyter Notebook
├── README.md # Project documentation
└── dataset/ link: https://drive.google.com/file/d/11nlBa-K628Jx0NGxf4Etx_u_9ZwFnC8H/view?usp=sharing  


---

## 📊 Dataset

> ⚠️ The dataset is also uploaded separately. Please download it and place it in the root directory (or a `dataset/` folder) before running the notebook.

**File:** `f1_strategy_dataset_v4.csv`

**Shape:** 101,371 rows × 16 columns

| Feature | Description |
|---|---|
| `Driver` | Driver abbreviation (e.g., ALB, VER) |
| `LapNumber` | Lap number in the race |
| `Compound` | Tyre compound (SOFT, MEDIUM, HARD, WET, INTERMEDIATE) |
| `Stint` | Stint number for the driver |
| `TyreLife` | Laps completed on the current tyre set |
| `Position` | Track position |
| `LapTime (s)` | Lap time in seconds |
| `Race` | Grand Prix name |
| `Year` | Season year |
| `LapTime_Delta` | Lap-over-lap time change |
| `Cumulative_Degradation` | Cumulative tyre degradation |
| `PitStop` | 🎯 Target — 1 if pit stop occurred, 0 otherwise |
| `PitNextLap` | Whether a pit stop occurs on the next lap |
| `RaceProgress` | Normalized race completion ratio |
| `Normalized_TyreLife` | Normalized tyre age |
| `Position_Change` | Change in track position |

---

## ⚙️ Methodology

### 1. Data Preprocessing
- Sorted data chronologically by `Year`, `Race`, `Driver`, and `LapNumber`
- Median imputation applied to `LapTime (s)` and `TyreLife`
- Label encoding for categorical features: `Driver`, `Compound`, `Race`

### 2. Exploratory Data Analysis (EDA)
- Target class distribution (PitStop: 75,868 non-stop vs. 25,503 stop laps)
- Average lap time trend across lap numbers
- Tyre Life vs. Lap Time scatter by compound type
- Pit stop rate per tyre compound
- Full correlation heatmap across all numeric features

### 3. Feature Engineering
- **Pace Trend:** 3-lap rolling mean of lap time per driver per race
- **Lag Feature:** Previous lap time (`LapTime_lag1`) per driver

### 4. Feature Extraction — PCA
- Applied `StandardScaler` followed by `PCA (n_components=5)`
- Cumulative explained variance: ~62.8%

### 5. Model Training
- **Algorithm:** Random Forest Classifier (`class_weight='balanced'`)
- **Validation:** TimeSeriesSplit (5 folds) — preserves temporal order
- **Tuning:** GridSearchCV over `n_estimators`, `max_depth`, `min_samples_split`

---

## 🏆 Results

| Metric | Value |
|---|---|
| **Best CV F1 Score** | 0.483 |
| **Test Accuracy** | 74.2% |
| **Test F1 Score** | 0.624 |
| Precision (Class 1 — Pit) | 0.63 |
| Recall (Class 1 — Pit) | 0.62 |

**Best Hyperparameters:**
- `n_estimators`: 150
- `max_depth`: 10
- `min_samples_split`: 5

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data-lightgrey?logo=pandas)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-teal)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)

- **Data Handling:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **ML Pipeline:** `scikit-learn` — `RandomForestClassifier`, `GridSearchCV`, `TimeSeriesSplit`, `PCA`, `StandardScaler`, `LabelEncoder`

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Run the Notebook
```bash
git clone https://github.com/your-username/F1-Pitstop-Time-Analysis.git
cd F1-Pitstop-Time-Analysis
# Add the dataset CSV to the root or dataset/ folder
jupyter notebook F1_Pitstop-Time-Analysis.ipynb
```

---

## 📬 Contact

**Ishaan Sharma**
BTech Computer Science (AI & ML)
🔗 [GitHub](https://github.com/ishaan690)
