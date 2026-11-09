# Customer360 — Customer Segmentation & Retention Intelligence

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B.svg)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-red.svg)](https://xgboost.readthedocs.io/)
[![SQLite3](https://img.shields.io/badge/SQLite-Persistent%20Store-blue.svg)](https://sqlite.org/)
[![ReportLab & OpenPyXL](https://img.shields.io/badge/Reports-Excel%20%26%20PDF-brightgreen.svg)]()
[![Tests Passing](https://img.shields.io/badge/tests-37%20passed%20100%25-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Customer360** is an enterprise-grade data science, machine learning, and retention intelligence platform. It seamlessly unifies unsupervised behavioral cohort segmentation, 5-algorithm predictive churn modeling with 5-fold cross-validation, threshold tuning, counterfactual what-if simulation, persistent SQLite intervention workflows, transparent data quality auditing, and publication-ready multi-sheet Excel & PDF reporting.

Built with strict data leakage prevention, probability calibration diagnostics, and modern UI/UX design suitable for executive business analytics demonstrations, academic evaluations, and technical viva panels.

---

## 🏛️ Project Architecture

```
customer360/
├── app.py                      # Main entry point, platform architecture & live roadmap
├── pages/                      # Multi-page Streamlit application modules
│   ├── 1_Dashboard.py          # Page 1: Executive KPIs, Revenue at Risk, Filters & Structured Insights
│   ├── 2_Segmentation.py       # Page 2: Auto-K Selection, PCA, Cluster Churn & Transactional RFM
│   ├── 3_Churn_Prediction.py   # Page 3: 5 Classifiers, 5-Fold CV, Overfitting Gap, Threshold & Simulator
│   ├── 4_Retention_Center.py   # Page 4: Priority Score Breakdown, SQLite Queue, Contacts, Notes & Outcomes
│   ├── 5_Data_Upload.py        # Page 5: Transparent Quality Audit, Data Cleaning Engine & Readiness Badges
│   ├── 6_Model_Performance.py  # Page 6: 5-Model Benchmark Leaderboard, Calibration Curves & Selection Rationale
│   └── 7_Reports.py            # Page 7: Styled Multi-Sheet Excel, Executive PDF, 1-Page Brief & SQLite History
├── src/                        # Modular core analytical and machine learning engine
│   ├── __init__.py
│   ├── database.py             # SQLite persistence: interventions, contacts, notes, outcomes & history
│   ├── data_loader.py          # Robust CSV ingestion, encoding handlers & synthetic generators
│   ├── validation.py           # Transparent quality score (0-100), cleaning pipeline & analytical readiness
│   ├── preprocessing.py        # Leakage-free ColumnTransformer, median imputation & scaling
│   ├── segmentation.py         # K-Means (K=2..8 auto-K), unique naming, cluster churn & RFM
│   ├── churn_prediction.py     # 5 Algorithms (LogReg, RF, DT, GradBoost, XGBoost), 5-Fold CV & What-If
│   ├── model_evaluation.py     # Generalization gap, PR-AUC, Brier score, calibration & threshold tuning
│   ├── recommendations.py      # Multi-factor Priority Score, prescriptive rules & message preview
│   ├── reporting.py            # OpenPyXL styled multi-sheet Excel & ReportLab PDF generators
│   └── ui_components.py        # Custom CSS, KPI card builders, Plotly theme templates & sidebar
├── data/
│   ├── customer360.db          # Persistent SQLite database store
│   ├── telco_customer_churn.csv # Benchmark IBM Telco dataset (7,043 customer accounts)
│   ├── retail_transactions.csv  # Transaction-level purchase history for RFM (3,000 orders)
│   └── README.md
├── models/                     # Serialized scikit-learn & XGBoost pipelines (.joblib)
├── tests/                      # Automated unit and integration testing suite (pytest)
│   ├── test_enhancements.py    # Master test suite for 17 advanced platform capabilities
│   ├── test_churn_prediction.py
│   ├── test_data_loader.py
│   ├── test_model_evaluation.py
│   ├── test_preprocessing.py
│   ├── test_recommendations.py
│   ├── test_reporting.py
│   ├── test_segmentation.py
│   └── test_validation.py
├── .streamlit/
│   └── config.toml             # Executive theme & layout configuration
├── requirements.txt            # Package dependencies (scikit-learn, xgboost, reportlab, openpyxl, plotly)
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start & Installation

### 1. Prerequisites
- Python 3.11 or higher
- Git

### 2. Clone Repository & Setup Virtual Environment
```bash
# Clone repository
git clone https://github.com/ZubairWarsi-1/CRSANDSEG.git
cd "CRSANDSEG"

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Windows (CMD):
.\venv\Scripts\activate.bat
# Linux / macOS:
source venv/bin/activate
```

### 3. Install Locked Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Execute Full Automated Test Suite (100% Pass)
```bash
pytest tests/ -v
```

### 5. Launch Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🌟 Comprehensive Functional Capabilities

### 📊 Page 1: Dashboard Overview
- **Executive KPI Cards**: Real-time calculations of Total Customers, Active Accounts, Average Spending, Historical Churn Rate, High-Risk Accounts ($\ge 65\%$), and Expected Monthly Revenue Exposure.
- **Expected Monthly Revenue Exposure**:
  $$\text{Revenue at Risk} = \sum (\text{Monthly Charges} \times \text{Predicted Churn Probability})$$
  Clearly distinguished as statistical exposure rather than guaranteed immediate loss.
- **Interactive Portfolio Filters**: Dynamic cross-filtering by Contract Type, Tenure Range slider, Internet Service, Historical Churn Status, and Predicted Risk Category.
- **Contract Churn Analysis**: Evaluates Month-to-month, One-year, and Two-year churn percentages using the mathematically correct **group denominator**:
  $$\text{Contract Churn Rate} = \frac{\text{Churn in Contract Group}}{\text{Total in Contract Group}} \times 100$$
- **Automated Structured Insights**: Formatted with Observation, Supporting Metric, Business Interpretation, and Prescriptive Recommended Action.

### 🔍 Page 2: Customer Segmentation
- **Dual Segmentation Engine**:
  1. **Behavioral K-Means Clustering**:
     - **Auto-K Evaluation ($K=2 \dots 8$)**: Evaluates inertia and silhouette scores, explicitly justifying $K=3$ (high cohesion ~0.45 and balanced cluster sizes) vs $K=4$ (~0.41).
     - **Collinearity & Redundancy Warnings**: Automatically checks for feature redundancy among Tenure, MonthlyCharges, and TotalCharges ($r > 0.80$).
     - **Unique Data-Driven Cluster Naming**: Discovers interpretable names such as *"Cluster 1: New / Early-Tenure Budget-Conscious"* and *"Cluster 2: Long-Term Veteran High-Spending"*.
     - **Cluster-Level Churn Analysis**: Table and charts tracking attrition rates within each cluster using group denominators.
     - **Prescriptive Segment Actions**: Tailored strategies (Onboarding Guidance, Loyalty Perks, Value Rescue) per cluster.
  2. **Transactional RFM Analysis**:
     - Strict verification of transaction-level timestamps and order amounts.
     - Educational notice explaining why subscription datasets (Telco) use feature clustering while transaction datasets use RFM.
     - Monetary calculated strictly as the sum of order amounts per customer.

### 🎯 Page 3: Churn Prediction
- **5 Supervised Architectures**:
  - Logistic Regression (L2 regularized)
  - Random Forest Classifier (Constrained tree depth to prevent memorization)
  - Decision Tree Classifier
  - Gradient Boosting Classifier
  - XGBoost Classifier
- **Stratified 5-Fold Cross-Validation**: Embedded inside the ColumnTransformer pipeline to prevent leakage.
- **Overfitting Diagnostics**: Compares In-Sample Train F1 vs Out-of-Sample Test F1, alerting users when the generalization gap exceeds 0.12.
- **Interactive Threshold Tuning**: Real-time slider (0.10 to 0.90) recalculating Precision, Recall, F1, and Confusion Matrix with cost-benefit explanations.
- **Precision-Recall (PR) Curve**: PR curve with PR-AUC score displayed alongside the ROC curve.
- **What-If Counterfactual Simulator**: Modify Contract, Spending, Tenure, and Support features for any customer to observe simulated risk changes in real-time.

### 🛡️ Page 4: Retention Center
- **Transparent Multi-Factor Priority Score**:
  $$\text{Priority Score} = (\text{Churn Probability} \times 50) + \left(\frac{\text{Monthly Spend}}{120} \times 35\right) + \left(\left(1 - \frac{\text{Tenure}}{72}\right) \times 15\right)$$
  Displays full point breakdown for each account.
- **Persistent SQLite Queue**: Workflows and assignments are stored in SQLite (`customer_interventions`) and survive server restarts.
- **Customer Contact Timeline**: Log customer contacts (Email, Phone, SMS, Meeting) with outcomes and representative names (`contact_history`).
- **Retention Notes Manager**: Add collaborative internal account notes (`retention_notes`).
- **Observed Retention Outcome Tracker**: Records actual verified outcomes (Retained, Churned, Upgraded, Downgraded) and computes true observed retention success rates.
- **Personalized Communication Previews**: Preview customized Email and SMS copy containing customer variables and discount percentages before dispatching.

### 📁 Page 5: Data Upload & Data Quality
- **Transparent Data Quality Score (0–100)**: Transparent deductions for missing cells, duplicate rows, duplicate IDs, zero-variance columns, negative financials, and impossible tenure.
- **User-Controlled Data Cleaning Engine**:
  - Remove duplicate rows
  - Correct whitespace strings in numeric columns (e.g. TotalCharges)
  - Impute missing values (median for numeric, mode for categorical)
  - Cap extreme outliers using IQR method ($3.0 \times \text{IQR}$)
  - Normalizes text categories
  - Generates Before vs After comparison summary table with clean CSV download and platform activation.
- **Analytical Module Readiness Badges**: Real-time status cards (Ready, Needs Preparation, Incompatible) for Churn Prediction, K-Means Clustering, and RFM Analysis.
- **SQLite Upload History**: Persistent log of all ingested datasets.

### 📈 Page 6: Model Performance & Benchmarking
- **5-Model Competitive Leaderboard**: Side-by-side comparison table including Test Accuracy, Recall, Precision, F1-Score, ROC-AUC, PR-AUC, Brier Score, Train F1, Generalization Gap, and Overfitting Status.
- **Model Selection Rationale**: Recommends the optimal model balancing Recall and Precision against retention intervention costs.
- **Probability Calibration Curve**: Compares predicted probabilities against empirical observed frequencies with Brier score evaluation.

### 📄 Page 7: Reporting & Deliverables
- **Multi-Sheet Styled Excel (.xlsx)**: Generated via OpenPyXL with 5 formatted sheets (Executive Summary, Segments, High-Risk Churn, Retention Plan, Model Leaderboard).
- **Publication-Grade Executive Audit PDF (.pdf)**: Built with ReportLab featuring corporate branding, KPI tables, data quality audit, and strategic action plans.
- **One-Page Leadership Briefing PDF (.pdf)**: High-impact 1-page executive summary designed for leadership.
- **6 Granular CSV Exports**: Churn Predictions, High-Risk Queue, Segmentation, Retention Plan, Model Evaluation, Quality Audit.
- **SQLite Report Generation History**: Tracks timestamps, report types, formats, record counts, and file sizes.

---

## 🧠 Data Science & Machine Learning Theory

### 1. Data Leakage Prevention
To prevent train-test contamination, all imputation, categorical one-hot encoding, and numerical standard scaling occur **strictly inside the folds** via scikit-learn `Pipeline` and `ColumnTransformer`. Customer identifiers and leakage columns are stripped before feature extraction.

### 2. Handling Class Imbalance
In churn datasets where churners are the minority (~27%), models optimize class weights inversely proportional to class frequencies:
$$w_j = \frac{N}{2 \cdot N_j}$$
Evaluation prioritizes **Recall (Sensitivity)** and **PR-AUC** over raw classification accuracy.

### 3. Revenue at Risk Definition
Unlike historical loss, Revenue at Risk reflects forward-looking statistical revenue exposure:
$$\text{Expected Monthly Exposure} = \sum_{i=1}^{N} (\text{MonthlyCharges}_i \times P(\text{Churn}_i))$$

---

## 🎓 Academic Viva & Project Defense Guide

| Defense Question | Technical Viva Answer |
| :--- | :--- |
| **Why is K=3 recommended over K=4 in clustering?** | K=3 achieves the highest silhouette score (~0.4475 vs ~0.412 for K=4) while maintaining balanced cluster proportions (no degenerate cluster < 8%). K=4 adds a redundant split that does not improve mathematical separation. |
| **Why monitor the Generalization Gap?** | An unconstrained tree model can easily memorize training data (Train F1 = 1.00) while dropping to 0.70 on unseen test data. Tracking $\Delta = \text{Train F1} - \text{Test F1}$ and setting depth limits ($\le 6$) ensures true operational generalization. |
| **What does the Brier Score measure?** | The Brier score measures the mean squared difference between predicted probabilities and actual outcomes ($[0, 1]$). Lower Brier scores indicate well-calibrated probabilities, which is essential for accurate financial risk calculations. |
| **Why calculate Contract Churn with group denominators?** | Calculating contract churn against the total portfolio denominator skews the perceived risk. Dividing churn within month-to-month contracts by the total month-to-month subscribers reveals the true within-group attrition rate (~43%). |
| **Why is SQLite used instead of session state?** | Streamlit session state is ephemeral and resets on browser reload or server restart. SQLite provides persistent relational storage for intervention queues, contact event timelines, notes, and outcome verification. |

---

## 📜 License
This project is licensed under the **MIT License**. Created for academic and production machine learning excellence.
