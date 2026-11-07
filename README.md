# Customer360 — Customer Segmentation & Retention Intelligence

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B.svg)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests Passing](https://img.shields.io/badge/tests-20%20passed-brightgreen.svg)]()

> **Customer360** is a full-stack data science and machine learning web application engineered for executive customer intelligence, behavioral cohort segmentation, predictive churn modeling, and prescriptive retention workflow management.

Designed to production standards with leakage-free scikit-learn preprocessing pipelines, explainable machine learning predictions, and a modern, high-contrast dashboard interface tailored for academic project evaluations, technical vivas, and enterprise analytics demonstrations.

---

## 🏛️ Project Architecture

```
customer360/
├── app.py                      # Main application entry point & roadmap
├── pages/                      # Multi-page Streamlit application
│   ├── 1_Dashboard.py          # Page 1: Portfolio KPIs, distributions & data-driven insights
│   ├── 2_Segmentation.py       # Page 2: RFM Analysis & K-Means Clustering with PCA
│   ├── 3_Churn_Prediction.py   # Page 3: Supervised ML pipelines & individual customer explainer
│   ├── 4_Retention_Center.py   # Page 4: Value-at-Risk prioritization & session action tracker
│   ├── 5_Data_Upload.py        # Page 5: CSV ingestion, schema audit & role mapping
│   ├── 6_Model_Performance.py  # Page 6: Train vs Test diagnostics, ROC, confusion matrix
│   └── 7_Reports.py            # Page 7: Filtered CSV downloads & Markdown audits
├── src/                        # Core Python analytical and machine learning library
│   ├── __init__.py
│   ├── data_loader.py          # Robust CSV loading, encoders, and synthetic generators
│   ├── validation.py           # Schema health check, quality scoring, role inference
│   ├── preprocessing.py        # ColumnTransformer, Imputation, Scaling & Stratification
│   ├── segmentation.py         # RFM quartile scoring, K-Means, Elbow & Silhouette
│   ├── churn_prediction.py     # Logistic Regression & Random Forest with Joblib
│   ├── model_evaluation.py     # Metrics (F1, AUC, Recall), ROC curves, confusion matrices
│   ├── recommendations.py      # Rule-based prescriptive retention engine
│   ├── reporting.py            # CSV export serializers & Markdown reports
│   └── ui_components.py        # Custom CSS, KPI tiles & Plotly theme templates
├── data/
│   ├── telco_customer_churn.csv # Authentic IBM Telco Customer Churn dataset (7,043 rows)
│   ├── retail_transactions.csv  # Transaction-level order history for RFM (3,000 orders)
│   └── README.md
├── models/                     # Serialized Scikit-learn pipelines (.joblib)
├── tests/                      # Automated unit testing suite (pytest)
├── .streamlit/
│   └── config.toml             # Custom theme configuration
├── requirements.txt            # Locked package dependencies
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
# Clone the repository
git clone <repo-url>
cd "Customer Segmentation and Retention System"

# Create a virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Windows (CMD):
.\venv\Scripts\activate.bat
# Linux / macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run Automated Unit Tests
```bash
pytest tests/ -v
```

### 5. Launch Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🌟 Key Functional Modules (Page-by-Page)

### 📊 Page 1: Dashboard Overview
- **Executive KPIs**: Real-time calculations of Total Customers, Active Accounts, Average Spending, Historical Churn Rate, and High-Risk Portfolio Count directly from the active dataset.
- **Visual Distributions**: Donut chart of attrition proportions, tenure distributions, and spending box-plots.
- **Automated Business Insights**: Dynamically discovers statistically significant vulnerability factors (e.g., contract types, tenure drop-off cliffs, and service gaps like Fiber Optic without Tech Support).

### 🔍 Page 2: Customer Segmentation
- **Dual Analytical Modes**:
  1. **Unsupervised K-Means Clustering**: Automatically scales numerical features, evaluates $k=2 \dots 6$ using the **Elbow Method** and **Silhouette Scores**, projects cohorts onto a 2D PCA plane, and generates interpretable segment labels (e.g., *"Cluster 1: High Spend / Low Tenure"*).
  2. **Behavioral RFM Analysis**: Calculates Recency (days), Frequency (orders), and Monetary (spending) quartiles from transaction-level records to classify customers into *Champions*, *Loyal Customers*, *Potential Loyalists*, *At-Risk Customers*, and *Lost / Inactive*.

### 🎯 Page 3: Churn Prediction
- **Leakage-Free Pipeline**: Features `ColumnTransformer` with median imputation, `StandardScaler`, and `OneHotEncoder(handle_unknown='ignore')`.
- **Class-Imbalance Compensation**: Employs `class_weight='balanced'` on both Logistic Regression and Random Forest classifiers.
- **Individual Explainability**: Breaks down single customer risk into positive drivers (risk accelerators) and negative protective factors.
- **Interactive Simulator**: Test hypothetical customer adjustments to observe real-time probability shifts.

### 🛡️ Page 4: Retention Center
- **Value-at-Risk Prioritization**: Ranks customer intervention urgency using a composite Priority Score:
  $$\text{Priority Score} = \text{Churn Probability} \times \min\left(100, \frac{\text{Monthly Spend}}{1.5}\right)$$
- **Prescriptive Actions**: Transparent rule-based workflows:
  - *Personalized Win-Back Offer* (High risk + high spending / Champions)
  - *Onboarding Guidance* (Early tenure $\le 6$ months + high risk)
  - *Customer Support Follow-Up* (Service complaints or high billing friction)
  - *Loyalty Rewards* (High value + low/medium risk)
  - *Re-engagement Campaign* (Declining activity)
- **Session Action Tracker**: Live status workflow updating customer statuses (*Pending*, *In Progress*, *Completed*).

### 📁 Page 5: Data Upload & Governance
- **Self-Service CSV Ingestion**: Drag-and-drop custom CSV datasets.
- **Schema & Missingness Audit**: Inspects data types, null cell frequencies, and duplicate rows.
- **Data Quality Score (0–100)**: Automatically computes an objective health grade (*Excellent*, *Good*, *Fair*, *Poor*).
- **Column Role Mapping**: Allows users to override or confirm Customer ID, Target Churn, Spending, and Tenure columns.

### 📈 Page 6: Model Performance & Diagnostics
- **Leaderboard Comparison**: Side-by-side benchmark table comparing Logistic Regression vs Random Forest.
- **Generalization Audit**: Side-by-side comparison of **Train Set vs Test Set** metrics to inspect overfitting.
- **Interactive Confusion Matrix**: Heatmap displaying True Negatives, False Positives, False Negatives, and True Positives.
- **Receiver Operating Characteristic (ROC)**: Plotly curve with area under curve (AUC) metric.
- **Per-Class Classification Report**: Class-level Precision, Recall, and F1-Scores.

### 📄 Page 7: Reports & Intelligence Exports
- **CSV Deliverables**:
  - `churn_predictions.csv`
  - `high_risk_customers.csv`
  - `segmentation_results.csv`
  - `retention_action_plan.csv` (including session action updates)
  - `model_evaluation_summary.csv`
  - `data_quality_audit.csv`
- **Markdown Governance Summary**: One-click download of the complete dataset audit report.

---

## 🧠 Machine Learning & Data Science Methodology

### 1. Data Preprocessing & Leakage Prevention
To prevent data snooping and data leakage:
- Categorical features are encoded and numerical features are scaled **strictly inside the cross-validation and pipeline steps** using `sklearn.pipeline.Pipeline` and `ColumnTransformer`.
- Target identifiers (`customerID`) and target labels (`Churn`) are decoupled from the predictor feature matrix $X$.
- `train_test_split` uses **stratified sampling** to maintain identical minority-class proportions across both splits.

### 2. Handling Class Imbalance
In churn datasets, retained customers typically outnumber churned customers (e.g., 73% vs 27% in the Telco dataset). Using raw accuracy can produce deceptive results. We mitigate this through:
- Setting `class_weight='balanced'` in estimators, which adjusts weights inversely proportional to class frequencies:
  $$w_j = \frac{n}{k \cdot n_j}$$
- Evaluating models primarily on **Recall (Sensitivity)** and **ROC-AUC** rather than raw accuracy.

### 3. Clustering Validation (K-Means)
- Features are scaled to zero mean and unit variance before Euclidean distance calculation.
- **Silhouette Coefficient**:
  $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
  where $a(i)$ is mean intra-cluster distance and $b(i)$ is mean nearest-cluster distance. Evaluated across $k \in [2, 6]$.

---

## 🎓 College Viva & Academic Evaluation Guide

When presenting this project for an academic evaluation, panel defense, or technical interview, review the following key questions:

| Evaluation Topic | Key Concept / Viva Defense Answer |
| :--- | :--- |
| **Why use a Pipeline instead of transforming the dataset beforehand?** | Transforming features (like scaling or imputation) on the full dataset before splitting leaks test set statistical distribution (mean and variance) into the training set. A `Pipeline` fits transformers strictly on training folds and only applies `transform()` to test folds. |
| **Why is Recall often preferred over Precision in Churn Prediction?** | A False Negative (missing an actual churner) results in lost recurring customer revenue (high customer acquisition cost). A False Positive (contacting a retained customer with a discount) incurs only minor promotional cost. Maximizing Recall captures the maximum number of vulnerable accounts. |
| **Why not use RFM directly on the Telco Churn dataset?** | RFM analysis requires transaction-level timestamps and individual order purchase values. The Telco dataset is a subscription-level snapshot with aggregate tenure and recurring monthly charges. Customer360 correctly uses K-Means for the subscription dataset and provides a transaction log dataset for RFM. |
| **How is the optimal number of clusters ($k$) determined?** | We combine the **Elbow Method** (identifying the inflection point where Within-Cluster Sum of Squares / Inertia diminishes) with the **Silhouette Score** (measuring how distinct and cohesive clusters are, ideally $> 0.40$). |
| **How does class_weight='balanced' work mathematically?** | It penalizes misclassifications of the minority (churn) class more heavily by weighting loss gradients by $\frac{N}{2 \cdot N_{\text{churn}}}$. |

---

## 🔒 Security & Data Governance
- All user file uploads are strictly parsed as data tables without code evaluation.
- No client-side PII or credentials are exposed or sent to third-party endpoints.
- Model artifacts are cleanly serialized with `joblib` and isolated from production Git branches.

---

## 📜 License
This project is released under the **MIT License**. Created for academic and production machine learning excellence.

