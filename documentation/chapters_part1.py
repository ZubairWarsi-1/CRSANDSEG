"""
Customer360 Documentation Content - Part 1
Chapters 1 through 5: Project Overview, Tech Stack, Architecture, Modules, Dataset.
"""

import os
from reportlab.platypus import Paragraph, Spacer, PageBreak, KeepTogether
from documentation.pdf_components import (
    create_callout, create_styled_table, create_image_flowable
)

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")


def get_chapter_1(styles):
    """Chapter 1: Project Overview"""
    flowables = []
    flowables.append(Paragraph("Chapter 1 &mdash; Project Overview & Business Value", styles['ChapterHeading']))
    flowables.append(Spacer(1, 6))

    flowables.append(Paragraph(
        "<b>Customer360</b> is an enterprise-grade Customer Segmentation and Retention Intelligence Platform designed "
        "to empower commercial enterprises, subscription service providers, and data science teams with end-to-end "
        "analytical intelligence. Modern enterprises face severe financial leakage when customers churn unexpectedly. "
        "Customer360 addresses this challenge by transforming raw transactional and demographic customer data into "
        "rigorous predictive churn scores, unsupervised behavioural clusters, proactive multi-factor retention priority "
        "rankings, and real-time revenue exposure calculations.",
        styles['Body']
    ))

    flowables.append(Paragraph("1.1 The Business Problem", styles['SectionHeading']))
    flowables.append(Paragraph(
        "Across subscription industries (such as telecommunications, SaaS, and financial services) and retail commerce, "
        "acquiring a new customer costs <b>5 to 7 times more</b> than retaining an existing customer. Despite this economic reality, "
        "most organizations operate reactively: they intervene only after a cancellation request is filed or after a customer "
        "ceases transactions entirely. By that point, customer dissatisfaction is entrenched and retention efficacy drops below 15%.",
        styles['Body']
    ))
    flowables.append(Paragraph(
        "Key operational failures addressed by Customer360 include:",
        styles['Body']
    ))
    flowables.append(Paragraph("&bull; <b>Data Silos &amp; Inconsistent Schemas:</b> Disparate transaction logs, contract terms, and customer service records fail to unite into a single analytical profile.", styles['Bullet']))
    flowables.append(Paragraph("&bull; <b>Blind High-Value Churn:</b> Premium customers churn silently because aggregate metrics mask individual dissatisfaction and behavioural decay.", styles['Bullet']))
    flowables.append(Paragraph("&bull; <b>Inefficient Resource Allocation:</b> Retention incentives (discounts, credits) are distributed randomly rather than focused on accounts with high churn risk and high financial value.", styles['Bullet']))
    flowables.append(Paragraph("&bull; <b>Black-Box Machine Learning:</b> Commercial teams distrust predictive models when predictions lack transparent explanations, what-if simulations, or clinical diagnostics.", styles['Bullet']))

    flowables.append(Spacer(1, 4))
    flowables.append(create_callout(
        "<b>Core Economic Thesis:</b> A 5% increase in customer retention rate can produce between 25% and 95% increase "
        "in long-term operating profits, as loyal customers generate higher lifetime value, demand less support overhead, "
        "and present frequent cross-selling opportunities.",
        title="BUSINESS JUSTIFICATION",
        level="success",
        styles=styles
    ))
    flowables.append(Spacer(1, 8))

    flowables.append(Paragraph("1.2 The Customer360 Solution", styles['SectionHeading']))
    flowables.append(Paragraph(
        "Customer360 provides an integrated 360-degree customer intelligence framework combining both <b>Unsupervised "
        "Segmentation</b> (identifying organic behavioural cohorts via K-Means and RFM analysis) and <b>Supervised Machine Learning</b> "
        "(forecasting customer churn probabilities with calibrated classification models). By synthesizing risk, value, and tenure into "
        "a composite <b>Priority Score</b>, the platform directs retention resources where they maximize commercial return.",
        styles['Body']
    ))

    flowables.append(Paragraph("1.3 Primary Stakeholders &amp; User Personas", styles['SectionHeading']))
    headers = ["User Persona", "Primary Objectives", "Key Application Workflows Utilized"]
    rows = [
        [
            "<b>Retention &amp; Success Managers</b>",
            "Identify urgent churn flight risks, review automated action plans, and track customer contact interventions.",
            "Retention Center, Individual Customer Drilldown, SQLite Interaction Notes."
        ],
        [
            "<b>Marketing Campaign Directors</b>",
            "Tailor incentive campaigns and promotional bundles to distinct customer cohorts and behavioural segments.",
            "Customer Segmentation, 2D PCA Cluster Map, RFM Quintile Analysis."
        ],
        [
            "<b>Data Scientists &amp; ML Engineers</b>",
            "Audit data contracts, evaluate classification metrics, benchmark algorithms, and tune decision thresholds.",
            "Data Upload Quality Checks, Model Performance &amp; Diagnostics, Cross-Validation."
        ],
        [
            "<b>Executive Leadership &amp; CFO</b>",
            "Monitor aggregate portfolio health, total Revenue at Risk, and financial ROI of retention programs.",
            "Executive Dashboard, Revenue Exposure Breakdowns, Executive PDF Reports."
        ]
    ]
    flowables.append(create_styled_table(headers, rows, [110, 194, 200], styles=styles))
    flowables.append(Spacer(1, 10))

    img_path = os.path.join(ASSETS_DIR, "modules_flowchart.png")
    flowables.append(create_image_flowable(img_path, "Figure 1.1: Customer360 Platform Modules and User Journey Flowchart", 490, styles=styles))

    return flowables


def get_chapter_2(styles):
    """Chapter 2: Technology Stack"""
    flowables = []
    flowables.append(Paragraph("Chapter 2 &mdash; Technology Stack &amp; Software Ecosystem", styles['ChapterHeading']))
    flowables.append(Spacer(1, 6))

    flowables.append(Paragraph(
        "Customer360 is built exclusively with production-grade, open-source Python enterprise technologies. "
        "The architecture emphasizes deterministic execution, reproducibility, clean separation of concerns, and zero "
        "reliance on brittle third-party cloud SaaS dependencies. The complete software stack is detailed below.",
        styles['Body']
    ))

    flowables.append(Paragraph("2.1 Architectural Technology Matrix", styles['SectionHeading']))
    headers = ["Component / Library", "Version", "Role in Customer360 Architecture", "Verification Standard"]
    rows = [
        ["<b>Python</b>", "3.11.x", "Core programming runtime, type hints, and backend computation engine.", "System runtime standard"],
        ["<b>Streamlit</b>", "&ge; 1.30.0", "Reactive multi-page web application interface and state management.", "7 verified page modules"],
        ["<b>Pandas</b>", "&ge; 2.2.0", "In-memory DataFrame manipulation, cleaning, aggregation, and transformations.", "13-step data pipeline"],
        ["<b>NumPy</b>", "&ge; 1.26.0", "Vectorized linear algebra, numerical array operations, and metric math.", "Sub-second vector computations"],
        ["<b>Scikit-learn</b>", "&ge; 1.4.0", "Machine learning pipelines, 5 classifiers, K-Means clustering, and CV.", "42 unit/integration tests"],
        ["<b>XGBoost</b>", "&ge; 2.0.0", "Extreme gradient boosted decision tree classifier with L1/L2 regularization.", "Integrated into benchmark suite"],
        ["<b>Plotly</b>", "&ge; 5.18.0", "Interactive data visualizations, 2D PCA scatters, ROC curves, and gauges.", "Zero-latency canvas render"],
        ["<b>SQLite3</b>", "3.x Built-in", "Embedded ACID relational database for retention notes, interventions, and audit logs.", "Zero external DB server required"],
        ["<b>OpenPyXL</b>", "&ge; 3.1.0", "Multi-sheet styled Microsoft Excel workbook exporter (.xlsx).", "5 structured worksheets with headers"],
        ["<b>ReportLab</b>", "&ge; 5.1.0", "Two-pass publication-grade PDF dossier generation engine.", "Flowable layout with dynamic headers"]
    ]
    flowables.append(create_styled_table(headers, rows, [95, 55, 234, 120], styles=styles))
    flowables.append(Spacer(1, 10))

    img_path = os.path.join(ASSETS_DIR, "tech_stack_diagram.png")
    flowables.append(create_image_flowable(img_path, "Figure 2.1: Customer360 Technology Stack & Component Interactions", 490, styles=styles))

    flowables.append(Spacer(1, 8))
    flowables.append(Paragraph("2.2 Design Philosophy &amp; Engineering Decisions", styles['SectionHeading']))
    flowables.append(Paragraph(
        "<b>Local Execution with Zero Cloud Overhead:</b> Customer360 is engineered to run locally or containerized on any "
        "standard server without requiring expensive managed cloud data platforms. By embedding SQLite for state persistence "
        "and utilizing Scikit-learn pipelines with Joblib serialization, the entire analytical lifecycle executes within memory.",
        styles['Body']
    ))
    flowables.append(Paragraph(
        "<b>Strict Data Contracts &amp; Defensiveness:</b> All user uploads are subjected to automated type sniffing, "
        "schema verification, and sanity checks. The system strictly avoids broad exception suppression; errors are identified at the "
        "source and reported with corrective recommendations.",
        styles['Body']
    ))

    return flowables


def get_chapter_3(styles):
    """Chapter 3: System Architecture"""
    flowables = []
    flowables.append(Paragraph("Chapter 3 &mdash; System Architecture &amp; Data Engineering", styles['ChapterHeading']))
    flowables.append(Spacer(1, 6))

    flowables.append(Paragraph(
        "Customer360 follows a strict <b>Layered Modular Architecture</b>. This decouples user interface rendering "
        "from analytical computation, persistence, and reporting. Even if the Streamlit presentation layer is adapted "
        "or replaced in future iterations, the underlying machine learning pipelines, preprocessing transformations, "
        "and relational database schemas remain 100% reusable and decoupled.",
        styles['Body']
    ))

    img_path = os.path.join(ASSETS_DIR, "arch_diagram.png")
    flowables.append(create_image_flowable(img_path, "Figure 3.1: Layered System Architecture and Component Topology", 490, styles=styles))
    flowables.append(Spacer(1, 8))

    flowables.append(Paragraph("3.1 Detailed Layer Specifications", styles['SectionHeading']))
    
    flowables.append(Paragraph("<b>1. Presentation Layer (Streamlit Pages):</b>", styles['BodyBold']))
    flowables.append(Paragraph(
        "Comprises <code>app.py</code> and seven dedicated page scripts located in <code>pages/</code>. Each script manages "
        "reactive session state, UI widgets (sliders, file uploaders, selectboxes), and renders Plotly figures. It connects "
        "to the backend exclusively via public functions in the <code>src/</code> package.",
        styles['Body']
    ))

    flowables.append(Paragraph("<b>2. Data Ingestion &amp; Validation Engine (src/validation.py &amp; src/data_loader.py):</b>", styles['BodyBold']))
    flowables.append(Paragraph(
        "Inspects raw CSV and Excel uploads, enforces column contracts, handles encoding anomalies (UTF-8-SIG, ASCII, Latin-1), "
        "computes transparent 0&ndash;100 data quality scores, checks analytical readiness, and detects multicollinear redundancy.",
        styles['Body']
    ))

    flowables.append(Paragraph("<b>3. Machine Learning &amp; Analytics Core (src/churn_prediction.py &amp; src/segmentation.py):</b>", styles['BodyBold']))
    flowables.append(Paragraph(
        "Implements Scikit-learn pipelines with ColumnTransformer. Manages 5 classification algorithms, hyperparameter constraints, "
        "5-fold stratified cross-validation, probability calibration, K-Means clustering (K=2..8 auto-selection via Silhouette/Inertia), "
        "PCA 2D projection, and transactional RFM quintile calculation.",
        styles['Body']
    ))

    flowables.append(Paragraph("<b>4. Retention &amp; Decision Engine (src/recommendations.py):</b>", styles['BodyBold']))
    flowables.append(Paragraph(
        "Executes business logic: computes multi-factor Customer Priority Scores (Risk 50%, Value 35%, Tenure Urgency 15%), "
        "calculates individual and portfolio Revenue at Risk, generates automated tailored retention action plans, and projects ROI.",
        styles['Body']
    ))

    flowables.append(Paragraph("<b>5. Persistence &amp; Storage Layer (src/database.py &amp; data/customer360.db):</b>", styles['BodyBold']))
    flowables.append(Paragraph(
        "SQLite relational database maintaining 6 structured tables: <code>customer_interventions</code>, "
        "<code>contact_history</code>, <code>retention_notes</code>, <code>retention_outcomes</code>, "
        "<code>upload_history</code>, and <code>report_history</code>. Guarantees permanent audit trails across sessions.",
        styles['Body']
    ))

    flowables.append(Spacer(1, 4))
    flowables.append(create_callout(
        "<b>Architectural Separation:</b> Raw customer datasets are treated as immutable inputs. Transformations are executed "
        "in-memory or written to cached session objects, preventing data corruption and ensuring reproducible pipelines.",
        title="ARCHITECTURAL INVARIANT",
        level="note",
        styles=styles
    ))

    return flowables


def get_chapter_4(styles):
    """Chapter 4: Complete Project Modules"""
    flowables = []
    flowables.append(Paragraph("Chapter 4 &mdash; Comprehensive Application Modules (All 7 Pages)", styles['ChapterHeading']))
    flowables.append(Spacer(1, 6))

    flowables.append(Paragraph(
        "The Customer360 application interface is organized into seven specialized, interconnected modules accessible "
        "via the Streamlit sidebar. Every module provides dedicated functionality, business calculations, and visual diagnostics.",
        styles['Body']
    ))

    modules = [
        ("Module 1: Executive Dashboard (1_Dashboard.py)", [
            ("Purpose &amp; Objective", "Provides executive leadership and commercial managers with a high-level real-time portfolio snapshot of customer health, risk distribution, and revenue exposure."),
            ("Primary Input Data", "Active validated customer dataset merged with precomputed or batch churn probabilities and segment labels."),
            ("Calculated KPIs", "Total Customers, Active/Retained Count, Average Monthly Spend, Historical Churn Rate, High-Risk Customer Count, Expected Monthly Revenue at Risk ($)."),
            ("Key Visualizations", "Churn Risk Distribution Donut (Low, Medium, High), Revenue Exposure by Contract Bar Chart, Payment Method Churn Breakdown, High-Risk Customer Table."),
            ("Operational Workflow", "Managers inspect aggregate risk, review monthly revenue exposure, and filter by contract type (Month-to-month vs One/Two year) to prioritize campaign focus.")
        ]),
        ("Module 2: Customer Segmentation (2_Segmentation.py)", [
            ("Purpose &amp; Objective", "Discovers hidden customer behavioural cohorts using unsupervised K-Means clustering and transactional RFM quintile segmentation."),
            ("Primary Input Data", "Normalized numerical and categorical customer attributes or transactional ledger records (TransactionID, CustomerID, Date, Amount)."),
            ("Analytical Logic", "Automatic K selection (K=2 to 8) evaluated via Silhouette Coefficient and Inertia elbow; PCA 2D projection for spatial cluster visualization; RFM 1-5 scoring."),
            ("Key Visualizations", "2D PCA Cluster Scatter Plot, Silhouette vs K Line Chart, Cluster Persona Profile Tables (comparing cluster mean tenure, monthly charges, and churn rate)."),
            ("Operational Workflow", "Marketing teams select optimal K, inspect cluster personas (e.g. 'At-Risk High Spenders', 'Budget Loyalists'), and export targeted segment rosters for campaigns.")
        ]),
        ("Module 3: Churn Prediction (3_Churn_Prediction.py)", [
            ("Purpose &amp; Objective", "Forecasts individual customer churn probabilities using trained supervised machine learning pipelines, with drilldown diagnostics and counterfactual simulation."),
            ("Primary Input Data", "Customer attributes; pre-trained or user-trained classification pipeline (Logistic Regression, Random Forest, Gradient Boosting, XGBoost)."),
            ("Analytical Logic", "Estimates calibrated churn probability P(Churn) in [0, 1]; calculates individual feature contributions; executes counterfactual What-If simulations."),
            ("Key Visualizations", "Churn Probability Gauge, Feature Importance Bar Chart, What-If Sensitivity Sliders (modifying contract term, monthly charges, tech support)."),
            ("Operational Workflow", "Account managers search a specific customer ID, examine risk drivers, and adjust hypothetical service settings to see how switching contract drops churn risk.")
        ]),
        ("Module 4: Retention Center (4_Retention_Center.py)", [
            ("Purpose &amp; Objective", "Operational command center for executing retention interventions, calculating ROI, generating personalized action plans, and logging CRM history."),
            ("Primary Input Data", "Predicted churn probabilities, monthly charges, tenure, and customer contact records from SQLite database."),
            ("Analytical Logic", "Multi-factor Priority Score (0-100) = (Risk Points x 0.50) + (Value Points x 0.35) + (Tenure Urgency x 0.15); Financial ROI = Net Revenue Saved / Intervention Cost."),
            ("Key Visualizations", "Customer Priority Leaderboard, 4-Quadrant Retention Strategy Matrix, Retention Budget vs ROI Simulator, SQLite Contact History Table."),
            ("Operational Workflow", "Retention agents review prioritized accounts, inspect automated recommendation playbooks, assign team members, and record contact notes with one-click persistence.")
        ]),
        ("Module 5: Data Upload &amp; Cleaning (5_Data_Upload.py)", [
            ("Purpose &amp; Objective", "Robust data intake portal supporting file uploads (CSV, Excel), automated schema sniffing, 0-100 quality scoring, and one-click 13-step cleaning."),
            ("Primary Input Data", "Raw user-uploaded customer files (Telco subscription or Retail transactional formats) or one-click synthetic benchmark generator."),
            ("Analytical Logic", "Header normalization, whitespace stripping, TotalCharges coercion, median/mode imputation, IQR outlier detection, feature redundancy checks (correlation > 0.95)."),
            ("Key Visualizations", "Data Quality Score Gauge (0-100), Missing Values Heatmap / Bar Chart, Pre vs Post Cleaning Comparison Table, SQLite Ingestion History Table."),
            ("Operational Workflow", "Users upload a dirty CSV, immediately review data health scores, click 'Clean & Prepare Dataset', verify cleaned output, and push to active session.")
        ]),
        ("Module 6: Model Performance (6_Model_Performance.py)", [
            ("Purpose &amp; Objective", "Rigorous evaluation and diagnostic laboratory comparing 5 classification algorithms on held-out test data with zero data leakage."),
            ("Primary Input Data", "80/20 train/test split of active customer dataset evaluated across 5-fold stratified cross-validation."),
            ("Analytical Logic", "Computes Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC (with prevalence baseline), Brier Score calibration, and decision threshold sweeps (0.20 to 0.80)."),
            ("Key Visualizations", "ROC Curves with 0.50 Baseline, Precision-Recall Curves with Prevalence Baseline, Confusion Matrix Heatmap, Threshold Tuning Curve, Generalization Gap Chart."),
            ("Operational Workflow", "Data scientists compare benchmark tables, evaluate trade-offs between precision and recall, select the optimal decision threshold, and persist the champion pipeline.")
        ]),
        ("Module 7: Reports &amp; Exports (7_Reports.py)", [
            ("Purpose &amp; Objective", "Multi-channel export center producing 6 standardized CSV deliverables (UTF-8-SIG), styled multi-sheet Excel workbooks, and executive PDF dossiers."),
            ("Primary Input Data", "Active predictions, segment assignments, retention plans, data quality metrics, and model evaluation tables."),
            ("Analytical Logic", "Generates Excel workbooks with custom header palettes, freeze panes, and auto-adjusted column widths; compiles ReportLab PDF dossiers; logs exports to SQLite."),
            ("Key Visualizations", "Export Readiness Checklists, Report Download Cards, SQLite Report Generation History Log."),
            ("Operational Workflow", "Users configure export parameters, generate multi-tab Excel spreadsheets or executive PDF summaries, and download client-ready deliverables with a single click.")
        ])
    ]

    for title, details in modules:
        flowables.append(Paragraph(f"<b>{title}</b>", styles['SectionHeading']))
        rows = []
        for d_title, d_desc in details:
            rows.append([Paragraph(f"<b>{d_title}</b>", styles['TableCellBold']), Paragraph(d_desc, styles['TableCell'])])
        t = create_styled_table(["Attribute", "Module Specification"], rows, [130, 374], styles=styles)
        flowables.append(t)
        flowables.append(Spacer(1, 8))

    return flowables


def get_chapter_5(styles):
    """Chapter 5: Dataset & Data Requirements"""
    flowables = []
    flowables.append(Paragraph("Chapter 5 &mdash; Dataset Specifications &amp; Data Contracts", styles['ChapterHeading']))
    flowables.append(Spacer(1, 6))

    flowables.append(Paragraph(
        "Customer360 natively supports two distinct industry data archetypes: <b>Subscription Customer Profiles</b> "
        "(e.g. Telco, SaaS, Utilities) and <b>Transactional Ledgers</b> (e.g. E-Commerce, Retail). The platform includes "
        "verified real-world benchmark datasets for both formats located in the <code>data/</code> directory.",
        styles['Body']
    ))

    flowables.append(Paragraph("5.1 Telco Subscription Benchmark Dataset (data/telco_customer_churn.csv)", styles['SectionHeading']))
    flowables.append(Paragraph(
        "The Telco dataset contains <b>7,043 customer accounts</b> across <b>21 distinct demographic, contract, and usage features</b>. "
        "The overall historical churn prevalence is <b>26.54% (1,869 churned customers)</b>, representing a standard realistic "
        "class imbalance scenario. The complete feature dictionary is defined below.",
        styles['Body']
    ))

    headers = ["Feature Name", "Data Type", "Role", "Allowed / Observed Values", "Business Significance"]
    rows = [
        ["<code>customerID</code>", "String / ID", "Primary Key", "7,043 unique alphanumeric strings", "Account identifier; excluded from ML features."],
        ["<code>gender</code>", "Categorical", "Demographic", "Male, Female", "Gender parity tracking."],
        ["<code>SeniorCitizen</code>", "Binary Int", "Demographic", "0 (No), 1 (Yes)", "Senior demographic vulnerability indicator."],
        ["<code>Partner</code>", "Categorical", "Demographic", "Yes, No", "Household stability factor."],
        ["<code>Dependents</code>", "Categorical", "Demographic", "Yes, No", "Family account indicator."],
        ["<code>tenure</code>", "Integer", "Usage / Value", "1 to 72 months", "Length of customer relationship; crucial churn driver."],
        ["<code>PhoneService</code>", "Categorical", "Service", "Yes, No", "Basic voice connectivity."],
        ["<code>MultipleLines</code>", "Categorical", "Service", "No, Yes, No phone service", "Multi-line voice expansion."],
        ["<code>InternetService</code>", "Categorical", "Core Service", "DSL, Fiber optic, No", "High-churn driver (Fiber optic churn > 40%)."],
        ["<code>OnlineSecurity</code>", "Categorical", "Value-Add", "Yes, No, No internet service", "Retention protector (accounts with security churn less)."],
        ["<code>OnlineBackup</code>", "Categorical", "Value-Add", "Yes, No, No internet service", "Cloud backup attachment."],
        ["<code>DeviceProtection</code>", "Categorical", "Value-Add", "Yes, No, No internet service", "Hardware warranty protection."],
        ["<code>TechSupport</code>", "Categorical", "Value-Add", "Yes, No, No internet service", "High retention protector."],
        ["<code>StreamingTV</code>", "Categorical", "Entertainment", "Yes, No, No internet service", "Digital entertainment consumption."],
        ["<code>StreamingMovies</code>", "Categorical", "Entertainment", "Yes, No, No internet service", "Premium video consumption."],
        ["<code>Contract</code>", "Categorical", "Commitment", "Month-to-month, One year, Two year", "Single strongest churn predictor."],
        ["<code>PaperlessBilling</code>", "Categorical", "Billing", "Yes, No", "Digital billing preference."],
        ["<code>PaymentMethod</code>", "Categorical", "Billing", "Electronic check, Mailed check, Bank transfer, Credit card", "Electronic check exhibits highest churn rate."],
        ["<code>MonthlyCharges</code>", "Float", "Financial", "$18.25 to $118.75", "Monthly billing amount; base for revenue exposure."],
        ["<code>TotalCharges</code>", "Float", "Financial", "$18.80 to $8,684.80 (11 whitespace blanks)", "Cumulative spend; requires whitespace repair."],
        ["<code>Churn</code>", "Categorical / Target", "Supervised Target", "Yes (1,869), No (5,174)", "Ground-truth churn indicator."]
    ]
    flowables.append(create_styled_table(headers, rows, [85, 60, 65, 144, 150], styles=styles))
    flowables.append(Spacer(1, 10))

    flowables.append(Paragraph("5.2 Transactional Retail Benchmark Dataset (data/retail_transactions.csv)", styles['SectionHeading']))
    flowables.append(Paragraph(
        "For transactional commerce applications, Customer360 includes <code>data/retail_transactions.csv</code> comprising "
        "<b>3,000 transaction records</b> across 6 normalized fields: <code>TransactionID</code>, <code>CustomerID</code>, "
        "<code>TransactionDate</code>, <code>Quantity</code>, <code>UnitPrice</code>, and <code>TotalAmount</code>. "
        "The application automatically transforms this transactional ledger into RFM metrics (Recency in days, Frequency in count of orders, "
        "and Monetary value in cumulative spend) per customer.",
        styles['Body']
    ))

    flowables.append(Spacer(1, 4))
    flowables.append(create_callout(
        "<b>Data Quality Trap &mdash; TotalCharges Whitespace Bug:</b> In the raw Telco dataset, 11 records have tenure = 0 "
        "and contain a blank whitespace string <code>' '</code> in <code>TotalCharges</code>. Ordinary CSV parsers fail to "
        "identify this as NaN, causing downstream numerical math to throw crashes. Customer360's cleaning pipeline automatically "
        "coerces these strings to NaN and imputes them with the column median ($1,397.47).",
        title="KNOWN DATA ANOMALY & FIX",
        level="warning",
        styles=styles
    ))

    return flowables
