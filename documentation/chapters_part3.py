"""
Customer360 Documentation Content - Part 3
Chapters 11 through 15: Reports & Exports, Developer Setup, Troubleshooting, Roadmap, Conclusion & Glossary.
"""

import os
from reportlab.lib import colors
from reportlab.platypus import Paragraph, Spacer, PageBreak, KeepTogether, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle
from documentation.pdf_components import (
    create_callout, create_styled_table, create_image_flowable, NAVY
)

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")


def get_chapter_11(styles):
    """Chapter 11: Reports & Export Specifications"""
    flowables = []
    flowables.append(Paragraph("Chapter 11 &mdash; Reports &amp; Export Specifications", styles['ChapterHeading']))
    flowables.append(Spacer(1, 6))

    flowables.append(Paragraph(
        "Customer360 provides enterprise reporting capabilities implemented in <code>src/reporting.py</code>. "
        "The system exports structured data in three industry-standard formats: <b>Modular CSV Files</b>, "
        "<b>Multi-Sheet Microsoft Excel Workbooks</b>, and <b>Executive PDF Dossiers</b>. All exports are permanently "
        "logged in the SQLite database.",
        styles['Body']
    ))

    flowables.append(Paragraph("11.1 The Six Standardized Modular CSV Deliverables", styles['SectionHeading']))
    flowables.append(Paragraph(
        "All CSV files are encoded in <code>UTF-8-SIG</code> (UTF-8 with Byte Order Mark) to ensure seamless compatibility "
        "with Microsoft Excel across Windows and macOS without character corruption.",
        styles['Body']
    ))

    headers = ["Deliverable Filename Pattern", "Key Included Schema / Columns", "Primary Purpose"]
    rows = [
        [
            "<code>churn_predictions_*.csv</code>",
            "<code>customerID, MonthlyCharges, tenure, churn_probability, predicted_churn, risk_category</code>",
            "Full portfolio batch churn scoring output for downstream BI ingestion."
        ],
        [
            "<code>high_risk_customers_*.csv</code>",
            "<code>customerID, MonthlyCharges, tenure, churn_probability, priority_score, contract</code>",
            "Filtered high-risk accounts (P(Churn) &ge; 0.65) sorted by Priority Score."
        ],
        [
            "<code>customer_segments_*.csv</code>",
            "<code>customerID, cluster_id, cluster_name, PC1, PC2, tenure, MonthlyCharges</code>",
            "Unsupervised clustering assignments, PCA coordinates, and segment labels."
        ],
        [
            "<code>retention_action_plan_*.csv</code>",
            "<code>customerID, priority_score, churn_probability, primary_risk_factor, recommended_action, incentive_offer</code>",
            "Actionable playbooks for retention agents and automated email marketing systems."
        ],
        [
            "<code>model_evaluation_summary_*.csv</code>",
            "<code>model_name, test_accuracy, precision, recall, f1_score, roc_auc, pr_auc, brier_score</code>",
            "Technical diagnostic summary of hold-out benchmark metrics for governance."
        ],
        [
            "<code>data_quality_report_*.csv</code>",
            "<code>total_records, total_features, missing_pct, duplicate_count, quality_score, quality_grade</code>",
            "Data contract audit log certifying dataset hygiene prior to modeling."
        ]
    ]
    flowables.append(create_styled_table(headers, rows, [140, 204, 160], styles=styles))
    flowables.append(Spacer(1, 10))

    flowables.append(Paragraph("11.2 Multi-Sheet Microsoft Excel Workbook Architecture (.xlsx)", styles['SectionHeading']))
    flowables.append(Paragraph(
        "The Excel export generates a workbook styled with OpenPyXL containing <b>5 structured worksheets</b>:",
        styles['Body']
    ))
    flowables.append(Paragraph("&bull; <b>1. Executive Summary:</b> Key portfolio KPIs, total revenue at risk, churn breakdown, and generation timestamp.", styles['Bullet']))
    flowables.append(Paragraph("&bull; <b>2. Churn Predictions:</b> Full customer roster with calibrated probabilities, risk tiers, and charges.", styles['Bullet']))
    flowables.append(Paragraph("&bull; <b>3. Customer Segments:</b> Cluster assignments, RFM quintile scores, and persona definitions.", styles['Bullet']))
    flowables.append(Paragraph("&bull; <b>4. High-Risk Action Plan:</b> Top 100 accounts requiring urgent intervention with recommended playbooks.", styles['Bullet']))
    flowables.append(Paragraph("&bull; <b>5. Data Quality Audit:</b> Column-by-column missingness, data types, and quality scoring audit trail.", styles['Bullet']))

    flowables.append(Spacer(1, 4))
    flowables.append(Paragraph("11.3 SQLite Report Generation History Audit", styles['SectionHeading']))
    flowables.append(Paragraph(
        "Every generated deliverable triggers an automatic audit record in the <code>report_history</code> SQLite table, "
        "recording the report name, file type, row count, target destination, and timestamp. This ensures full compliance "
        "and governance tracking.",
        styles['Body']
    ))

    return flowables


def get_chapter_12(styles):
    """Chapter 12: Installation, Configuration & Developer Guide"""
    flowables = []
    flowables.append(Paragraph("Chapter 12 &mdash; Installation, Configuration &amp; Developer Setup Guide", styles['ChapterHeading']))
    flowables.append(Spacer(1, 6))

    flowables.append(Paragraph(
        "This chapter provides explicit technical instructions for software engineers, DevOps specialists, and data scientists "
        "setting up, running, and developing Customer360 in local or containerized environments.",
        styles['Body']
    ))

    flowables.append(Paragraph("12.1 System Prerequisites", styles['SectionHeading']))
    headers = ["Component", "Minimum Requirement", "Recommended Production Specification"]
    rows = [
        ["Operating System", "Windows 10 / 11, Ubuntu 20.04+, macOS 12+", "Windows 11 / Ubuntu 22.04 LTS"],
        ["Python Runtime", "Python 3.10.x", "Python 3.11.x (verified 3.11.9)"],
        ["System Memory (RAM)", "4 GB RAM", "8 GB to 16 GB RAM"],
        ["Disk Storage", "500 MB free space", "2 GB free space (including models & database)"],
        ["Shell Environment", "PowerShell (Windows) or Bash (Linux/macOS)", "PowerShell 7.x / Bash"]
    ]
    flowables.append(create_styled_table(headers, rows, [120, 184, 200], styles=styles))
    flowables.append(Spacer(1, 8))

    flowables.append(Paragraph("12.2 Step-by-Step Installation Commands", styles['SectionHeading']))
    flowables.append(Paragraph("<b>Step 1: Clone the Repository &amp; Navigate to Directory:</b>", styles['BodyBold']))
    flowables.append(Paragraph("<code>git clone https://github.com/ZubairWarsi-1/CRSANDSEG.git<br/>cd \"Customer Segmentation and Retention System\"</code>", styles['CodeBlock']))
    flowables.append(Spacer(1, 4))

    flowables.append(Paragraph("<b>Step 2: Create and Activate a Dedicated Virtual Environment:</b>", styles['BodyBold']))
    flowables.append(Paragraph("<code># Windows (PowerShell):<br/>python -m venv venv<br/>.\\venv\\Scripts\\Activate.ps1<br/><br/># Linux / macOS (Bash):<br/>python3 -m venv venv<br/>source venv/bin/activate</code>", styles['CodeBlock']))
    flowables.append(Spacer(1, 4))

    flowables.append(Paragraph("<b>Step 3: Install Required Dependencies:</b>", styles['BodyBold']))
    flowables.append(Paragraph("<code>pip install --upgrade pip<br/>pip install -r requirements.txt</code>", styles['CodeBlock']))
    flowables.append(Spacer(1, 4))

    flowables.append(Paragraph("<b>Step 4: Execute the Automated Pytest Verification Suite:</b>", styles['BodyBold']))
    flowables.append(Paragraph("<code>pytest tests/ -v</code>", styles['CodeBlock']))
    flowables.append(Paragraph("<i>Expected Result: 42 passed in ~45 seconds (100% pass rate).</i>", styles['Body']))
    flowables.append(Spacer(1, 4))

    flowables.append(Paragraph("<b>Step 5: Launch the Streamlit Web Application:</b>", styles['BodyBold']))
    flowables.append(Paragraph("<code>streamlit run app.py</code>", styles['CodeBlock']))
    flowables.append(Paragraph("<i>The application will launch automatically at http://localhost:8501.</i>", styles['Body']))

    flowables.append(Spacer(1, 4))
    flowables.append(create_callout(
        "<b>PowerShell Execution Policy on Windows:</b> If you encounter an execution policy error when activating the "
        "virtual environment (<code>Scripts\\Activate.ps1 cannot be loaded</code>), run:<br/>"
        "<code>Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass</code> in your PowerShell session.",
        title="WINDOWS SETUP TIP",
        level="warning",
        styles=styles
    ))

    return flowables


def get_chapter_13(styles):
    """Chapter 13: Troubleshooting & Frequently Asked Questions"""
    flowables = []
    flowables.append(Paragraph("Chapter 13 &mdash; Troubleshooting Guide &amp; Operational FAQ", styles['ChapterHeading']))
    flowables.append(Spacer(1, 6))

    flowables.append(Paragraph(
        "This section documents solutions to common technical issues, data validation errors, and operational questions "
        "encountered during development or user deployment.",
        styles['Body']
    ))

    flowables.append(Paragraph("13.1 Technical Diagnostic &amp; Resolution Guide", styles['SectionHeading']))
    headers = ["Observed Issue / Error", "Root Cause Analysis", "Verified Resolution &amp; Fix"]
    rows = [
        [
            "<code>KeyError: 'missing_pct'</code> during Data Upload",
            "Inspection dictionary returned inconsistent schema keys (missing_pct vs missing_percentage).",
            "Use standardized <code>inspect_dataset()</code> returning both keys. Wrap all UI dictionary lookups in safe <code>.get()</code> accessors with defaults."
        ],
        [
            "<code>AttributeError: 'list' object has no attribute 'empty'</code>",
            "Database query returned Python <code>list</code> of dicts, but Streamlit code checked <code>.empty</code> (a Pandas DataFrame attribute).",
            "Implemented <code>DbResultList(list)</code> wrapper in <code>src/database.py</code> supporting <code>.empty</code>, <code>.columns</code>, and <code>.to_dataframe()</code>."
        ],
        [
            "TotalCharges contains non-numeric strings / space crashes",
            "Raw Telco dataset has 11 records with whitespace <code>' '</code> for tenure=0 customers.",
            "Handled automatically in <code>src/data_loader.py</code> via <code>clean_whitespace_numeric=True</code>, coercing blanks to NaN and imputing with median."
        ],
        [
            "<code>sqlite3.OperationalError: database is locked</code>",
            "Concurrent threads or open transactions locking SQLite database file.",
            "All DB connections in <code>src/database.py</code> use <code>with sqlite3.connect(...) as conn:</code> with a 15-second busy timeout."
        ],
        [
            "Plotly charts not rendering in browser",
            "Browser WebGL acceleration disabled or ad-blocker interfering with CDN scripts.",
            "Enable hardware acceleration in Chrome/Edge, or use default SVG rendering fallback configured in <code>src/ui_components.py</code>."
        ]
    ]
    flowables.append(create_styled_table(headers, rows, [130, 174, 200], styles=styles))
    flowables.append(Spacer(1, 8))

    flowables.append(Paragraph("13.2 Frequently Asked Questions (FAQ)", styles['SectionHeading']))
    faqs = [
        ("Q: What does a 70% churn risk score actually mean for an account?", 
         "A: A 70% churn probability means that based on current contract terms, support usage, and payment history, this customer's profile matches historical patterns where 70 out of 100 accounts cancelled their service within the subsequent billing period. It demands immediate proactive contact."),
        ("Q: How often should the machine learning models be retrained?", 
         "A: For commercial subscription environments, models should be retrained on a monthly basis, or whenever a major pricing change, marketing promotion, or competitive market shift occurs to prevent concept drift."),
        ("Q: Can I use this application for non-telecom datasets?", 
         "A: Yes! Customer360 includes generic column role inference. You can map any customer dataset containing an ID, tenure/lifetime, billing/spend, and a binary churn target, or upload raw transaction ledgers for automatic RFM segmentation."),
        ("Q: Where is customer intervention data stored? Will it be lost on restart?", 
         "A: All customer interventions, agent notes, upload records, and report logs are saved permanently in the embedded SQLite relational database at <code>data/customer360.db</code>. It persists permanently across application restarts.")
    ]
    for q, a in faqs:
        flowables.append(Paragraph(f"<b>{q}</b>", styles['SubSectionHeading']))
        flowables.append(Paragraph(a, styles['Body']))

    return flowables


def get_chapter_14(styles):
    """Chapter 14: Limitations & Future Enhancements"""
    flowables = []
    flowables.append(Paragraph("Chapter 14 &mdash; System Limitations &amp; Future Engineering Roadmap", styles['ChapterHeading']))
    flowables.append(Spacer(1, 6))

    flowables.append(Paragraph(
        "To ensure academic and professional integrity, this chapter documents the current boundaries of the system "
        "and details the strategic engineering roadmap for future development iterations.",
        styles['Body']
    ))

    flowables.append(Paragraph("14.1 Documented System Boundaries &amp; Current Limitations", styles['SectionHeading']))
    flowables.append(Paragraph(
        "&bull; <b>Cross-Sectional vs Longitudinal Nature:</b> The Telco benchmark dataset is a cross-sectional snapshot rather than a time-series event log. While tenure serves as a temporal proxy, survival analysis (Cox Proportional Hazards) would provide richer hazard rates over continuous time.",
        styles['Bullet']
    ))
    flowables.append(Paragraph(
        "&bull; <b>Single-Node SQLite Architecture:</b> SQLite is optimal for single-server or local enterprise deployments up to hundreds of thousands of customer records. For distributed organizations with hundreds of concurrent retention agents, migration to PostgreSQL is recommended.",
        styles['Bullet']
    ))
    flowables.append(Paragraph(
        "&bull; <b>Synchronous Ingestion Execution:</b> Uploading multi-gigabyte files (> 1 million records) can cause brief UI latency in Streamlit. Asynchronous worker queues (e.g. Celery / Redis) would decouple processing from UI rendering.",
        styles['Bullet']
    ))

    flowables.append(Spacer(1, 4))
    flowables.append(Paragraph("14.2 Future Strategic Engineering Roadmap", styles['SectionHeading']))
    headers = ["Milestone", "Target Capability", "Technical Architecture &amp; Implementation"]
    rows = [
        ["<b>Phase 1: Real-Time Webhooks</b>", "Automated CRM event streaming", "FastAPI ingestion gateway receiving real-time cancellation signals from Stripe, HubSpot, or Salesforce."],
        ["<b>Phase 2: Deep Sequential Learning</b>", "Temporal event stream modeling", "Recurrent Neural Networks (LSTM) or Transformer architectures processing clickstream and support ticket sequences."],
        ["<b>Phase 3: Automated Retraining Pipeline</b>", "Continuous MLOps &amp; Drift Detection", "Automated Kolmogorov-Smirnov statistical drift tests triggering model re-fitting when data distributions shift."],
        ["<b>Phase 4: Customer Lifetime Value (CLV)</b>", "Joint Churn &amp; Revenue Forecasting", "Two-stage hurdle regression estimating both churn hazard and expected residual spending over a 36-month horizon."]
    ]
    flowables.append(create_styled_table(headers, rows, [110, 160, 234], styles=styles))

    return flowables


def get_chapter_15(styles):
    """Chapter 15: Conclusion, Project Outcomes & Team Credits"""
    flowables = []
    flowables.append(Paragraph("Chapter 15 &mdash; Conclusion, Project Outcomes &amp; Academic Credits", styles['ChapterHeading']))
    flowables.append(Spacer(1, 6))

    flowables.append(Paragraph(
        "<b>Customer360 &mdash; Customer Segmentation and Retention Intelligence Platform</b> demonstrates the successful "
        "synthesis of rigorous machine learning theory, defensive software engineering, and actionable commercial decision science. "
        "By uniting 13-step data preprocessing, 5 supervised classifiers, unsupervised K-Means and RFM clustering, multi-factor "
        "priority scoring, and multi-channel reporting into an intuitive Streamlit interface, the platform delivers verifiable "
        "commercial value and academic excellence.",
        styles['Body']
    ))

    flowables.append(Paragraph("15.1 Summary of Delivered System Capabilities", styles['SectionHeading']))
    flowables.append(Paragraph("&bull; <b>Production-Grade Codebase:</b> Fully verified, robust Python architecture with 42 automated Pytest unit and integration tests passing at 100%.", styles['Bullet']))
    flowables.append(Paragraph("&bull; <b>Empirical Benchmarks:</b> Verified Gradient Boosting champion model achieving 80.62% accuracy, 0.8492 ROC-AUC, and 0.6618 PR-AUC on held-out test data.", styles['Bullet']))
    flowables.append(Paragraph("&bull; <b>Commercial Impact:</b> Rigorous Expected Monthly Revenue Exposure calculations ($121,480/month on Telco benchmark) directing retention investments effectively.", styles['Bullet']))
    flowables.append(Paragraph("&bull; <b>Multi-Channel Exports:</b> Seamless delivery of 6 standardized CSV files, 5-sheet styled Excel workbooks, and publication-quality PDF dossiers.", styles['Bullet']))

    flowables.append(Spacer(1, 10))

    # Prominent Project Contributors Card ("Project Made By")
    flowables.append(Paragraph("15.2 Project Team &amp; Academic Contributors", styles['SectionHeading']))
    
    t_head = Paragraph("<b>PROJECT MADE BY</b>", ParagraphStyle('TH', fontName='Helvetica-Bold', fontSize=10, textColor=NAVY, alignment=1))
    contrib_table = Table([
        [t_head, "", ""],
        [
            Paragraph(
                "<b>Zubair Warsi</b><br/>"
                "<b>Roll No:</b> 2303410100057<br/>"
                "<font color='#0284C7' size='7.5'>Lead Software Architecture &amp; System Engineering</font>",
                styles['Body']
            ),
            Paragraph(
                "<b>Sameer Mansoori</b><br/>"
                "<b>Roll No:</b> 2303410100040<br/>"
                "<font color='#0284C7' size='7.5'>Machine Learning Models &amp; Diagnostic Pipelines</font>",
                styles['Body']
            ),
            Paragraph(
                "<b>Mohd Shahbaaz Khan</b><br/>"
                "<b>Roll No:</b> 2303410100029<br/>"
                "<font color='#0284C7' size='7.5'>Data Preprocessing &amp; UI Experience Design</font>",
                styles['Body']
            )
        ]
    ], colWidths=[168, 168, 168])
    contrib_table.setStyle(TableStyle([
        ('SPAN', (0, 0), (2, 0)),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#EFF6FF")),
        ('BACKGROUND', (0, 1), (-1, 1), colors.white),
        ('BOX', (0, 0), (-1, -1), 1.2, colors.HexColor("#2563EB")),
        ('INNERGRID', (0, 1), (-1, 1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('ALIGN', (0, 1), (-1, 1), 'CENTER'),
    ]))
    flowables.append(contrib_table)

    return flowables


def get_appendix_glossary(styles):
    """Glossary of Technical Terms & Mathematical Formulas"""
    flowables = []
    flowables.append(Paragraph("Appendix &mdash; Mathematical Glossary &amp; Formulas", styles['ChapterHeading']))
    flowables.append(Spacer(1, 6))

    flowables.append(Paragraph(
        "This appendix provides formal mathematical definitions and explanations for all algorithmic concepts "
        "and evaluation metrics implemented in Customer360.",
        styles['Body']
    ))

    terms = [
        ("Accuracy", "Accuracy = (TP + TN) / (TP + TN + FP + FN)", "Proportion of total correct predictions. Can be misleading in imbalanced datasets."),
        ("Precision (PPV)", "Precision = TP / (TP + FP)", "Fraction of predicted churners who actually churned. Crucial when retention incentives have non-trivial unit costs."),
        ("Recall (Sensitivity)", "Recall = TP / (TP + FN)", "Fraction of actual churners correctly captured by the model. Crucial when customer loss cost is high."),
        ("F1-Score", "F1 = 2 &times; (Precision &times; Recall) / (Precision + Recall)", "Harmonic mean of precision and recall; balanced indicator for imbalanced classification."),
        ("ROC-AUC", "Area Under Receiver Operating Characteristic Curve", "Measures classification discrimination across all possible decision thresholds (0.0 to 1.0). Random guess = 0.50."),
        ("PR-AUC", "Area Under Precision-Recall Curve", "Area under the precision vs recall curve. Essential for imbalanced classification. Baseline = positive class prevalence (0.265)."),
        ("Brier Score", "Brier Score = (1/N) &Sigma; (p_i - y_i)&sup2;", "Mean squared error of calibrated probability estimates. Measures reliability of probability values (0 is perfect)."),
        ("K-Means Inertia", "Inertia = &Sigma;<sub>k=1..K</sub> &Sigma;<sub>x &isin; C_k</sub> || x - &mu;_k ||&sup2;", "Sum of squared Euclidean distances of samples to their assigned cluster centroid. Minimized during cluster fitting."),
        ("Silhouette Coefficient", "s(i) = (b(i) - a(i)) / max(a(i), b(i))", "Measures cluster cohesion vs separation (-1.0 to +1.0). High score indicates well-separated, dense clusters."),
        ("Revenue at Risk", "Exposure = &Sigma; (MonthlyCharges_i &times; P(Churn)_i)", "Risk-adjusted probability-weighted recurring revenue exposure. Differentiates expected loss from actual loss."),
        ("Priority Score", "Score = 0.50(Risk) + 0.35(Value) + 0.15(Tenure)", "Multi-factor index ranking accounts from 0 to 100 to maximize commercial ROI on proactive retention.")
    ]

    headers = ["Term / Metric", "Mathematical Formulation", "Commercial / Technical Significance"]
    rows = []
    for t_name, t_form, t_sig in terms:
        rows.append([Paragraph(f"<b>{t_name}</b>", styles['TableCellBold']), Paragraph(f"<code>{t_form}</code>", styles['TableCell']), Paragraph(t_sig, styles['TableCell'])])

    flowables.append(create_styled_table(headers, rows, [110, 180, 214], styles=styles))
    return flowables
