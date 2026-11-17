"""
Master Documentation PDF Builder for Customer360.
Compiles all 15 chapters, assets, tables, and credits into 3 production PDFs:
1. Customer_360_Complete_Project_Documentation.pdf
2. Customer_360_User_Manual.pdf
3. Customer_360_Technical_Documentation.pdf
"""

import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, PageBreak, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors

# Ensure documentation package is importable
DOC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(DOC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from documentation.pdf_components import (
    NumberedCanvas, get_document_styles, create_cover_page_flowables,
    create_styled_table, NAVY, BLUE, LIGHT_BLUE, BORDER_COLOR, WHITE, BG_LIGHT
)
from documentation.chapters_part1 import (
    get_chapter_1, get_chapter_2, get_chapter_3, get_chapter_4, get_chapter_5
)
from documentation.chapters_part2 import (
    get_chapter_6, get_chapter_7, get_chapter_8, get_chapter_9, get_chapter_10
)
from documentation.chapters_part3 import (
    get_chapter_11, get_chapter_12, get_chapter_13, get_chapter_14, get_chapter_15,
    get_appendix_glossary
)


def create_toc_flowables(toc_items, styles):
    """Generates a clean Table of Contents overview table."""
    flowables = []
    flowables.append(Paragraph("Table of Contents &amp; Executive Summary", styles['ChapterHeading']))
    flowables.append(Spacer(1, 6))
    flowables.append(Paragraph(
        "This document is organized into structured chapters designed to provide comprehensive coverage "
        "of the Customer360 platform from business, operational, and architectural perspectives.",
        styles['Body']
    ))
    flowables.append(Spacer(1, 4))

    headers = ["Chapter #", "Chapter Title", "Core Focus &amp; Coverage"]
    rows = []
    for num, title, focus in toc_items:
        rows.append([
            Paragraph(f"<b>{num}</b>", styles['TableCellBold']),
            Paragraph(f"<b>{title}</b>", styles['TableCellBold']),
            Paragraph(focus, styles['TableCell'])
        ])

    table = create_styled_table(headers, rows, [65, 175, 264], styles=styles)
    flowables.append(table)
    return flowables


def build_complete_documentation():
    """Builds Document 1: Complete Project Documentation (All 15 Chapters + Glossary)."""
    pdf_path = os.path.join(DOC_DIR, "Customer_360_Complete_Project_Documentation.pdf")
    print(f"Building Complete Documentation: {pdf_path}...")

    styles = get_document_styles()
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    story = []

    # Cover Page
    story.extend(create_cover_page_flowables(
        doc_title="Customer 360",
        doc_subtitle="Customer Segmentation &amp; Retention Intelligence Platform",
        doc_type="Complete Master Technical &amp; User Documentation Suite",
        styles=styles
    ))
    story.append(PageBreak())

    # Table of Contents
    toc_items = [
        ("Chapter 1", "Project Overview &amp; Business Value", "Problem formulation, economic thesis, target personas, and system overview."),
        ("Chapter 2", "Technology Stack &amp; Ecosystem", "Python 3.11, Streamlit, Scikit-learn, XGBoost, SQLite, Plotly, and OpenPyXL roles."),
        ("Chapter 3", "System Architecture &amp; Data Engineering", "Layered architecture, component decoupling, and data flow pipeline."),
        ("Chapter 4", "Comprehensive Application Modules", "In-depth breakdown of all 7 Streamlit pages: inputs, logic, KPIs, and usage."),
        ("Chapter 5", "Dataset Specifications &amp; Data Contracts", "Telco subscription (7,043 rows) and Retail transactional schema contracts."),
        ("Chapter 6", "Data Processing Pipeline (13 Steps)", "Automated cleaning, whitespace repair, median/mode imputation, and scaling."),
        ("Chapter 7", "Machine Learning &amp; Analytics Methodology", "K-Means, Silhouette, PCA, 5 Churn classifiers, Stratified CV, and Class Weighting."),
        ("Chapter 8", "Model Performance &amp; Verified Benchmarks", "Hold-out test results (N=1,409), ROC/PR curves, Brier score, and threshold tuning."),
        ("Chapter 9", "Business Outcomes &amp; Revenue at Risk", "Expected Monthly Revenue Exposure formula, 0-100 Priority Score, and ROI."),
        ("Chapter 10", "Complete End-User Operational Manual", "Step-by-step operating instructions for business users across all 7 modules."),
        ("Chapter 11", "Reports &amp; Export Specifications", "6 standardized CSV files (UTF-8-SIG), 5-sheet styled Excel workbook, and PDF."),
        ("Chapter 12", "Installation &amp; Developer Setup Guide", "System requirements, virtual environment setup, Pytest suite, and launch steps."),
        ("Chapter 13", "Troubleshooting &amp; Operational FAQ", "Diagnostics for data errors, SQLite locks, empty charts, and user FAQ."),
        ("Chapter 14", "System Limitations &amp; Future Roadmap", "Cross-sectional constraints, real-time webhooks, deep learning, and CLV."),
        ("Chapter 15", "Conclusion &amp; Academic Credits", "Project outcomes, verified capabilities, and Project Made By contributor credits."),
        ("Appendix", "Mathematical Glossary &amp; Formulas", "Formulas and definitions for Accuracy, Precision, Recall, F1, ROC/PR AUC, etc.")
    ]
    story.extend(create_toc_flowables(toc_items, styles))
    story.append(PageBreak())

    # Chapters 1 to 15 + Appendix
    chapters = [
        get_chapter_1, get_chapter_2, get_chapter_3, get_chapter_4, get_chapter_5,
        get_chapter_6, get_chapter_7, get_chapter_8, get_chapter_9, get_chapter_10,
        get_chapter_11, get_chapter_12, get_chapter_13, get_chapter_14, get_chapter_15,
        get_appendix_glossary
    ]

    for i, ch_func in enumerate(chapters):
        story.extend(ch_func(styles))
        if i < len(chapters) - 1:
            story.append(PageBreak())

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Complete Documentation generated successfully: {pdf_path}")


def build_user_manual():
    """Builds Document 2: Dedicated End-User Manual."""
    pdf_path = os.path.join(DOC_DIR, "Customer_360_User_Manual.pdf")
    print(f"Building End-User Manual: {pdf_path}...")

    styles = get_document_styles()
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    story = []

    # Cover Page
    story.extend(create_cover_page_flowables(
        doc_title="Customer 360",
        doc_subtitle="Customer Segmentation &amp; Retention Intelligence Platform",
        doc_type="End-User Operational Manual &amp; Business Workflow Guide",
        styles=styles
    ))
    story.append(PageBreak())

    # TOC for User Manual
    toc_items = [
        ("Section 1", "System Introduction &amp; Value", "What is Customer360 and how it prevents customer churn."),
        ("Section 2", "Platform Navigation &amp; 7 Modules", "Detailed visual guide to all 7 pages and workflows."),
        ("Section 3", "Dataset Specifications &amp; Requirements", "Understanding subscription and transaction customer files."),
        ("Section 4", "Revenue at Risk &amp; Retention Prioritization", "How to interpret Priority Scores (0-100) and revenue exposure."),
        ("Section 5", "Step-by-Step Operating Guide", "Step-by-step instructions for uploading, predicting, and acting."),
        ("Section 6", "Generating Deliverables &amp; Reports", "Downloading 6 CSV reports, multi-sheet Excel files, and PDFs."),
        ("Section 7", "Troubleshooting &amp; User FAQ", "Answers to common questions and quick operational fixes."),
        ("Section 8", "System Contributors &amp; Credits", "Project team and development contributors.")
    ]
    story.extend(create_toc_flowables(toc_items, styles))
    story.append(PageBreak())

    chapters = [
        get_chapter_1, get_chapter_4, get_chapter_5, get_chapter_9,
        get_chapter_10, get_chapter_11, get_chapter_13, get_chapter_15
    ]

    for i, ch_func in enumerate(chapters):
        story.extend(ch_func(styles))
        if i < len(chapters) - 1:
            story.append(PageBreak())

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"User Manual generated successfully: {pdf_path}")


def build_technical_documentation():
    """Builds Document 3: Dedicated Technical Architecture & Developer Specification."""
    pdf_path = os.path.join(DOC_DIR, "Customer_360_Technical_Documentation.pdf")
    print(f"Building Technical Documentation: {pdf_path}...")

    styles = get_document_styles()
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    story = []

    # Cover Page
    story.extend(create_cover_page_flowables(
        doc_title="Customer 360",
        doc_subtitle="Customer Segmentation &amp; Retention Intelligence Platform",
        doc_type="Technical Architecture, Machine Learning &amp; Developer Guide",
        styles=styles
    ))
    story.append(PageBreak())

    # TOC for Technical Documentation
    toc_items = [
        ("Section 1", "System Problem &amp; Architecture Overview", "Architecture fundamentals and analytical objectives."),
        ("Section 2", "Technology Stack &amp; Dependencies", "Python 3.11, Streamlit, Scikit-learn, XGBoost, SQLite stack."),
        ("Section 3", "Layered System Architecture", "Presentation, application, validation, ML, and persistence layers."),
        ("Section 4", "Dataset Specifications &amp; Contracts", "Telco subscription and retail transaction schemas."),
        ("Section 5", "13-Step Preprocessing Pipeline", "Deterministic data cleaning, imputation, scaling, and IQR clipping."),
        ("Section 6", "Machine Learning &amp; Analytics Methodology", "K-Means, PCA, 5 Churn classifiers, Stratified CV, Class Weighting."),
        ("Section 7", "Model Performance &amp; Verified Benchmarks", "Empirical metrics, ROC/PR curves, Brier score, and threshold sweeps."),
        ("Section 8", "Export Architecture &amp; Database Schemas", "6 CSV contracts, OpenPyXL workbooks, SQLite schema."),
        ("Section 9", "Installation &amp; Developer Setup", "Prerequisites, venv setup, Pytest test suite, and launch instructions."),
        ("Section 10", "Technical Troubleshooting &amp; Diagnostics", "KeyError fixes, SQLite locks, and error handling."),
        ("Section 11", "System Limitations &amp; Future Roadmap", "Cross-sectional data boundaries, real-time webhooks, and CLV."),
        ("Section 12", "Conclusion &amp; Academic Credits", "Project outcomes and Project Made By contributor credits."),
        ("Appendix", "Mathematical Glossary &amp; Formulas", "Formal definitions of classification and clustering metrics.")
    ]
    story.extend(create_toc_flowables(toc_items, styles))
    story.append(PageBreak())

    chapters = [
        get_chapter_1, get_chapter_2, get_chapter_3, get_chapter_5,
        get_chapter_6, get_chapter_7, get_chapter_8, get_chapter_11,
        get_chapter_12, get_chapter_13, get_chapter_14, get_chapter_15,
        get_appendix_glossary
    ]

    for i, ch_func in enumerate(chapters):
        story.extend(ch_func(styles))
        if i < len(chapters) - 1:
            story.append(PageBreak())

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Technical Documentation generated successfully: {pdf_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("CUSTOMER360 MASTER DOCUMENTATION PDF COMPILATION")
    print("=" * 60)
    build_complete_documentation()
    print("-" * 60)
    build_user_manual()
    print("-" * 60)
    build_technical_documentation()
    print("=" * 60)
    print("ALL THREE PDF DOCUMENTS COMPILED SUCCESSFULLY!")
    print("=" * 60)
