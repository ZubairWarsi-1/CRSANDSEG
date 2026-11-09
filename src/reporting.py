"""
Customer360 - Reporting and Multi-Format Export Module
Generates CSV exports, multi-sheet Excel workbooks (.xlsx), professional PDF executive audits,
and one-page executive summaries with SQLite report history logging.
"""

from typing import Dict, Any, List, Optional
import io
from datetime import datetime
import pandas as pd
import numpy as np

# OpenPyXL for Excel
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# ReportLab for PDF
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    KeepTogether,
    HRFlowable
)

from src.database import log_report_generation


def export_dataframe_to_csv(df: pd.DataFrame) -> bytes:
    """Serialize a pandas DataFrame into utf-8 encoded CSV bytes for download."""
    buffer = io.StringIO()
    df.to_csv(buffer, index=False, encoding="utf-8")
    return buffer.getvalue().encode("utf-8")


def export_multi_sheet_excel(sheets_dict: Dict[str, pd.DataFrame]) -> bytes:
    """
    Export multiple analytical dataframes into a styled multi-sheet Excel (.xlsx) workbook.
    """
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for sheet_name, df in sheets_dict.items():
            safe_name = sheet_name[:31]  # Excel max sheet name length
            if df is None or df.empty:
                df = pd.DataFrame({"Status": ["No data available for this section"]})
            df.to_excel(writer, sheet_name=safe_name, index=False)

    # Re-open workbook from buffer to apply professional formatting
    buffer.seek(0)
    wb = openpyxl.load_workbook(buffer)

    header_fill = PatternFill(start_color="1E3A8A", end_color="1E3A8A", fill_type="solid")
    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    data_font = Font(name="Calibri", size=10)
    border_side = Side(style="thin", color="E2E8F0")
    thin_border = Border(left=border_side, right=border_side, top=border_side, bottom=border_side)

    for sheet in wb.worksheets:
        sheet.views.sheetView[0].showGridLines = True
        
        # Style headers
        for col_idx in range(1, sheet.max_column + 1):
            cell = sheet.cell(row=1, column=col_idx)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center")
            sheet.row_dimensions[1].height = 26

            # Auto-adjust column width
            max_len = max(len(str(cell.value or "")), 12)
            col_letter = get_column_letter(col_idx)
            sheet.column_dimensions[col_letter].width = min(max_len + 4, 45)

        # Style data cells
        for row in range(2, min(sheet.max_row + 1, 1000)):
            for col in range(1, sheet.max_column + 1):
                cell = sheet.cell(row=row, column=col)
                cell.font = data_font
                cell.border = thin_border
                if row % 2 == 0:
                    cell.fill = PatternFill(start_color="F8FAFC", end_color="F8FAFC", fill_type="solid")

    out_buffer = io.BytesIO()
    wb.save(out_buffer)
    return out_buffer.getvalue()


def generate_pdf_report(report_data: Dict[str, Any]) -> bytes:
    """
    Generate a publication-grade executive PDF intelligence report using ReportLab.
    Includes KPIs, Revenue at Risk, Segmentation, Churn Model Evaluation,
    Prescriptive Retention Queue, and Methodological Disclaimers.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1E3A8A"),
        spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#64748B"),
        spaceAfter=12
    )
    h2_style = ParagraphStyle(
        "SectionH2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=12,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        "DocBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155")
    )
    caption_style = ParagraphStyle(
        "DocCaption",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#64748B"),
        spaceAfter=6
    )

    story = []

    # Title & Metadata
    story.append(Paragraph("Customer360 — Customer Intelligence Platform", title_style))
    story.append(Paragraph("Executive Retention Intelligence, Predictive Modeling & Cohort Segmentation Audit", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563EB"), spaceAfter=10))

    # Meta table
    now_str = datetime.now().strftime("%B %d, %Y - %H:%M:%S")
    dataset_name = report_data.get("dataset_name", "IBM Telco Customer Churn")
    dataset_source = report_data.get("dataset_source", "Benchmark Dataset")
    row_count = report_data.get("row_count", 0)
    col_count = report_data.get("col_count", 0)
    quality_score = report_data.get("quality_score", "N/A")

    meta_table_data = [
        [
            Paragraph("<b>Generated:</b> " + now_str, body_style),
            Paragraph("<b>Dataset:</b> " + str(dataset_name), body_style)
        ],
        [
            Paragraph(f"<b>Dimensions:</b> {row_count:,} rows × {col_count} features", body_style),
            Paragraph(f"<b>Data Classification:</b> {dataset_source}", body_style)
        ],
        [
            Paragraph(f"<b>Data Quality Score:</b> {quality_score}/100", body_style),
            Paragraph(f"<b>Active Model:</b> {report_data.get('selected_model', 'Random Forest')}", body_style)
        ]
    ]
    meta_table = Table(meta_table_data, colWidths=[270, 270])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # Section 1: Executive KPI & Revenue at Risk Summary
    story.append(Paragraph("1. Executive Portfolio KPIs & Revenue at Risk", h2_style))
    kpi_data = [
        ["Total Customers", "Active Accounts", "Avg Monthly Spend", "Historical Churn", "High-Risk Accounts", "Monthly Revenue at Risk"],
        [
            f"{row_count:,}",
            f"{report_data.get('active_customers', 'N/A')}",
            f"${report_data.get('avg_spending', 0.0):,.2f}",
            f"{report_data.get('churn_rate', 'N/A')}",
            f"{report_data.get('high_risk_count', 'N/A')}",
            f"${report_data.get('revenue_at_risk', 0.0):,.2f}"
        ]
    ]
    kpi_table = Table(kpi_data, colWidths=[90, 90, 90, 90, 90, 90])
    kpi_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#FFFFFF")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 6))
    story.append(Paragraph("<i>Note: Revenue at Risk represents expected monthly revenue exposure calculated as &Sigma;(Monthly Charges &times; Churn Probability).</i>", caption_style))

    # Section 2: Churn Prediction & Model Evaluation
    story.append(Paragraph("2. Churn Prediction & Machine Learning Performance", h2_style))
    eval_metrics = report_data.get("evaluation_metrics", {})
    if eval_metrics:
        model_table_data = [
            ["Algorithm", "Test Accuracy", "Test Recall", "Test Precision", "Test F1-Score", "Test ROC-AUC", "5-Fold CV F1"]
        ]
        for m in eval_metrics.get("models_summary", []):
            model_table_data.append([
                m.get("Algorithm", "N/A"),
                f"{m.get('Test Accuracy', 0)*100:.1f}%",
                f"{m.get('Test Recall (Sensitivity)', 0)*100:.1f}%",
                f"{m.get('Test Precision', 0)*100:.1f}%",
                f"{m.get('Test F1-Score', 0):.3f}",
                f"{m.get('Test ROC-AUC', 0):.3f}" if isinstance(m.get('Test ROC-AUC'), (int, float)) else str(m.get('Test ROC-AUC')),
                str(m.get("CV F1 (5-Fold)", "N/A"))
            ])
        m_table = Table(model_table_data, colWidths=[120, 70, 70, 70, 70, 70, 70])
        m_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563EB")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("PADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(m_table)
    else:
        story.append(Paragraph("Model evaluation benchmarks pending. Run Churn Prediction to generate performance matrices.", body_style))

    story.append(Spacer(1, 10))

    # Section 3: Customer Segmentation Summary
    story.append(Paragraph("3. Customer Behavioral Segmentation", h2_style))
    seg_summary = report_data.get("segmentation_summary", [])
    if seg_summary:
        seg_table_data = [["Cluster / Segment Name", "Customer Count", "% Share", "Avg Tenure", "Avg Spend", "Churn Rate"]]
        for s in seg_summary[:5]:
            seg_table_data.append([
                str(s.get("ClusterName", s.get("Segment", "N/A"))),
                f"{s.get('CustomerCount', 0):,}",
                f"{s.get('CustomerPercentage', s.get('CustomerPct', 0)):.1f}%",
                f"{s.get('Avg_tenure', s.get('AvgRecency', 'N/A'))}",
                f"${s.get('Avg_MonthlyCharges', s.get('AvgMonetary', 0.0)):,.2f}",
                f"{s.get('HistoricalChurnRate', 'N/A')}%" if s.get('HistoricalChurnRate') is not None else "N/A"
            ])
        s_table = Table(seg_table_data, colWidths=[180, 70, 60, 75, 75, 80])
        s_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("PADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(s_table)
    else:
        story.append(Paragraph("Customer segmentation profiles pending execution.", body_style))

    story.append(Spacer(1, 10))

    # Section 4: Prescriptive Retention Actions
    story.append(Paragraph("4. Recommended Retention Strategies", h2_style))
    story.append(Paragraph(
        "• <b>Personalized Win-Back Offers:</b> Dispatched to high-risk, premium spending accounts.<br/>"
        "• <b>Onboarding Guidance:</b> Proactive onboarding check-in for accounts in first 6 months.<br/>"
        "• <b>Customer Support Follow-Up:</b> Dedicated resolution outreach for accounts with service friction.<br/>"
        "• <b>VIP Loyalty Rewards:</b> Relationship preservation for high-tenure champions.",
        body_style
    ))

    story.append(Spacer(1, 10))

    # Section 5: Academic & Methodological Disclaimers
    story.append(Paragraph("5. Methodological & Statistical Disclaimers", h2_style))
    story.append(Paragraph(
        "1. <b>Statistical Nature:</b> Predicted churn probabilities represent mathematical estimates derived from observational features. They do not constitute deterministic guarantees.<br/>"
        "2. <b>Correlation vs Causation:</b> Identified feature importances reflect statistical correlations within the training dataset and do not automatically prove direct real-world causality.<br/>"
        "3. <b>Data Governance:</b> Retention workflows should be combined with qualitative customer relationship feedback.",
        caption_style
    ))

    doc.build(story)
    pdf_bytes = buffer.getvalue()

    # Log generation to SQLite
    try:
        log_report_generation(
            report_name=f"Executive_Intelligence_Report_{dataset_name}",
            dataset_name=str(dataset_name),
            report_type="PDF",
            selected_model=report_data.get("selected_model", "Random Forest"),
            evaluation_metrics=report_data.get("evaluation_metrics", {})
        )
    except Exception:
        pass

    return pdf_bytes


def generate_one_page_executive_summary_pdf(data: Dict[str, Any]) -> bytes:
    """
    Generate a concise, one-page executive summary PDF briefing.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=18, fontName="Helvetica-Bold", textColor=colors.HexColor("#1E3A8A"), spaceAfter=2)
    sub_style = ParagraphStyle("Sub", parent=styles["Normal"], fontSize=9, textColor=colors.HexColor("#64748B"), spaceAfter=10)
    h2_style = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=11, fontName="Helvetica-Bold", textColor=colors.HexColor("#0F172A"), spaceBefore=8, spaceAfter=4)
    body_style = ParagraphStyle("Body", parent=styles["Normal"], fontSize=8.5, leading=12, textColor=colors.HexColor("#334155"))

    story = []
    story.append(Paragraph("Customer360 — One-Page Executive Briefing", title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y')} | Dataset: {data.get('dataset_name', 'Active Dataset')}", sub_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2563EB"), spaceAfter=8))

    # Core Metric Grid
    kpi_grid = [
        ["Total Portfolio", "Active Customers", "Churn Rate", "Revenue Exposure"],
        [
            f"{data.get('row_count', 0):,}",
            f"{data.get('active_customers', 'N/A')}",
            f"{data.get('churn_rate', 'N/A')}",
            f"${data.get('revenue_at_risk', 0.0):,.2f}"
        ]
    ]
    t = Table(kpi_grid, colWidths=[135, 135, 135, 135])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

    story.append(Paragraph("Key Findings & Attrition Drivers", h2_style))
    insights = data.get("insights", [
        "Month-to-month contracts exhibit significantly higher churn vulnerability than multi-year contracts.",
        "Attrition is heavily concentrated in the first 12 months of customer lifecycle.",
        "Fiber optic accounts without bundled tech support represent prime candidates for proactive intervention."
    ])
    for ins in insights[:4]:
        story.append(Paragraph(f"• {ins}", body_style))

    story.append(Spacer(1, 10))
    story.append(Paragraph("Prescriptive Intervention Plan", h2_style))
    story.append(Paragraph(
        "1. <b>Executive Retention:</b> Focus immediate high-touch outreach on high-spending accounts with >65% predicted churn probability.<br/>"
        "2. <b>Early-Life Check-ins:</b> Trigger 30-day onboarding assistance for all subscribers in tenure months 1–6.<br/>"
        "3. <b>Service Bundling:</b> Package technical support with fiber products to remove primary technical cancellation triggers.",
        body_style
    ))

    doc.build(story)
    return buffer.getvalue()


def generate_data_quality_markdown(inspection: Dict[str, Any]) -> str:
    """Generate an executive markdown report summarizing dataset health and readiness."""
    deductions_str = "\n".join([f"- {d}" for d in inspection.get("deductions", [])]) or "- No major deductions applied."
    md = f"""# Customer360 - Data Quality Audit Report

**Overall Quality Score:** {inspection.get('quality_score', 'N/A')}/100 ({inspection.get('grade', 'N/A')})  
**Dataset Dimensions:** {inspection.get('row_count', 0):,} rows × {inspection.get('col_count', 0)} columns  
**Memory Footprint:** {inspection.get('memory_mb', 0)} MB  
**Duplicate Rows:** {inspection.get('duplicate_count', 0):,} ({inspection.get('duplicate_pct', 0)}%)  
**Duplicate Customer IDs:** {inspection.get('duplicate_ids', 0):,}  
**Total Missing Cells:** {inspection.get('total_missing_cells', 0):,}  
**Zero-Variance Columns:** {len(inspection.get('constant_cols', []))}  

### Quality Score Deduction Breakdown
{deductions_str}

---

### Column Breakdown
| Column Name | Inferred Type | Missing Count | Missing % | Unique Count |
| :--- | :--- | :--- | :--- | :--- |
"""
    for col in inspection.get("columns_info", []):
        md += f"| `{col['name']}` | {col['type']} | {col['missing_count']} | {col['missing_pct']}% | {col['unique_count']} |\n"

    md += """
---
### Analytical Readiness
- **Supervised Churn Modeling:** Requires a binary target column (e.g., 'Churn' with Yes/No or 1/0).
- **RFM Segmentation:** Requires transaction date, customer identifier, and monetary purchase value.
- **K-Means Clustering:** Requires at least 2 numerical features with non-zero variance.

*Generated by Customer360 Customer Intelligence Platform.*
"""
    return md


def get_data_quality_df(inspection: Dict[str, Any]) -> pd.DataFrame:
    """Convert column inspection details into a DataFrame for export."""
    cols = inspection.get("columns_info", [])
    if not cols:
        return pd.DataFrame()
    return pd.DataFrame(cols)
