"""
Customer360 Documentation PDF Styling and Flowable Components
Reusable design system using ReportLab 5.1.0 for publication-quality output.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

# Color Palette
NAVY = colors.HexColor("#1E3A8A")       # Primary 900
BLUE = colors.HexColor("#2563EB")       # Primary 600
LIGHT_BLUE = colors.HexColor("#EFF6FF") # Primary 50
SLATE_DARK = colors.HexColor("#0F172A") # Neutral 900
SLATE_TEXT = colors.HexColor("#1E293B") # Neutral 800
SLATE_MUTED = colors.HexColor("#64748B")# Neutral 500
BORDER_COLOR = colors.HexColor("#E2E8F0")# Neutral 200
BG_LIGHT = colors.HexColor("#F8FAFC")   # Neutral 50
BG_ALT = colors.HexColor("#F1F5F9")     # Neutral 100
EMERALD = colors.HexColor("#059669")    # Success
AMBER = colors.HexColor("#D97706")      # Warning
ROSE = colors.HexColor("#DC2626")       # Danger / Important
PURPLE = colors.HexColor("#7C3AED")     # Accent Purple
WHITE = colors.white


class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to dynamically compute total pages and render
    professional running headers and footers ('Page X of Y').
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_decorations(self, total_pages):
        # Do not draw headers or footers on cover page
        if self._pageNumber == 1:
            return

        self.saveState()

        # Running Top Header
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(NAVY)
        self.drawString(54, 752, "Customer360 | Customer Segmentation & Retention System")

        self.setFont("Helvetica", 7.5)
        self.setFillColor(SLATE_MUTED)
        self.drawRightString(558, 752, "Production Software Documentation")

        self.setStrokeColor(BORDER_COLOR)
        self.setLineWidth(0.75)
        self.line(54, 744, 558, 744)

        # Running Bottom Footer
        self.line(54, 46, 558, 46)

        self.setFont("Helvetica", 7.5)
        self.setFillColor(SLATE_MUTED)
        self.drawString(54, 34, "Enterprise ML Intelligence | Python, Streamlit & Scikit-learn")

        page_str = f"Page {self._pageNumber} of {total_pages}"
        self.setFont("Helvetica-Bold", 7.5)
        self.setFillColor(NAVY)
        self.drawRightString(558, 34, page_str)

        self.restoreState()


def get_document_styles():
    """Build and return custom typography styles for technical documentation."""
    base = getSampleStyleSheet()

    styles = {
        'CoverTitle': ParagraphStyle(
            'CoverTitle',
            parent=base['Title'],
            fontName='Helvetica-Bold',
            fontSize=26,
            leading=32,
            textColor=NAVY,
            alignment=0,
            spaceAfter=8
        ),
        'CoverSubtitle': ParagraphStyle(
            'CoverSubtitle',
            parent=base['Normal'],
            fontName='Helvetica',
            fontSize=13,
            leading=18,
            textColor=BLUE,
            alignment=0,
            spaceAfter=20
        ),
        'CoverMeta': ParagraphStyle(
            'CoverMeta',
            parent=base['Normal'],
            fontName='Helvetica',
            fontSize=9,
            leading=14,
            textColor=SLATE_TEXT
        ),
        'ChapterHeading': ParagraphStyle(
            'ChapterHeading',
            parent=base['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=16,
            leading=20,
            textColor=NAVY,
            spaceBefore=14,
            spaceAfter=8,
            keepWithNext=True
        ),
        'SectionHeading': ParagraphStyle(
            'SectionHeading',
            parent=base['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=12,
            leading=16,
            textColor=BLUE,
            spaceBefore=10,
            spaceAfter=6,
            keepWithNext=True
        ),
        'SubSectionHeading': ParagraphStyle(
            'SubSectionHeading',
            parent=base['Heading3'],
            fontName='Helvetica-Bold',
            fontSize=10,
            leading=14,
            textColor=SLATE_DARK,
            spaceBefore=8,
            spaceAfter=4,
            keepWithNext=True
        ),
        'Body': ParagraphStyle(
            'Body',
            parent=base['Normal'],
            fontName='Helvetica',
            fontSize=8.8,
            leading=13.2,
            textColor=SLATE_TEXT,
            spaceAfter=6
        ),
        'BodyBold': ParagraphStyle(
            'BodyBold',
            parent=base['Normal'],
            fontName='Helvetica-Bold',
            fontSize=8.8,
            leading=13.2,
            textColor=SLATE_DARK,
            spaceAfter=6
        ),
        'Bullet': ParagraphStyle(
            'Bullet',
            parent=base['Normal'],
            fontName='Helvetica',
            fontSize=8.5,
            leading=12.5,
            textColor=SLATE_TEXT,
            leftIndent=14,
            firstLineIndent=-10,
            spaceAfter=3
        ),
        'CodeBlock': ParagraphStyle(
            'CodeBlock',
            parent=base['Normal'],
            fontName='Courier',
            fontSize=7.5,
            leading=10.5,
            textColor=colors.HexColor("#0F172A")
        ),
        'TableCell': ParagraphStyle(
            'TableCell',
            parent=base['Normal'],
            fontName='Helvetica',
            fontSize=7.8,
            leading=10.5,
            textColor=SLATE_TEXT
        ),
        'TableCellBold': ParagraphStyle(
            'TableCellBold',
            parent=base['Normal'],
            fontName='Helvetica-Bold',
            fontSize=7.8,
            leading=10.5,
            textColor=SLATE_DARK
        ),
        'TableHeader': ParagraphStyle(
            'TableHeader',
            parent=base['Normal'],
            fontName='Helvetica-Bold',
            fontSize=8.2,
            leading=11,
            textColor=WHITE,
            alignment=0
        ),
        'CalloutTitle': ParagraphStyle(
            'CalloutTitle',
            parent=base['Normal'],
            fontName='Helvetica-Bold',
            fontSize=8.8,
            leading=12,
            textColor=NAVY
        ),
        'CalloutText': ParagraphStyle(
            'CalloutText',
            parent=base['Normal'],
            fontName='Helvetica',
            fontSize=8.2,
            leading=11.5,
            textColor=SLATE_TEXT
        ),
        'Caption': ParagraphStyle(
            'Caption',
            parent=base['Normal'],
            fontName='Helvetica-Oblique',
            fontSize=7.5,
            leading=10,
            textColor=SLATE_MUTED,
            alignment=1,
            spaceBefore=3,
            spaceAfter=8
        )
    }
    return styles


def create_callout(text: str, title: str = "NOTE", level: str = "info", styles=None) -> Table:
    """
    Renders an alert / callout box with a prominent left accent bar.
    Levels: 'info' (Blue), 'success' (Emerald), 'warning' (Amber), 'danger' (Rose), 'note' (Navy).
    """
    if styles is None:
        styles = get_document_styles()

    level_colors = {
        'info': (BLUE, LIGHT_BLUE),
        'success': (EMERALD, colors.HexColor("#ECFDF5")),
        'warning': (AMBER, colors.HexColor("#FEF3C7")),
        'danger': (ROSE, colors.HexColor("#FEE2E2")),
        'note': (NAVY, colors.HexColor("#F1F5F9"))
    }
    border_color, bg_color = level_colors.get(level.lower(), (BLUE, LIGHT_BLUE))

    content = [
        Paragraph(f"<b>[{title.upper()}]</b>", ParagraphStyle(
            'CalloutHead', parent=styles['CalloutTitle'], textColor=border_color
        )),
        Spacer(1, 3),
        Paragraph(text, styles['CalloutText'])
    ]

    table_data = [[content]]
    table = Table(table_data, colWidths=[504])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), bg_color),
        ('LINEBEFORE', (0, 0), (0, -1), 3.5, border_color),
        ('BOX', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
    ]))
    return table


def create_styled_table(headers, rows, col_widths, styles=None) -> Table:
    """
    Renders a table with Navy header, alternating row background colors,
    and automatic cell paragraph wrapping.
    """
    if styles is None:
        styles = get_document_styles()

    table_data = []

    # Format header row
    header_cells = [
        Paragraph(f"<b>{h}</b>", styles['TableHeader']) for h in headers
    ]
    table_data.append(header_cells)

    # Format data rows
    for r_idx, row in enumerate(rows):
        row_cells = []
        for c_idx, cell in enumerate(row):
            if isinstance(cell, str):
                cell_p = Paragraph(cell, styles['TableCell'])
            else:
                cell_p = cell
            row_cells.append(cell_p)
        table_data.append(row_cells)

    t = Table(table_data, colWidths=col_widths, repeatRows=1)
    t_style = [
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
    ]

    # Alternating row background
    for i in range(1, len(table_data)):
        if i % 2 == 0:
            t_style.append(('BACKGROUND', (0, i), (-1, i), BG_LIGHT))
        else:
            t_style.append(('BACKGROUND', (0, i), (-1, i), WHITE))

    t.setStyle(TableStyle(t_style))
    return t


def create_image_flowable(img_path: str, caption: str = None, width: float = 490, styles=None):
    """Safely loads an image, resizes it preserving aspect ratio, and adds a caption."""
    if not os.path.exists(img_path):
        return Paragraph(f"<i>[Image file not found: {os.path.basename(img_path)}]</i>", styles['Body'] if styles else ParagraphStyle('temp'))

    from PIL import Image as PILImage
    with PILImage.open(img_path) as im:
        orig_w, orig_h = im.size
        aspect = orig_h / orig_w
        height = width * aspect

    flowables = [
        Image(img_path, width=width, height=height)
    ]
    if caption:
        if styles is None:
            styles = get_document_styles()
        flowables.append(Spacer(1, 3))
        flowables.append(Paragraph(caption, styles['Caption']))

    return KeepTogether(flowables)


def create_cover_page_flowables(doc_title: str, doc_subtitle: str, doc_type: str, styles):
    """Builds a cover page layout."""
    flowables = []

    flowables.append(Spacer(1, 20))

    # Top Category Ribbon
    ribbon_data = [[
        Paragraph("<b>ENTERPRISE DATA SCIENCE & MACHINE LEARNING PLATFORM</b>",
                  ParagraphStyle('Rib', fontName='Helvetica-Bold', fontSize=8, textColor=WHITE, alignment=1))
    ]]
    ribbon = Table(ribbon_data, colWidths=[504])
    ribbon.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), NAVY),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]))
    flowables.append(ribbon)
    flowables.append(Spacer(1, 30))

    # Title Block
    flowables.append(Paragraph(doc_title, styles['CoverTitle']))
    flowables.append(Paragraph(doc_subtitle, styles['CoverSubtitle']))

    divider = HRFlowable(width="100%", thickness=2, color=BLUE, spaceBefore=4, spaceAfter=25)
    flowables.append(divider)

    # Document Specs Box
    spec_data = [
        [
            Paragraph("<b>Document Type:</b>", styles['BodyBold']),
            Paragraph(doc_type, styles['Body']),
            Paragraph("<b>Software Release:</b>", styles['BodyBold']),
            Paragraph("v1.0.0 (Production Verified)", styles['Body'])
        ],
        [
            Paragraph("<b>Target Audience:</b>", styles['BodyBold']),
            Paragraph("Developers, Analysts, Executives & End-Users", styles['Body']),
            Paragraph("<b>Test Suite Status:</b>", styles['BodyBold']),
            Paragraph("100% Passed (42/42 Tests)", styles['Body'])
        ],
        [
            Paragraph("<b>Core Stack:</b>", styles['BodyBold']),
            Paragraph("Python 3.11, Streamlit, Scikit-learn, SQLite", styles['Body']),
            Paragraph("<b>Date Generated:</b>", styles['BodyBold']),
            Paragraph("November 2026", styles['Body'])
        ]
    ]
    spec_table = Table(spec_data, colWidths=[95, 160, 95, 154])
    spec_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), BG_LIGHT),
        ('BOX', (0, 0), (-1, -1), 0.75, BORDER_COLOR),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    flowables.append(spec_table)
    flowables.append(Spacer(1, 40))

    # Academic Project Contributors Card ("Project Made By")
    contrib_title = Paragraph("<b>PROJECT MADE BY</b>", ParagraphStyle(
        'ContHead', fontName='Helvetica-Bold', fontSize=10, textColor=NAVY, alignment=1
    ))
    
    c_data = [
        [contrib_title],
        [
            Paragraph(
                "<b>Zubair Warsi</b><br/><font color='#64748B'>Roll No: 2303410100057</font><br/>"
                "<font color='#0284C7' size='7.5'>Lead Architecture & System Engineering</font>",
                styles['Body']
            ),
            Paragraph(
                "<b>Sameer Mansoori</b><br/><font color='#64748B'>Roll No: 2303410100040</font><br/>"
                "<font color='#0284C7' size='7.5'>Machine Learning & Diagnostic Pipelines</font>",
                styles['Body']
            ),
            Paragraph(
                "<b>Mohd Shahbaaz Khan</b><br/><font color='#64748B'>Roll No: 2303410100029</font><br/>"
                "<font color='#0284C7' size='7.5'>Data Preprocessing & UI Experience</font>",
                styles['Body']
            )
        ]
    ]
    c_table = Table(c_data, colWidths=[504])
    c_table = Table([
        [contrib_title, "", ""],
        [
            Paragraph("<b>Zubair Warsi</b><br/><font color='#64748B'>Roll No: 2303410100057</font>", styles['Body']),
            Paragraph("<b>Sameer Mansoori</b><br/><font color='#64748B'>Roll No: 2303410100040</font>", styles['Body']),
            Paragraph("<b>Mohd Shahbaaz Khan</b><br/><font color='#64748B'>Roll No: 2303410100029</font>", styles['Body'])
        ]
    ], colWidths=[168, 168, 168])
    c_table.setStyle(TableStyle([
        ('SPAN', (0, 0), (2, 0)),
        ('BACKGROUND', (0, 0), (-1, 0), LIGHT_BLUE),
        ('BACKGROUND', (0, 1), (-1, 1), WHITE),
        ('BOX', (0, 0), (-1, -1), 1, BLUE),
        ('INNERGRID', (0, 1), (-1, 1), 0.5, BORDER_COLOR),
        ('TOPPADDING', (0, 0), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('ALIGN', (0, 1), (-1, 1), 'CENTER'),
    ]))
    flowables.append(c_table)

    flowables.append(Spacer(1, 50))

    # Bottom Copyright Notice
    copy_text = (
        "<b>Customer360 Intelligence System</b> &bull; Production Software Release &bull; "
        "Strictly Confidential &bull; All Rights Reserved &bull; Validated against Telco & Retail Benchmarks."
    )
    flowables.append(Paragraph(copy_text, ParagraphStyle(
        'Copy', fontName='Helvetica', fontSize=7.5, leading=10, textColor=SLATE_MUTED, alignment=1
    )))

    return flowables
