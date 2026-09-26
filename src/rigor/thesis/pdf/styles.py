"""
Academic PDF Styles
===================
ReportLab paragraph and table styles for academic thesis formatting.
Uses Windows TrueType fonts for high-quality rendering.
"""

import os

# ---------------------------------------------------------------------------
# Register TrueType fonts (Times New Roman) for crisp rendering.
# Supports Windows, macOS, and Linux.
# Falls back to built-in Type1 fonts if TTF files are not found.
# ---------------------------------------------------------------------------
import platform as _platform

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import TableStyle


def _find_times_fonts() -> dict:
    """Locate Times New Roman TTF files across platforms."""
    system = _platform.system()

    if system == "Windows":
        candidates = [
            {
                'TNR':       os.path.join(r'C:\Windows\Fonts', 'times.ttf'),
                'TNR-Bold':  os.path.join(r'C:\Windows\Fonts', 'timesbd.ttf'),
                'TNR-Italic': os.path.join(r'C:\Windows\Fonts', 'timesi.ttf'),
                'TNR-BI':    os.path.join(r'C:\Windows\Fonts', 'timesbi.ttf'),
            }
        ]
    elif system == "Darwin":  # macOS
        # macOS stores Times New Roman with full names (spaces)
        _mac_dirs = ["/Library/Fonts", os.path.expanduser("~/Library/Fonts")]
        # Try "Times New Roman*.ttf" first (Microsoft Office install), then "Times*.ttf"
        _mac_names = [
            {
                'TNR':       'Times New Roman.ttf',
                'TNR-Bold':  'Times New Roman Bold.ttf',
                'TNR-Italic': 'Times New Roman Italic.ttf',
                'TNR-BI':    'Times New Roman Bold Italic.ttf',
            },
        ]
        candidates = []
        for names in _mac_names:
            for d in _mac_dirs:
                m = {k: os.path.join(d, v) for k, v in names.items()}
                candidates.append(m)
    else:  # Linux
        _linux_dirs = [
            "/usr/share/fonts/truetype/msttcorefonts",
            "/usr/share/fonts/TTF",
            os.path.expanduser("~/.fonts"),
        ]
        candidates = []
        for d in _linux_dirs:
            candidates.append({
                'TNR':       os.path.join(d, 'Times_New_Roman.ttf'),
                'TNR-Bold':  os.path.join(d, 'Times_New_Roman_Bold.ttf'),
                'TNR-Italic': os.path.join(d, 'Times_New_Roman_Italic.ttf'),
                'TNR-BI':    os.path.join(d, 'Times_New_Roman_Bold_Italic.ttf'),
            })

    # Return first complete set found
    for mapping in candidates:
        if all(os.path.exists(p) for p in mapping.values()):
            return mapping
    return {}

_TTF_MAP = _find_times_fonts()

# Font names that styles will reference
FONT_REGULAR = 'Times-Roman'
FONT_BOLD = 'Times-Bold'
FONT_ITALIC = 'Times-Italic'
FONT_BOLD_ITALIC = 'Times-BoldItalic'

# Try to register TrueType fonts; update names on success
_ttf_registered = False
if _TTF_MAP and all(os.path.exists(p) for p in _TTF_MAP.values()):
    try:
        pdfmetrics.registerFont(TTFont('TNR', _TTF_MAP['TNR']))
        pdfmetrics.registerFont(TTFont('TNR-Bold', _TTF_MAP['TNR-Bold']))
        pdfmetrics.registerFont(TTFont('TNR-Italic', _TTF_MAP['TNR-Italic']))
        pdfmetrics.registerFont(TTFont('TNR-BI', _TTF_MAP['TNR-BI']))

        # Register as a font family so <b> and <i> tags in Paragraphs work
        from reportlab.pdfbase.pdfmetrics import registerFontFamily
        registerFontFamily(
            'TNR',
            normal='TNR',
            bold='TNR-Bold',
            italic='TNR-Italic',
            boldItalic='TNR-BI',
        )

        FONT_REGULAR = 'TNR'
        FONT_BOLD = 'TNR-Bold'
        FONT_ITALIC = 'TNR-Italic'
        FONT_BOLD_ITALIC = 'TNR-BI'
        _ttf_registered = True
    except Exception:
        pass  # Fall back to Type1


class AcademicPDFStyles:
    """All paragraph and table styles for the academic thesis PDF.

    Design: Times New Roman throughout, justified body text,
    numbered sections, formal table formatting with horizontal rules only.
    """

    def __init__(self):
        self.base = getSampleStyleSheet()
        self._build_styles()

    def _build_styles(self):
        # --- Title page ---
        self.title = ParagraphStyle(
            'ThesisTitle', parent=self.base['Title'],
            fontName=FONT_BOLD, fontSize=22,
            alignment=TA_CENTER, spaceAfter=12, leading=28,
            textColor=colors.HexColor('#000000'),
        )
        self.subtitle = ParagraphStyle(
            'ThesisSubtitle', parent=self.base['Normal'],
            fontName=FONT_ITALIC, fontSize=14,
            alignment=TA_CENTER, spaceAfter=6,
            textColor=colors.HexColor('#222222'),
        )
        self.author = ParagraphStyle(
            'ThesisAuthor', parent=self.base['Normal'],
            fontName=FONT_REGULAR, fontSize=12,
            alignment=TA_CENTER, spaceAfter=4,
            textColor=colors.HexColor('#111111'),
        )
        self.date_style = ParagraphStyle(
            'ThesisDate', parent=self.base['Normal'],
            fontName=FONT_REGULAR, fontSize=11,
            alignment=TA_CENTER, spaceAfter=4,
            textColor=colors.HexColor('#333333'),
        )

        # --- Section headings ---
        self.heading1 = ParagraphStyle(
            'ThesisH1', parent=self.base['Heading1'],
            fontName=FONT_BOLD, fontSize=14,
            spaceBefore=24, spaceAfter=12,
            textColor=colors.HexColor('#000000'),
            keepWithNext=True,
        )
        self.heading2 = ParagraphStyle(
            'ThesisH2', parent=self.base['Heading2'],
            fontName=FONT_BOLD, fontSize=12,
            spaceBefore=18, spaceAfter=8,
            textColor=colors.HexColor('#111111'),
            keepWithNext=True,
        )
        self.heading3 = ParagraphStyle(
            'ThesisH3', parent=self.base['Heading3'],
            fontName=FONT_BOLD_ITALIC, fontSize=11,
            spaceBefore=12, spaceAfter=6,
            textColor=colors.HexColor('#222222'),
            keepWithNext=True,
        )

        # --- Body text ---
        self.body = ParagraphStyle(
            'ThesisBody', parent=self.base['BodyText'],
            fontName=FONT_REGULAR, fontSize=11,
            alignment=TA_JUSTIFY, spaceAfter=8,
            leading=15,
            textColor=colors.HexColor('#000000'),
        )
        self.body_italic = ParagraphStyle(
            'ThesisBodyItalic', parent=self.body,
            fontName=FONT_ITALIC,
        )

        # --- Abstract ---
        self.abstract_heading = ParagraphStyle(
            'AbstractHeading', parent=self.base['Normal'],
            fontName=FONT_BOLD, fontSize=12,
            alignment=TA_CENTER, spaceBefore=12, spaceAfter=6,
            textColor=colors.HexColor('#000000'),
        )
        self.abstract = ParagraphStyle(
            'ThesisAbstract', parent=self.body,
            fontName=FONT_ITALIC, fontSize=10,
            leftIndent=36, rightIndent=36, spaceAfter=6,
            leading=13,
        )

        # --- Figure captions ---
        self.figure_caption = ParagraphStyle(
            'FigureCaption', parent=self.base['Normal'],
            fontName=FONT_REGULAR, fontSize=9,
            alignment=TA_CENTER, spaceBefore=6, spaceAfter=16,
            textColor=colors.HexColor('#222222'),
            leading=12,
        )

        # --- Table captions ---
        self.table_caption = ParagraphStyle(
            'TableCaption', parent=self.base['Normal'],
            fontName=FONT_BOLD, fontSize=10,
            alignment=TA_LEFT, spaceBefore=12, spaceAfter=4,
            textColor=colors.HexColor('#000000'),
            leading=12,
        )

        # --- Disclaimer / footnote ---
        self.disclaimer = ParagraphStyle(
            'Disclaimer', parent=self.base['Normal'],
            fontName=FONT_ITALIC, fontSize=8,
            alignment=TA_LEFT, spaceBefore=4, spaceAfter=2,
            textColor=colors.HexColor('#444444'),
            leading=10,
        )

        # --- Bullet list item ---
        self.bullet = ParagraphStyle(
            'ThesisBullet', parent=self.body,
            leftIndent=24, bulletIndent=12,
            spaceBefore=2, spaceAfter=2,
        )

        # --- Table of Contents entries ---
        self.toc_h1 = ParagraphStyle(
            'TOCH1', parent=self.base['Normal'],
            fontName=FONT_REGULAR, fontSize=11,
            alignment=TA_LEFT, spaceBefore=6, spaceAfter=6,
            leftIndent=12, leading=16,
            textColor=colors.HexColor('#000000'),
        )
        self.toc_h2 = ParagraphStyle(
            'TOCH2', parent=self.toc_h1,
            fontSize=10, leftIndent=36, spaceBefore=2, spaceAfter=2,
            textColor=colors.HexColor('#222222'),
        )

    def metrics_table_style(self) -> TableStyle:
        """Academic booktabs table: heavy top/bottom rules, thin header rule."""
        return TableStyle([
            # Fonts — 10pt matches academic convention (slightly smaller than 11pt body)
            ('FONTNAME', (0, 0), (-1, 0), FONT_BOLD),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('FONTNAME', (0, 1), (-1, -1), FONT_REGULAR),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#000000')),
            # Alignment
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            # Rules: heavy top, thin mid, heavy bottom (booktabs)
            ('LINEABOVE', (0, 0), (-1, 0), 2.0, colors.black),
            ('LINEBELOW', (0, 0), (-1, 0), 0.75, colors.black),
            ('LINEBELOW', (0, -1), (-1, -1), 2.0, colors.black),
            # Generous padding
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ])

    def monthly_returns_table_style(self, n_rows: int,
                                     cell_colors: list | None = None) -> TableStyle:
        """Color-coded monthly returns table style.

        Args:
            n_rows: Number of data rows (excluding header).
            cell_colors: Optional list of (col, row, Color) tuples for per-cell
                         background coloring. Row indices are 1-based (0 = header).
        """
        style_commands = [
            # Fonts
            ('FONTNAME', (0, 0), (-1, 0), FONT_BOLD),
            ('FONTSIZE', (0, 0), (-1, -1), 7.5),
            ('FONTNAME', (0, 1), (-1, -1), FONT_REGULAR),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#000000')),
            # Alignment
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('ALIGN', (0, 0), (0, -1), 'CENTER'),
            # Rules
            ('LINEABOVE', (0, 0), (-1, 0), 1.5, colors.black),
            ('LINEBELOW', (0, 0), (-1, 0), 0.75, colors.black),
            ('LINEBELOW', (0, -1), (-1, -1), 1.5, colors.black),
            # Padding
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('LEFTPADDING', (0, 0), (-1, -1), 3),
            ('RIGHTPADDING', (0, 0), (-1, -1), 3),
            # Header row background
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f0f0f0')),
            ('FONTNAME', (0, 0), (-1, 0), FONT_BOLD),
            # Year column (first column) bold + background
            ('FONTNAME', (0, 1), (0, -1), FONT_BOLD),
            ('BACKGROUND', (0, 1), (0, -1), colors.HexColor('#f0f0f0')),
        ]

        # Per-cell background colors for the heatmap effect
        if cell_colors:
            for col_idx, row_idx, color in cell_colors:
                style_commands.append(
                    ('BACKGROUND', (col_idx, row_idx), (col_idx, row_idx), color)
                )

        return TableStyle(style_commands)
