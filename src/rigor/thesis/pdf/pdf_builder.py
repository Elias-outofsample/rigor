"""
ReportLab PDF Builder
=====================
Manages the PDF document construction: paragraphs, figures, tables, page layout.

Figures and math equations are rendered as SVG vector graphics (via svglib)
for infinite-resolution crispness in the PDF output.
"""

import html
import io
import os

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
)

from .styles import FONT_REGULAR, AcademicPDFStyles


class ReportLabPDFBuilder:
    """Builds the thesis PDF using ReportLab Platypus flowables."""

    PAGE_WIDTH = letter[0]
    PAGE_HEIGHT = letter[1]
    TEXT_WIDTH = PAGE_WIDTH - 2.1 * inch  # Slightly under 1" margins to prevent overflow

    def __init__(self, output_path: str, title: str = "", author: str = ""):
        self.output_path = output_path
        self.title = title
        self.author = author
        self.styles = AcademicPDFStyles()
        self.elements: list = []
        self.figure_counter = 0
        self.table_counter = 0
        self._section_counter = 0
        self._temp_files: list[str] = []

    # ------------------------------------------------------------------
    # Section numbering
    # ------------------------------------------------------------------

    def next_section_number(self) -> int:
        """Increment and return the next section number."""
        self._section_counter += 1
        return self._section_counter

    # ------------------------------------------------------------------
    # Content addition methods
    # ------------------------------------------------------------------

    def add_paragraph(self, text: str, style_name: str = 'body'):
        """Add a paragraph with the named style."""
        style = getattr(self.styles, style_name, self.styles.body)
        self.elements.append(Paragraph(text, style))

    def add_heading(self, text: str, level: int = 1):
        """Add a heading at the given level (1, 2, or 3)."""
        style_map = {1: self.styles.heading1, 2: self.styles.heading2, 3: self.styles.heading3}
        style = style_map.get(level, self.styles.heading1)
        self.elements.append(Paragraph(text, style))

    def add_spacer(self, height: float = 12):
        """Add vertical space."""
        self.elements.append(Spacer(1, height))

    def add_page_break(self):
        """Insert a page break."""
        self.elements.append(PageBreak())

    def add_figure(self, image_path: str, caption: str,
                   width: float | None = None) -> int:
        """Add a figure image with an auto-numbered caption below.

        Supports PNG/JPG (raster) and SVG (vector). SVG files are converted
        to native ReportLab Drawing objects for infinite-resolution crispness.

        Returns the figure number.
        """
        self.figure_counter += 1
        max_width = self.TEXT_WIDTH
        width = min(width, max_width) if width else max_width

        if not os.path.exists(image_path):
            self.add_paragraph(f"[Figure {self.figure_counter} image not found: {image_path}]",
                               'body_italic')
            return self.figure_counter

        if image_path.lower().endswith('.svg'):
            # Vector rendering via svglib
            self._add_svg_figure(image_path, width)
        else:
            # Raster fallback (PNG/JPG)
            img = Image(image_path)
            img_w, img_h = img.imageWidth, img.imageHeight
            if img_w > 0:
                ratio = width / img_w
                scaled_h = img_h * ratio
                max_height = 550
                if scaled_h > max_height:
                    ratio = max_height / img_h
                img.drawWidth = img_w * ratio
                img.drawHeight = img_h * ratio
            self.elements.append(img)

        caption_text = f"<b>Figure {self.figure_counter}:</b> {html.escape(caption)}"
        caption_para = Paragraph(caption_text, self.styles.figure_caption)
        self.elements.append(caption_para)
        self.add_spacer(8)
        return self.figure_counter

    def _add_svg_figure(self, svg_path: str, target_width: float):
        """Embed an SVG file as a native vector Drawing."""
        try:
            from svglib.svglib import svg2rlg
            drawing = svg2rlg(svg_path)
            if drawing and drawing.width > 0:
                scale = target_width / drawing.width
                drawing.width = target_width
                drawing.height *= scale
                drawing.scale(scale, scale)
            self.elements.append(drawing)
        except Exception as e:
            print(f"  WARNING: SVG rendering failed for {svg_path}: {e}")
            # Fallback to raster
            img = Image(svg_path.replace('.svg', '.png'))
            self.elements.append(img)

    def add_table(self, data: list[list], caption: str,
                  col_widths: list[float] | None = None,
                  style_override=None,
                  cell_bg_colors: dict | None = None,
                  extra_rules: list | None = None,
                  bold_cells: list | None = None,
                  renderer: str = "native",
                  note: str = "") -> int:
        """Add a table with an auto-numbered caption above.

        Args:
            renderer: "native" for selectable Platypus Table (default),
                      "matplotlib" for image-based rendering (heatmaps).
            note: Optional italic footnote rendered under the table
                  (CHANGE-001 — table note, JFE/RFS academic convention).

        Returns the table number.
        """
        self.table_counter += 1

        caption_text = f"<b>Table {self.table_counter}:</b> {html.escape(caption)}"
        caption_para = Paragraph(caption_text, self.styles.table_caption)
        self.elements.append(caption_para)

        # Convert all cells to strings
        str_data = []
        for row in data:
            str_data.append([str(cell) if cell is not None else '' for cell in row])

        if renderer == "native" and not cell_bg_colors:
            self._add_native_table(str_data)
        else:
            self._add_matplotlib_table(str_data, cell_bg_colors, extra_rules, bold_cells)

        # CHANGE-001 — table note: render italic footnote under the table if provided
        if note:
            from reportlab.lib.enums import TA_LEFT
            from reportlab.lib.styles import ParagraphStyle
            note_style = ParagraphStyle(
                name="TableNote",
                parent=self.styles.body,
                fontName=self.styles.body.fontName,
                fontSize=max(7.5, self.styles.body.fontSize - 1.5),
                leading=max(9, self.styles.body.leading - 2),
                alignment=TA_LEFT,
                spaceBefore=2,
                spaceAfter=4,
                textColor=self.styles.body.textColor,
            )
            note_text = f"<i>Note.</i> {html.escape(note)}"
            self.elements.append(Paragraph(note_text, note_style))

        self.add_spacer(12)
        return self.table_counter

    def _add_native_table(self, str_data: list[list]):
        """Render table as native Platypus Table (selectable text).

        Uses dynamic column width calculation based on content length
        to avoid text cramming in headers or data cells.
        """

        if not str_data:
            return

        n_cols = max(len(row) for row in str_data) if str_data else 1
        cell_style = self.styles.body

        # Compute max content width per column (character count as proxy)
        max_lens = [0] * n_cols
        for row in str_data:
            for j, cell in enumerate(row):
                if j < n_cols:
                    max_lens[j] = max(max_lens[j], len(str(cell)))

        # Convert character counts to proportional widths with minimum
        # Use sqrt dampening so very long cells don't dominate
        import math
        dampened = [math.sqrt(max(ml, 3)) for ml in max_lens]
        total = sum(dampened)
        min_col = self.TEXT_WIDTH * 0.08  # Minimum 8% per column
        col_widths = [max(min_col, self.TEXT_WIDTH * d / total) for d in dampened]
        # Normalize to exactly fill TEXT_WIDTH
        scale = self.TEXT_WIDTH / sum(col_widths)
        col_widths = [w * scale for w in col_widths]

        # Build cell data — use Paragraph for first column if text is long,
        # raw strings for everything else (prevents word-break on numbers)
        para_data = []
        for i, row in enumerate(str_data):
            para_row = []
            for j, cell in enumerate(row):
                if j == 0 and len(cell) > 18:
                    if i == 0:
                        para_row.append(Paragraph(f"<b>{html.escape(cell)}</b>", cell_style))
                    else:
                        para_row.append(Paragraph(html.escape(cell), cell_style))
                else:
                    para_row.append(cell)
            para_data.append(para_row)

        tbl = Table(para_data, colWidths=col_widths, repeatRows=1)
        tbl.setStyle(self.styles.metrics_table_style())
        self.elements.append(tbl)

    def _add_matplotlib_table(self, str_data, cell_bg_colors, extra_rules, bold_cells):
        """Render table as matplotlib image (for heatmaps/complex styling)."""
        from ..charts.data_table import DataTableChart
        chart = DataTableChart()
        table_img = chart.generate(str_data, cell_bg_colors=cell_bg_colors,
                                    extra_rules=extra_rules, bold_cells=bold_cells)
        self.register_temp_file(table_img)

        img = Image(table_img)
        img_w, img_h = img.imageWidth, img.imageHeight
        if img_w > 0:
            ratio = self.TEXT_WIDTH / img_w
            scaled_h = img_h * ratio
            max_height = 550
            if scaled_h > max_height:
                ratio = max_height / img_h
            img.drawWidth = img_w * ratio
            img.drawHeight = img_h * ratio

        self.elements.append(img)

    def add_bullet_list(self, items: list[str]):
        """Add a bulleted list."""
        for item in items:
            bullet_text = f"\u2022  {item}"
            self.elements.append(Paragraph(bullet_text, self.styles.bullet))

    # ------------------------------------------------------------------
    # Math rendering (LaTeX via matplotlib mathtext)
    # ------------------------------------------------------------------

    def add_display_math(self, latex_str: str):
        """Render a LaTeX expression as a centered display equation.

        Uses SVG vector rendering via matplotlib + svglib for infinite
        resolution crispness — no raster pixelation at any zoom level.
        """
        drawing = self._render_math_vector(latex_str)
        if drawing is None:
            self.add_paragraph(f"<i>{html.escape(latex_str)}</i>", "body_italic")
            return
        # Center the equation horizontally
        drawing.hAlign = 'CENTER'
        self.add_spacer(4)
        self.elements.append(drawing)
        self.add_spacer(4)

    def _render_math_vector(self, latex_str: str, fontsize: float = 12,
                             scale: float = 1.0):
        """Render a LaTeX math expression to a ReportLab flowable.

        Primary pipeline: matplotlib mathtext → SVG → svglib → Drawing (vector).
        Fallback pipeline: matplotlib mathtext → PNG → reportlab Image (raster).
        The PNG fallback is used when svglib is unavailable.
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            matplotlib.rcParams['mathtext.fontset'] = 'cm'
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"  WARNING: matplotlib unavailable for '{latex_str[:40]}...': {e}")
            return None

        # Try SVG vector path first
        try:
            matplotlib.rcParams['svg.fonttype'] = 'path'
            from svglib.svglib import svg2rlg

            fig = plt.figure(figsize=(0.01, 0.01))
            fig.text(0, 0, f'${latex_str}$', fontsize=fontsize)
            buf = io.BytesIO()
            fig.savefig(buf, format='svg', bbox_inches='tight',
                        pad_inches=0.03, transparent=True)
            plt.close(fig)
            buf.seek(0)

            # svglib accepts any binary file-like object at runtime; its signature
            # only declares str | PathLike, so narrow the type for mypy here.
            drawing = svg2rlg(buf)  # type: ignore[arg-type]
            if drawing and scale != 1.0:
                drawing.width *= scale
                drawing.height *= scale
                drawing.scale(scale, scale)
            return drawing
        except ImportError:
            pass  # svglib not installed → fall through to PNG raster fallback
        except Exception as e:
            print(f"  WARNING: SVG math rendering failed for '{latex_str[:40]}...': {e}")

        # PNG raster fallback (always available with matplotlib)
        try:
            import tempfile

            from reportlab.lib.units import inch
            from reportlab.platypus import Image

            # Render at the body-text fontsize so the equation visually matches surrounding text.
            # 11pt body text → fontsize=11 yields a math expression at body height.
            fig = plt.figure(figsize=(0.01, 0.01))
            fig.text(0, 0, f'${latex_str}$', fontsize=fontsize)
            tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            tmp_path = tmp.name
            tmp.close()
            # 200 DPI is plenty for a non-stretched math image at native size
            fig.savefig(tmp_path, format='png', bbox_inches='tight',
                        pad_inches=0.02, transparent=False, dpi=200, facecolor='white')
            plt.close(fig)

            img = Image(tmp_path)
            # Use the NATURAL pixel size, converted at 200 DPI to inches → preserves
            # body-text-relative scale (no stretching, no pixelation, no oversize)
            DPI_OUT = 200.0
            natural_w_in = img.imageWidth / DPI_OUT
            natural_h_in = img.imageHeight / DPI_OUT
            # Cap at 4.5 inch wide for very long display equations
            max_w_in = 4.5 * scale
            if natural_w_in > max_w_in:
                shrink = max_w_in / natural_w_in
                natural_w_in *= shrink
                natural_h_in *= shrink
            img.drawWidth = natural_w_in * inch
            img.drawHeight = natural_h_in * inch
            return img
        except Exception as e:
            print(f"  WARNING: PNG fallback math rendering failed for '{latex_str[:40]}...': {e}")
            return None

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self):
        """Render all elements to the PDF file."""
        os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)

        # Deterministic output: reportlab's "invariant" mode pins the PDF's creation
        # timestamp and document /ID (otherwise set to wall-clock now() + a random id),
        # so the same inputs produce a byte-identical file -- no git churn on committed
        # theses. Only invisible metadata is affected; the rendered pages are unchanged.
        from reportlab import rl_config
        rl_config.invariant = 1

        doc = SimpleDocTemplate(
            self.output_path,
            pagesize=letter,
            rightMargin=1.0 * inch,
            leftMargin=1.0 * inch,
            topMargin=1.0 * inch,
            bottomMargin=0.75 * inch,
            title=self.title,
            author=self.author,
        )

        doc.build(
            self.elements,
            onFirstPage=self._first_page_footer,
            onLaterPages=self._later_pages_footer,
        )

        self._cleanup_temp_files()
        return self.output_path

    # ------------------------------------------------------------------
    # Page templates
    # ------------------------------------------------------------------

    @staticmethod
    def _first_page_footer(canvas, doc):
        """First page: no page number (title page)."""
        pass

    @staticmethod
    def _later_pages_footer(canvas, doc):
        """Later pages: centered page number at bottom."""
        canvas.saveState()
        canvas.setFont(FONT_REGULAR, 9)
        canvas.drawCentredString(
            doc.pagesize[0] / 2.0,
            0.5 * inch,
            str(doc.page),
        )
        canvas.restoreState()

    # ------------------------------------------------------------------
    # Temp file management
    # ------------------------------------------------------------------

    def register_temp_file(self, path: str):
        """Register a temp file for cleanup after build."""
        self._temp_files.append(path)

    def _cleanup_temp_files(self):
        """Remove temporary chart image files."""
        for f in self._temp_files:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except OSError:
                pass
        self._temp_files.clear()
