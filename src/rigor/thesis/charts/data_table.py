"""Render data tables as matplotlib figures with academic booktabs styling.

Uses manual ax.text() + ax.plot() for precise control over font, spacing,
and horizontal rules — matching the metrics_comparison_table.png reference.
"""

import tempfile

import matplotlib

matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from .academic_style import AcademicChartStyle


class DataTableChart:
    """Render tabular data as a matplotlib figure matching academic style."""

    def generate(self, data: list, cell_bg_colors: dict | None = None,
                 font_size: float | None = None, output_path: str | None = None,
                 extra_rules: list | None = None, bold_cells: list | None = None) -> str:
        """Render table data as matplotlib figure with booktabs rules.

        Args:
            data: List of lists. First row is the header.
            cell_bg_colors: Optional dict of {(row_1based, col): rgba_tuple}
                for per-cell background colors (e.g. monthly returns heatmap).
            font_size: Override font size. Auto-selected if None.
            output_path: Path to save PNG. Auto-generated if None.
            extra_rules: Optional list of {'after_row': int, 'linewidth': float}
                to draw additional horizontal rules after specific body rows.
            bold_cells: Optional list of (row_1based, col) tuples for bold text.

        Returns:
            Path to saved PNG file.
        """
        AcademicChartStyle.apply()

        header = data[0]
        body = data[1:]
        n_rows = len(body)
        n_cols = len(header)
        total_rows = n_rows + 1  # header + body

        # Auto-select font size based on column count.
        # All tables use fig_width=10.0 since the PDF builder scales them to
        # ~6.4" text width. Fonts are sized to be readable after that scaling.
        wide_table = n_cols > 10
        header_size: float
        if font_size is None:
            if wide_table:
                font_size = 14.0
                header_size = 14.0
            elif n_cols > 6:
                font_size = 10.0
                header_size = 10.0
            else:
                font_size = 11.0
                header_size = 12.0
        else:
            header_size = font_size

        # --- Compute figure size from desired row height in inches ---
        row_h_inches = 0.32 if wide_table else 0.38
        margin_inches = 0.25
        fig_height = total_rows * row_h_inches + 2 * margin_inches
        fig_height = max(1.5, fig_height)
        fig_width = 10.0

        # Axis-coordinate row height (content fills proportionally)
        content_frac = (total_rows * row_h_inches) / fig_height
        row_h = content_frac / total_rows
        margin_frac = (1.0 - content_frac) / 2
        top_y = 1.0 - margin_frac

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.axis('off')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        header_bottom_y = top_y - row_h
        bottom_y = top_y - total_rows * row_h

        # --- Content-aware column widths ---
        # Measure max content length per column as width proxy
        max_lens = []
        for j in range(n_cols):
            col_lens = [len(str(header[j]))]
            for row in body:
                if j < len(row):
                    col_lens.append(len(str(row[j])))
            max_lens.append(max(col_lens))

        # Sqrt dampening prevents long-text columns from starving short ones
        dampened = [ml ** 0.5 for ml in max_lens]
        total_dampened = sum(dampened) or 1
        min_frac = 0.05
        col_widths = [max(min_frac, d / total_dampened) for d in dampened]
        frac_sum = sum(col_widths)
        col_widths = [w / frac_sum for w in col_widths]

        # Compute column left edges and centers within usable area
        pad_x = 0.01
        usable = 1.0 - 2 * pad_x
        col_left = []
        col_center = []
        col_abs_w = []
        x_acc = pad_x
        for j in range(n_cols):
            w = col_widths[j] * usable
            col_left.append(x_acc)
            col_center.append(x_acc + w / 2)
            col_abs_w.append(w)
            x_acc += w

        left_x, right_x = pad_x, 1.0 - pad_x

        # --- Draw cell backgrounds first (below text) ---
        if cell_bg_colors:
            for (row_idx, col_idx), color in cell_bg_colors.items():
                if row_idx < 1 or row_idx > n_rows or col_idx >= n_cols:
                    continue
                cell_top = top_y - row_idx * row_h
                cell_bot = cell_top - row_h
                cx = col_left[col_idx]
                cw = col_abs_w[col_idx]
                rect = mpatches.FancyBboxPatch(
                    (cx, cell_bot), cw, row_h,
                    boxstyle="square,pad=0", facecolor=color, edgecolor='none')
                ax.add_patch(rect)

        # --- Draw horizontal rules ---
        ax.plot([left_x, right_x], [top_y, top_y],
                'k-', linewidth=2.0, clip_on=False)
        ax.plot([left_x, right_x], [header_bottom_y, header_bottom_y],
                'k-', linewidth=0.75, clip_on=False)
        ax.plot([left_x, right_x], [bottom_y, bottom_y],
                'k-', linewidth=2.0, clip_on=False)

        # --- Draw extra rules (e.g. separator before difference row) ---
        if extra_rules:
            for rule in extra_rules:
                row_idx = rule['after_row']  # 1-based body row
                lw = rule.get('linewidth', 0.75)
                rule_y = header_bottom_y - row_idx * row_h
                ax.plot([left_x, right_x], [rule_y, rule_y],
                        'k-', linewidth=lw, clip_on=False)

        # --- Draw header text ---
        text_pad = 0.005  # small padding inside first column
        for j, label in enumerate(header):
            if j == 0:
                ax.text(col_left[0] + text_pad, top_y - row_h / 2, label,
                        ha='left', va='center',
                        fontweight='bold', fontsize=header_size)
            else:
                ax.text(col_center[j], top_y - row_h / 2, label,
                        ha='center', va='center',
                        fontweight='bold', fontsize=header_size)

        # --- Draw body rows ---
        bold_set = set(bold_cells) if bold_cells else set()
        for i, row in enumerate(body):
            y = header_bottom_y - (i + 0.5) * row_h
            for j, val in enumerate(row):
                weight = 'bold' if (i + 1, j) in bold_set else 'normal'
                if j == 0:
                    ax.text(col_left[0] + text_pad, y, str(val),
                            ha='left', va='center', fontsize=font_size,
                            fontweight=weight)
                else:
                    ax.text(col_center[j], y, str(val),
                            ha='center', va='center', fontsize=font_size,
                            fontweight=weight)

        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        output_path = output_path or tempfile.mktemp(suffix='.png')
        return AcademicChartStyle.save_figure(fig, output_path)
