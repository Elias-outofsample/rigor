"""Color-coded monthly returns heatmap as matplotlib table."""

import tempfile

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from .academic_style import AcademicChartStyle


class MonthlyReturnsChart:

    def generate(self, monthly_pivot: pd.DataFrame, output_path: str | None = None) -> str:
        """Generate monthly returns heatmap. Returns path to saved PNG."""
        AcademicChartStyle.apply()

        n_rows = len(monthly_pivot)
        n_cols = len(monthly_pivot.columns)

        # Scale figure height by number of years
        fig_height = max(3.5, 0.32 * n_rows + 1.5)
        fig, ax = plt.subplots(figsize=(6.5, fig_height))
        ax.axis('off')

        # Create color map: red -> white -> green
        cmap = LinearSegmentedColormap.from_list(
            'rg', ['#9b2226', '#ffffff', '#2d6a4f'], N=256)

        # Build cell text and colors
        cell_text = []
        cell_colors = []
        vmax = max(abs(monthly_pivot.values[~np.isnan(monthly_pivot.values)].min()),
                   abs(monthly_pivot.values[~np.isnan(monthly_pivot.values)].max()),
                   0.01)

        for _, row in monthly_pivot.iterrows():
            row_text = []
            row_colors = []
            for val in row:
                if pd.isna(val):
                    row_text.append('')
                    row_colors.append('#ffffff')
                else:
                    row_text.append(f'{val * 100:.1f}%')
                    # Normalize to 0-1 range for colormap
                    norm_val = (val / vmax + 1) / 2
                    norm_val = max(0, min(1, norm_val))
                    rgba = cmap(norm_val)
                    row_colors.append(rgba)
            cell_text.append(row_text)
            cell_colors.append(row_colors)

        row_labels = [str(y) for y in monthly_pivot.index]
        col_labels = list(monthly_pivot.columns)

        table = ax.table(
            cellText=cell_text,
            cellColours=cell_colors,
            rowLabels=row_labels,
            colLabels=col_labels,
            loc='center',
            cellLoc='center',
        )

        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.3)

        # Style header row
        for j in range(n_cols):
            cell = table[0, j]
            cell.set_text_props(fontweight='bold', fontfamily='serif', fontsize=8)
            cell.set_facecolor('#f0f0f0')
            cell.set_edgecolor('#cccccc')

        # Style row labels
        for i in range(n_rows):
            cell = table[i + 1, -1]
            cell.set_text_props(fontweight='bold', fontfamily='serif', fontsize=8)
            cell.set_facecolor('#f0f0f0')
            cell.set_edgecolor('#cccccc')

        # Style all data cells
        for i in range(n_rows):
            for j in range(n_cols):
                cell = table[i + 1, j]
                cell.set_edgecolor('#e0e0e0')
                cell.set_text_props(fontfamily='serif', fontsize=7.5)

        fig.tight_layout()
        output_path = output_path or tempfile.mktemp(suffix='.png')
        return AcademicChartStyle.save_figure(fig, output_path)
