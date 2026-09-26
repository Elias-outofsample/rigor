"""
Academic Chart Style
====================
Centralized matplotlib configuration for publication-quality finance charts.
"""

import tempfile

import matplotlib

matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


class AcademicChartStyle:
    """Matplotlib style for academic finance papers.

    White background, serif fonts, light gray grid, no top/right spines.
    Muted, print-friendly color palette.
    """

    # Color palette — house standard
    STRATEGY_COLOR = '#1f4e79'    # Dark blue
    BENCHMARK_COLOR = '#808080'   # Gray
    POSITIVE_COLOR = '#2d6a4f'    # Dark green
    NEGATIVE_COLOR = '#9b2226'    # Dark red
    ACCENT_COLOR = '#e9c46a'      # Muted gold
    FILL_ALPHA = 0.3

    # Extended palette (8 colors, colorblind-friendly, B&W safe)
    PALETTE = [
        '#1f4e79',  # Dark blue (strategy)
        '#9b2226',  # Dark red (negative)
        '#2d6a4f',  # Dark green (positive)
        '#e9c46a',  # Muted gold (accent)
        '#6a4c93',  # Purple
        '#1982c4',  # Medium blue
        '#f4a261',  # Burnt orange
        '#808080',  # Gray (benchmark)
    ]

    # Quintile-specific palette (Q1=calm → Q5=extreme)
    QUINTILE_PALETTE = [
        '#4e79a7',  # Q1 — medium blue (calm)
        '#76b7b2',  # Q2 — teal
        '#edc949',  # Q3 — gold
        '#e15759',  # Q4 — coral red
        '#9b2226',  # Q5 — dark red (extreme)
    ]

    # Intraday phase colors
    PHASE_COLORS = {
        'I':   '#1f4e79',  # Dark blue — Uncertainty Dissipation
        'II':  '#2d6a4f',  # Dark green — Premium Reloading
        'III': '#9b2226',  # Dark red — Terminal Collapse
    }

    FIGURE_DPI = 600

    # NBER recession date ranges for shading
    NBER_RECESSIONS = [
        ('2001-03-01', '2001-11-30'),    # Dotcom recession
        ('2007-12-01', '2009-06-30'),    # Great Recession
        ('2020-02-01', '2020-04-30'),    # COVID recession
    ]

    @classmethod
    def apply(cls):
        """Apply academic style to matplotlib rcParams.

        Font sizes are calibrated for charts generated at 10" width
        and embedded at ~6.3" in a letter-size PDF (0.63x scale).
        A font.size of 14 here becomes ~9pt in the final PDF.
        """
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
            'font.size': 14,
            'axes.titlesize': 16,
            'axes.labelsize': 15,
            'xtick.labelsize': 13,
            'ytick.labelsize': 13,
            'legend.fontsize': 13,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linestyle': '--',
            'grid.color': '#cccccc',
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.edgecolor': '#333333',
            'lines.linewidth': 2.0,
            'figure.dpi': cls.FIGURE_DPI,
            'savefig.dpi': cls.FIGURE_DPI,
            'savefig.facecolor': 'white',
        })

    @classmethod
    def create_figure(cls, width: float = 10.0, height: float = 5.0,
                      nrows: int = 1, ncols: int = 1):
        """Create a figure with standard academic sizing."""
        cls.apply()
        fig, axes = plt.subplots(nrows, ncols, figsize=(width, height))
        return fig, axes

    @classmethod
    def format_date_axis(cls, ax):
        """Apply clean date formatting to a time-series x-axis.

        Automatically selects tick interval based on the visible date range.
        """
        xlim = ax.get_xlim()
        date_range_years = (xlim[1] - xlim[0]) / 365.25

        if date_range_years > 15:
            ax.xaxis.set_major_locator(mdates.YearLocator(5))
            ax.xaxis.set_minor_locator(mdates.YearLocator(1))
        elif date_range_years > 5:
            ax.xaxis.set_major_locator(mdates.YearLocator(2))
            ax.xaxis.set_minor_locator(mdates.YearLocator(1))
        elif date_range_years > 2:
            ax.xaxis.set_major_locator(mdates.YearLocator(1))
        else:
            ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.tick_params(axis='x', which='minor', length=3, width=0.5)
        ax.tick_params(axis='x', which='major', length=6, width=0.8)

    @classmethod
    def add_recession_shading(cls, ax):
        """Add light gray NBER recession shading to a time-series axis."""
        for start, end in cls.NBER_RECESSIONS:
            start_dt = pd.Timestamp(start)
            end_dt = pd.Timestamp(end)
            try:
                xlim = ax.get_xlim()
                if start_dt.toordinal() <= xlim[1] and end_dt.toordinal() >= xlim[0]:
                    ax.axvspan(start_dt, end_dt, alpha=0.08, color='#333333',
                               zorder=0, label='_nolegend_')
            except Exception:
                pass

    @classmethod
    def add_source_note(cls, fig, text="Source: Author's calculations."):
        """Add a source note below the figure."""
        fig.text(0.99, -0.02, text, ha='right', va='top',
                 fontsize=7, fontstyle='italic', color='#666666',
                 fontfamily='serif')

    @classmethod
    def save_figure(cls, fig, filepath: str | None = None) -> str:
        """Save figure to file. Returns the filepath."""
        if filepath is None:
            filepath = tempfile.mktemp(suffix='.png')
        fig.savefig(filepath, dpi=cls.FIGURE_DPI, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig)
        return filepath
