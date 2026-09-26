"""Monthly return distribution histogram with normal overlay."""

import tempfile

import numpy as np
import pandas as pd
from scipy import stats

from .academic_style import AcademicChartStyle


class ReturnDistributionChart:

    def generate(self, returns: pd.Series, output_path: str | None = None) -> str:
        """Generate return distribution histogram. Returns path to saved PNG."""
        # Convert to monthly returns
        monthly = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
        monthly_pct = monthly * 100

        fig, ax = AcademicChartStyle.create_figure(width=10.0, height=5.5)

        # Histogram
        n, bins, patches = ax.hist(
            monthly_pct.dropna(), bins=30,
            color=AcademicChartStyle.STRATEGY_COLOR,
            alpha=0.6, edgecolor='white', linewidth=0.5,
            density=True, label='Monthly Returns',
        )

        # Normal distribution overlay
        mu = monthly_pct.mean()
        sigma = monthly_pct.std()
        x = np.linspace(monthly_pct.min() - 5, monthly_pct.max() + 5, 200)
        normal = stats.norm.pdf(x, mu, sigma)
        ax.plot(x, normal, color=AcademicChartStyle.NEGATIVE_COLOR,
                linewidth=1.5, linestyle='--', label='Normal Distribution')

        # Annotate statistics
        skew = monthly_pct.skew()
        kurt = monthly_pct.kurtosis()
        stats_text = (f'Mean: {mu:.1f}%\nStd: {sigma:.1f}%\n'
                      f'Skew: {skew:.2f}\nKurtosis: {kurt:.2f}')
        ax.text(0.97, 0.95, stats_text, transform=ax.transAxes,
                fontsize=8, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          edgecolor='#cccccc', alpha=0.9),
                fontfamily='serif')

        ax.set_xlabel('Monthly Return (%)')
        ax.set_ylabel('Density')
        ax.legend(loc='upper left', frameon=True, facecolor='white',
                  edgecolor='#cccccc', fontsize=9)

        fig.tight_layout(pad=0.3)
        output_path = output_path or tempfile.mktemp(suffix='.png')
        return AcademicChartStyle.save_figure(fig, output_path)
