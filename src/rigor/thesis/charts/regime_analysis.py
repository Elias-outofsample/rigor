"""Regime analysis grouped bar chart."""

import tempfile

import numpy as np
import pandas as pd

from .academic_style import AcademicChartStyle


class RegimeAnalysisChart:

    def generate(self, regime_df: pd.DataFrame, output_path: str | None = None) -> str:
        """Generate regime performance bar chart. Returns path to saved PNG.

        regime_df: DataFrame with index = regime names,
                   columns include 'Ann. Return', 'Sharpe'.
        """
        fig, (ax1, ax2) = AcademicChartStyle.create_figure(
            width=10.0, height=5.0, nrows=1, ncols=2)

        regimes = regime_df.index.tolist()
        x = np.arange(len(regimes))
        bar_width = 0.5

        colors = [AcademicChartStyle.POSITIVE_COLOR if v >= 0
                  else AcademicChartStyle.NEGATIVE_COLOR
                  for v in regime_df['Ann. Return']]

        # Panel 1: Annualized Return
        ax1.bar(x, regime_df['Ann. Return'].values * 100,
                width=bar_width, color=colors, edgecolor='white')
        ax1.set_ylabel('Annualized Return (%)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(regimes, fontsize=9)
        ax1.axhline(y=0, color='#999999', linewidth=0.5)
        ax1.set_title('Returns by Regime', fontsize=11)

        # Panel 2: Sharpe
        colors_s = [AcademicChartStyle.POSITIVE_COLOR if v >= 0
                    else AcademicChartStyle.NEGATIVE_COLOR
                    for v in regime_df['Sharpe']]
        ax2.bar(x, regime_df['Sharpe'].values,
                width=bar_width, color=colors_s, edgecolor='white')
        ax2.set_ylabel('Sharpe Ratio')
        ax2.set_xticks(x)
        ax2.set_xticklabels(regimes, fontsize=9)
        ax2.axhline(y=0, color='#999999', linewidth=0.5)
        ax2.set_title('Risk-Adjusted Returns by Regime', fontsize=11)

        fig.tight_layout(pad=0.3, w_pad=2.0)
        output_path = output_path or tempfile.mktemp(suffix='.png')
        return AcademicChartStyle.save_figure(fig, output_path)
