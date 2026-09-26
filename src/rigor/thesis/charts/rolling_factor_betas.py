"""Rolling Carhart factor loadings (3-panel chart)."""

import tempfile

import pandas as pd

from .academic_style import AcademicChartStyle


class RollingFactorBetasChart:

    def generate(self, rolling_betas: pd.DataFrame, output_path: str | None = None) -> str:
        """Generate 3-panel rolling factor beta chart.

        Args:
            rolling_betas: DataFrame with columns 'alpha', 'mkt_rf', 'smb', 'hml', 'mom'

        Returns:
            Path to saved PNG.
        """
        fig, (ax1, ax2, ax3) = AcademicChartStyle.create_figure(
            width=10.0, height=10.0, nrows=3, ncols=1)

        # Panel 1: Rolling Carhart Alpha
        alpha_pct = rolling_betas['alpha'].values * 100
        ax1.fill_between(rolling_betas.index, 0, alpha_pct,
                         where=alpha_pct >= 0,
                         color=AcademicChartStyle.POSITIVE_COLOR,
                         alpha=AcademicChartStyle.FILL_ALPHA, interpolate=True)
        ax1.fill_between(rolling_betas.index, 0, alpha_pct,
                         where=alpha_pct < 0,
                         color=AcademicChartStyle.NEGATIVE_COLOR,
                         alpha=AcademicChartStyle.FILL_ALPHA, interpolate=True)
        ax1.plot(rolling_betas.index, alpha_pct,
                 color=AcademicChartStyle.STRATEGY_COLOR, linewidth=1.5)
        ax1.axhline(y=0, color='#999999', linewidth=0.5)
        ax1.set_ylabel('Alpha (% ann.)')
        ax1.set_title('Rolling 36-Month Carhart Alpha', fontsize=11)

        # Panel 2: Rolling Market Beta
        ax2.plot(rolling_betas.index, rolling_betas['mkt_rf'].values,
                 color=AcademicChartStyle.STRATEGY_COLOR, linewidth=1.5)
        ax2.axhline(y=1.0, color='#999999', linestyle=':', linewidth=0.8,
                     label='Beta = 1')
        ax2.set_ylabel('Market Beta')
        ax2.set_title('Rolling 36-Month Market Beta', fontsize=11)
        ax2.legend(loc='upper right', frameon=True, facecolor='white',
                   edgecolor='#cccccc', fontsize=9)

        # Panel 3: Rolling Momentum Loading
        ax3.plot(rolling_betas.index, rolling_betas['mom'].values,
                 color='#E8910C', linewidth=1.5, label='MOM')
        ax3.axhline(y=0, color='#999999', linewidth=0.5)
        ax3.set_ylabel('Factor Loading')
        ax3.set_title('Rolling 36-Month Momentum Loading', fontsize=11)
        ax3.set_xlabel('')
        ax3.legend(loc='upper right', frameon=True, facecolor='white',
                   edgecolor='#cccccc', fontsize=9)

        for ax in [ax1, ax2, ax3]:
            AcademicChartStyle.format_date_axis(ax)
            AcademicChartStyle.add_recession_shading(ax)

        AcademicChartStyle.add_source_note(fig)
        fig.tight_layout(pad=0.3, h_pad=1.5)
        output_path = output_path or tempfile.mktemp(suffix='.png')
        return AcademicChartStyle.save_figure(fig, output_path)
