"""Rolling annualized CAPM alpha chart."""

import tempfile

import pandas as pd

from .academic_style import AcademicChartStyle


class RollingAlphaChart:

    def generate(self, rolling_alpha: pd.Series, output_path: str | None = None) -> str:
        """Generate rolling alpha chart with green/red fill. Returns path to saved PNG."""
        fig, ax = AcademicChartStyle.create_figure(width=10.0, height=5.0)

        alpha_pct = rolling_alpha.values * 100

        ax.fill_between(rolling_alpha.index, 0, alpha_pct,
                        where=alpha_pct >= 0,
                        color=AcademicChartStyle.POSITIVE_COLOR,
                        alpha=AcademicChartStyle.FILL_ALPHA,
                        interpolate=True)
        ax.fill_between(rolling_alpha.index, 0, alpha_pct,
                        where=alpha_pct < 0,
                        color=AcademicChartStyle.NEGATIVE_COLOR,
                        alpha=AcademicChartStyle.FILL_ALPHA,
                        interpolate=True)

        ax.plot(rolling_alpha.index, alpha_pct,
                color=AcademicChartStyle.STRATEGY_COLOR,
                linewidth=1.5)

        ax.axhline(y=0, color='#999999', linewidth=0.5)
        ax.set_ylabel('Rolling Alpha (% annualized)')
        ax.set_xlabel('')

        AcademicChartStyle.format_date_axis(ax)
        AcademicChartStyle.add_recession_shading(ax)
        AcademicChartStyle.add_source_note(fig)
        fig.tight_layout(pad=0.3)
        output_path = output_path or tempfile.mktemp(suffix='.png')
        return AcademicChartStyle.save_figure(fig, output_path)
