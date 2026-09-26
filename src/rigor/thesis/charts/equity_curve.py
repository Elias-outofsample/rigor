"""Cumulative returns chart (log scale, with benchmark overlay)."""

import tempfile

import pandas as pd

from .academic_style import AcademicChartStyle


class EquityCurveChart:

    def generate(self, equity: pd.Series, benchmark_equity: pd.Series | None = None,
                 strategy_name: str = "Strategy", benchmark_name: str = "S&P 500",
                 log_scale: bool = True, output_path: str | None = None) -> str:
        """Generate equity curve chart. Returns path to saved PNG."""
        fig, ax = AcademicChartStyle.create_figure(width=10.0, height=5.5)

        ax.plot(equity.index, equity.values,
                color=AcademicChartStyle.STRATEGY_COLOR,
                linewidth=2.0, label=strategy_name)

        if benchmark_equity is not None:
            ax.plot(benchmark_equity.index, benchmark_equity.values,
                    color=AcademicChartStyle.BENCHMARK_COLOR,
                    linewidth=1.0, linestyle='--', label=benchmark_name)

        if log_scale:
            ax.set_yscale('log')
            ax.set_ylabel('Portfolio Value (log scale)')
        else:
            ax.set_ylabel('Portfolio Value')

        ax.set_xlabel('')
        ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#cccccc')

        # Format y-axis with dollar signs
        ax.yaxis.set_major_formatter(
            lambda x, p: f'${x:,.0f}' if x >= 1 else f'${x:.2f}'
        )

        AcademicChartStyle.format_date_axis(ax)
        AcademicChartStyle.add_recession_shading(ax)
        AcademicChartStyle.add_source_note(fig)
        fig.tight_layout(pad=0.3)
        output_path = output_path or tempfile.mktemp(suffix='.png')
        return AcademicChartStyle.save_figure(fig, output_path)
