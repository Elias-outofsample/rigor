"""Underwater equity (drawdown) chart."""

import tempfile

import pandas as pd

from .academic_style import AcademicChartStyle


class DrawdownChart:

    def generate(self, equity: pd.Series, benchmark_equity: pd.Series | None = None,
                 benchmark_name: str = "S&P 500", output_path: str | None = None) -> str:
        """Generate drawdown chart. Returns path to saved PNG."""
        fig, ax = AcademicChartStyle.create_figure(width=10.0, height=5.0)

        # Strategy drawdown
        rolling_max = equity.cummax()
        dd = (equity - rolling_max) / rolling_max * 100

        ax.fill_between(dd.index, dd.values, 0,
                        color=AcademicChartStyle.NEGATIVE_COLOR,
                        alpha=AcademicChartStyle.FILL_ALPHA)
        ax.plot(dd.index, dd.values,
                color=AcademicChartStyle.NEGATIVE_COLOR,
                linewidth=0.8, label='Strategy')

        # Benchmark drawdown
        if benchmark_equity is not None:
            bench_max = benchmark_equity.cummax()
            bench_dd = (benchmark_equity - bench_max) / bench_max * 100
            ax.plot(bench_dd.index, bench_dd.values,
                    color=AcademicChartStyle.BENCHMARK_COLOR,
                    linewidth=0.8, linestyle='--', label=benchmark_name)

        ax.set_ylabel('Drawdown (%)')
        ax.set_xlabel('')
        ax.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='#cccccc')
        ax.set_ylim(top=5)

        AcademicChartStyle.format_date_axis(ax)
        AcademicChartStyle.add_recession_shading(ax)
        AcademicChartStyle.add_source_note(fig)
        fig.tight_layout(pad=0.3)
        output_path = output_path or tempfile.mktemp(suffix='.png')
        return AcademicChartStyle.save_figure(fig, output_path)
