"""Rolling Sharpe ratio and rolling volatility (2-panel chart)."""

import tempfile

import numpy as np
import pandas as pd

from .academic_style import AcademicChartStyle


class RollingStatsChart:

    def generate(self, rolling_sharpe: pd.Series, rolling_vol: pd.Series,
                 bench_returns: pd.Series | None = None,
                 window: int = 756, output_path: str | None = None,
                 rf_annual: float = 0.02) -> str:
        """Generate 2-panel rolling stats chart. Returns path to saved PNG."""
        fig, (ax1, ax2) = AcademicChartStyle.create_figure(
            width=10.0, height=9.0, nrows=2, ncols=1)

        # Panel 1: Rolling Sharpe
        ax1.plot(rolling_sharpe.index, rolling_sharpe.values,
                 color=AcademicChartStyle.STRATEGY_COLOR,
                 linewidth=1.5, label='Strategy')
        ax1.axhline(y=1.0, color='#999999', linestyle=':', linewidth=0.8, label='Sharpe = 1')
        ax1.axhline(y=0, color='#cccccc', linestyle='-', linewidth=0.5)

        if bench_returns is not None and len(bench_returns) >= window:
            trading_days = 252
            bench_roll_mean = bench_returns.rolling(window).mean() * trading_days
            bench_roll_std = bench_returns.rolling(window).std() * np.sqrt(trading_days)
            bench_sharpe = ((bench_roll_mean - rf_annual) / bench_roll_std).dropna()
            ax1.plot(bench_sharpe.index, bench_sharpe.values,
                     color=AcademicChartStyle.BENCHMARK_COLOR,
                     linewidth=0.8, linestyle='--', label='Benchmark')

        ax1.set_ylabel('Rolling Sharpe')
        ax1.set_title('Rolling Sharpe Ratio (36-month)', fontsize=11)
        ax1.legend(loc='upper right', frameon=True, facecolor='white',
                   edgecolor='#cccccc', fontsize=9)

        # Panel 2: Rolling Volatility
        ax2.plot(rolling_vol.index, rolling_vol.values * 100,
                 color=AcademicChartStyle.STRATEGY_COLOR,
                 linewidth=1.5, label='Strategy')

        if bench_returns is not None and len(bench_returns) >= window:
            bench_vol = (bench_returns.rolling(window).std() * np.sqrt(252) * 100).dropna()
            ax2.plot(bench_vol.index, bench_vol.values,
                     color=AcademicChartStyle.BENCHMARK_COLOR,
                     linewidth=0.8, linestyle='--', label='Benchmark')

        ax2.set_ylabel('Annualized Volatility (%)')
        ax2.set_title('Rolling Volatility (36-month)', fontsize=11)
        ax2.set_xlabel('')
        ax2.legend(loc='upper right', frameon=True, facecolor='white',
                   edgecolor='#cccccc', fontsize=9)

        AcademicChartStyle.format_date_axis(ax1)
        AcademicChartStyle.format_date_axis(ax2)
        AcademicChartStyle.add_recession_shading(ax1)
        AcademicChartStyle.add_recession_shading(ax2)
        AcademicChartStyle.add_source_note(fig)
        fig.tight_layout(pad=0.3, h_pad=1.5)
        output_path = output_path or tempfile.mktemp(suffix='.png')
        return AcademicChartStyle.save_figure(fig, output_path)
