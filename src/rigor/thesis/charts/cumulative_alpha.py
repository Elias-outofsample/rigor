"""Cumulative Alpha Chart -- strategy excess return trajectory."""

import tempfile

import pandas as pd

from .academic_style import AcademicChartStyle

# Major crisis dates for annotation
CRISIS_EVENTS = [
    (pd.Timestamp('2000-03-10'), 'Dotcom'),
    (pd.Timestamp('2008-09-15'), 'GFC'),
    (pd.Timestamp('2020-02-20'), 'COVID'),
]


class CumulativeAlphaChart:
    """Cumulative excess return (strategy - benchmark) over time."""

    def generate(self, strategy_returns: pd.Series,
                 benchmark_returns: pd.Series,
                 strategy_name: str = "Strategy",
                 benchmark_name: str = "S&P 500",
                 output_path: str | None = None) -> str:
        """Generate cumulative alpha chart. Returns path to saved PNG."""
        fig, ax = AcademicChartStyle.create_figure(width=10.0, height=5.5)

        # Align dates
        common = strategy_returns.index.intersection(benchmark_returns.index)
        strat = strategy_returns.loc[common]
        bench = benchmark_returns.loc[common]

        # Cumulative excess return
        excess = strat - bench
        cum_excess = (1 + excess).cumprod() - 1

        # Plot with green/red fill
        ax.fill_between(
            cum_excess.index, 0, cum_excess.values * 100,
            where=cum_excess.values >= 0,
            color=AcademicChartStyle.POSITIVE_COLOR,
            alpha=AcademicChartStyle.FILL_ALPHA,
            interpolate=True,
        )
        ax.fill_between(
            cum_excess.index, 0, cum_excess.values * 100,
            where=cum_excess.values < 0,
            color=AcademicChartStyle.NEGATIVE_COLOR,
            alpha=AcademicChartStyle.FILL_ALPHA,
            interpolate=True,
        )
        ax.plot(cum_excess.index, cum_excess.values * 100,
                color=AcademicChartStyle.STRATEGY_COLOR,
                linewidth=1.5,
                label=f'Cumulative Alpha ({strategy_name} \u2212 {benchmark_name})')

        ax.axhline(y=0, color='#999999', linestyle='-', linewidth=0.5)

        # Crisis date annotations
        date_min, date_max = cum_excess.index[0], cum_excess.index[-1]
        y_max = cum_excess.values.max() * 100
        for crisis_date, label in CRISIS_EVENTS:
            if date_min <= crisis_date <= date_max:
                ax.axvline(crisis_date, color='#999999', linestyle=':',
                           linewidth=0.7, alpha=0.6)
                ax.text(crisis_date, y_max * 0.95, f' {label}',
                        fontsize=7, rotation=90, va='top', ha='right',
                        color='#666666')

        ax.set_ylabel('Cumulative Excess Return (%)')
        ax.set_xlabel('')
        ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#cccccc')

        AcademicChartStyle.format_date_axis(ax)
        fig.tight_layout(pad=0.3)
        output_path = output_path or tempfile.mktemp(suffix='.png')
        return AcademicChartStyle.save_figure(fig, output_path)
