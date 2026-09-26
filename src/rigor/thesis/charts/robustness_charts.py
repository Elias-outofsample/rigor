"""Robustness charts: walk-forward analysis bars."""

import tempfile

import numpy as np

from .academic_style import AcademicChartStyle


class RobustnessCharts:

    def generate_walk_forward(self, quality_report: dict,
                              output_path: str | None = None) -> str:
        """Generate walk-forward IS vs OOS bar chart from quality report data.

        Returns path to saved PNG, or empty string if data unavailable.
        """
        # Try to extract walk-forward data from quality report
        wf_data = self._extract_walk_forward(quality_report)
        if wf_data is None:
            return ''

        fig, ax = AcademicChartStyle.create_figure(width=10.0, height=5.0)

        categories = list(wf_data.keys())
        is_vals = [wf_data[c].get('is_sharpe', 0) for c in categories]
        oos_vals = [wf_data[c].get('oos_sharpe', 0) for c in categories]

        x = np.arange(len(categories))
        width = 0.35

        ax.bar(x - width / 2, is_vals, width,
               color=AcademicChartStyle.STRATEGY_COLOR,
               label='In-Sample', alpha=0.8)
        ax.bar(x + width / 2, oos_vals, width,
               color=AcademicChartStyle.ACCENT_COLOR,
               label='Out-of-Sample', alpha=0.8)

        ax.set_ylabel('Sharpe Ratio')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=9, rotation=45, ha='right')
        ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=9)
        ax.axhline(y=0, color='#999999', linewidth=0.5)

        fig.tight_layout(pad=0.3)
        output_path = output_path or tempfile.mktemp(suffix='.png')
        return AcademicChartStyle.save_figure(fig, output_path)

    def generate_quality_summary(self, quality_report: dict,
                                 output_path: str | None = None) -> str:
        """Generate quality category scores bar chart.

        Returns path to saved PNG, or empty string if data unavailable.
        """
        categories = self._extract_category_scores(quality_report)
        if not categories:
            return ''

        fig, ax = AcademicChartStyle.create_figure(width=10.0, height=5.0)

        names = list(categories.keys())
        scores = list(categories.values())
        x = np.arange(len(names))

        colors = [AcademicChartStyle.POSITIVE_COLOR if s >= 70
                  else AcademicChartStyle.ACCENT_COLOR if s >= 50
                  else AcademicChartStyle.NEGATIVE_COLOR
                  for s in scores]

        ax.barh(x, scores, color=colors, edgecolor='white', height=0.6)
        ax.set_yticks(x)
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel('Score')
        ax.set_xlim(0, 100)
        ax.axvline(x=70, color='#999999', linewidth=0.8, linestyle=':', label='Pass (70)')
        ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=9)

        fig.tight_layout(pad=0.3)
        output_path = output_path or tempfile.mktemp(suffix='.png')
        return AcademicChartStyle.save_figure(fig, output_path)

    def _extract_walk_forward(self, quality_report: dict) -> dict | None:
        """Extract walk-forward IS/OOS data from quality report JSON."""
        if not quality_report:
            return None

        # Look for walk-forward or cross-validation results
        results = quality_report.get('results', quality_report.get('test_results', {}))
        if isinstance(results, list):
            for r in results:
                name = r.get('name', '').lower()
                if 'walk' in name and 'forward' in name:
                    details = r.get('details', {})
                    if 'folds' in details:
                        wf = {}
                        for i, fold in enumerate(details['folds']):
                            wf[f'Fold {i + 1}'] = {
                                'is_sharpe': fold.get('is_sharpe', fold.get('train_sharpe', 0)),
                                'oos_sharpe': fold.get('oos_sharpe', fold.get('test_sharpe', 0)),
                            }
                        if wf:
                            return wf
        return None

    def _extract_category_scores(self, quality_report: dict) -> dict:
        """Extract category-level scores from quality report JSON."""
        if not quality_report:
            return {}

        # Try different JSON schema formats
        categories = quality_report.get('category_scores', {})
        if categories:
            return {k: v.get('score', v) if isinstance(v, dict) else v
                    for k, v in categories.items()}

        # Alternative: look in summary
        summary = quality_report.get('summary', {})
        if 'category_scores' in summary:
            return summary['category_scores']

        return {}
