"""
Strategy Backtest Paper Type
=============================
Delegates to the existing ThesisGenerator for strategy-backtest papers.
This paper type is a bridge: if someone passes paper_type="strategy_backtest"
to GenThesisGenerator, it builds the standard strategy thesis structure.

For full strategy thesis generation, use the original ThesisGenerator directly.
This type provides a lighter-weight alternative using narrative text fields.
"""
from typing import TYPE_CHECKING

from .base_paper_type import BasePaperType

if TYPE_CHECKING:
    from ..gen_sections.base_gen_section import BaseGenSection
    from ..gen_thesis_config import GenThesisConfig


class StrategyBacktestPaperType(BasePaperType):

    name = "strategy_backtest"
    description = "Strategy backtest thesis (narrative mode, not data-driven)"

    @property
    def drafting_order(self) -> list[str]:
        return [
            "introduction",
            "related_literature",
            "data_methodology",
            "results_analysis",
            "robustness",
            "discussion",
            "conclusion",
            "abstract",
        ]

    def get_sections(self, config: "GenThesisConfig") -> list["BaseGenSection"]:
        from ..gen_sections.abstract_gen import AbstractGenSection
        from ..gen_sections.appendix_gen import AppendixGenSection
        from ..gen_sections.conclusion_gen import ConclusionGenSection
        from ..gen_sections.data_methodology_gen import DataMethodologyGenSection
        from ..gen_sections.discussion_gen import DiscussionGenSection
        from ..gen_sections.introduction_gen import IntroductionGenSection
        from ..gen_sections.references_gen import ReferencesGenSection
        from ..gen_sections.related_literature import RelatedLiteratureSection
        from ..gen_sections.results_analysis import ResultsAnalysisSection
        from ..gen_sections.robustness_gen import RobustnessGenSection
        from ..gen_sections.title_page_gen import TitlePageGenSection
        from ..gen_sections.toc_gen import TOCGenSection

        sections = [
            TitlePageGenSection(),
            AbstractGenSection(),
            TOCGenSection(),
            IntroductionGenSection(),
        ]

        if config.has_section("related_literature"):
            sections.append(RelatedLiteratureSection())

        sections.append(DataMethodologyGenSection())

        if config.has_section("results_analysis"):
            sections.append(ResultsAnalysisSection())

        if config.has_section("robustness"):
            sections.append(RobustnessGenSection())

        if config.has_section("discussion"):
            sections.append(DiscussionGenSection())

        sections += [
            ConclusionGenSection(),
            ReferencesGenSection(),
            AppendixGenSection(),
        ]

        return [s for s in sections if s.has_content(config)]
