"""
Empirical Study / Working Paper (Short)
========================================
Compact structure for shorter empirical papers.

Canonical section order:
  Title Page → Abstract → Introduction → Data & Methodology →
  Results & Analysis → Discussion → Conclusion → References
"""
from typing import TYPE_CHECKING

from .base_paper_type import BasePaperType

if TYPE_CHECKING:
    from ..gen_sections.base_gen_section import BaseGenSection
    from ..gen_thesis_config import GenThesisConfig


class EmpiricalStudyPaperType(BasePaperType):

    name = "empirical_study"
    description = "Short empirical study or working paper"

    @property
    def drafting_order(self) -> list[str]:
        return [
            "introduction",
            "data_methodology",
            "results_analysis",
            "discussion",
            "robustness",
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
        from ..gen_sections.results_analysis import ResultsAnalysisSection
        from ..gen_sections.robustness_gen import RobustnessGenSection
        from ..gen_sections.title_page_gen import TitlePageGenSection
        from ..gen_sections.toc_gen import TOCGenSection

        sections = [
            TitlePageGenSection(),
            AbstractGenSection(),
            TOCGenSection(),
            IntroductionGenSection(),
            DataMethodologyGenSection(),
            ResultsAnalysisSection(),
        ]

        if config.has_section("discussion"):
            sections.append(DiscussionGenSection())

        if config.has_section("robustness"):
            sections.append(RobustnessGenSection())

        sections += [
            ConclusionGenSection(),
            ReferencesGenSection(),
            AppendixGenSection(),
        ]

        return [s for s in sections if s.has_content(config)]
