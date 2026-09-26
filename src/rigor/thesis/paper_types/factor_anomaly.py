"""
Factor / Anomaly Paper Type
============================
Structure: JF/RFS style for documenting a risk factor or market anomaly.

Canonical section order:
  Title Page → Abstract → Introduction → Related Literature →
  Theoretical Framework → Data & Methodology → Risk-Adjusted Results →
  Costs & Capacity → Robustness → Conclusion → References → Appendix
"""
from typing import TYPE_CHECKING

from .base_paper_type import BasePaperType

if TYPE_CHECKING:
    from ..gen_sections.base_gen_section import BaseGenSection
    from ..gen_thesis_config import GenThesisConfig


class FactorAnomalyPaperType(BasePaperType):

    name = "factor_anomaly"
    description = "Risk factor or market anomaly documentation (JF/RFS style)"

    @property
    def drafting_order(self) -> list[str]:
        return [
            "introduction",
            "related_literature",
            "theoretical_framework",
            "data_methodology",
            "risk_adjusted_results",
            "cost_capacity",
            "robustness",
            "conclusion",
            "abstract",
        ]

    def get_sections(self, config: "GenThesisConfig") -> list["BaseGenSection"]:
        from ..gen_sections.abstract_gen import AbstractGenSection
        from ..gen_sections.appendix_gen import AppendixGenSection
        from ..gen_sections.conclusion_gen import ConclusionGenSection
        from ..gen_sections.cost_capacity import CostCapacitySection
        from ..gen_sections.data_methodology_gen import DataMethodologyGenSection
        from ..gen_sections.introduction_gen import IntroductionGenSection
        from ..gen_sections.references_gen import ReferencesGenSection
        from ..gen_sections.related_literature import RelatedLiteratureSection
        from ..gen_sections.risk_adjusted_results import RiskAdjustedResultsSection
        from ..gen_sections.robustness_gen import RobustnessGenSection
        from ..gen_sections.theoretical_framework import TheoreticalFrameworkSection
        from ..gen_sections.title_page_gen import TitlePageGenSection
        from ..gen_sections.toc_gen import TOCGenSection

        sections = [
            TitlePageGenSection(),
            AbstractGenSection(),
            TOCGenSection(),
            IntroductionGenSection(),
            RelatedLiteratureSection(),
            TheoreticalFrameworkSection(),
            DataMethodologyGenSection(),
            RiskAdjustedResultsSection(),
        ]

        if config.has_section("cost_capacity"):
            sections.append(CostCapacitySection())

        if config.has_section("robustness"):
            sections.append(RobustnessGenSection())

        sections += [
            ConclusionGenSection(),
            ReferencesGenSection(),
            AppendixGenSection(),
        ]

        return [s for s in sections if s.has_content(config)]
