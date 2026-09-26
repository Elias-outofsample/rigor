"""
Market Phenomenon Paper Type
=============================
Structure: JFE/RFS style for documenting an empirical market phenomenon.

Canonical section order:
  Title Page → Abstract → Introduction → Institutional Background →
  Data & Methodology → Stylized Facts → Economic Mechanism →
  Robustness → Implications → Conclusion → References → Appendix
"""
from typing import TYPE_CHECKING

from .base_paper_type import BasePaperType

if TYPE_CHECKING:
    from ..gen_sections.base_gen_section import BaseGenSection
    from ..gen_thesis_config import GenThesisConfig


class MarketPhenomenonPaperType(BasePaperType):

    name = "market_phenomenon"
    description = "Empirical documentation of a market phenomenon (JFE/RFS style)"

    @property
    def drafting_order(self) -> list[str]:
        # Draft content first, abstract last (it summarizes everything)
        return [
            "introduction",
            "institutional_background",
            "data_methodology",
            "stylized_facts",
            "mechanism",
            "robustness",
            "implications",
            "conclusion",
            "abstract",      # Last — summarizes all the above
        ]

    def get_sections(self, config: "GenThesisConfig") -> list["BaseGenSection"]:
        from ..gen_sections.abstract_gen import AbstractGenSection
        from ..gen_sections.appendix_gen import AppendixGenSection
        from ..gen_sections.conclusion_gen import ConclusionGenSection
        from ..gen_sections.data_methodology_gen import DataMethodologyGenSection
        from ..gen_sections.implications import ImplicationsSection
        from ..gen_sections.institutional_background import InstitutionalBackgroundSection
        from ..gen_sections.introduction_gen import IntroductionGenSection
        from ..gen_sections.mechanism import MechanismSection
        from ..gen_sections.references_gen import ReferencesGenSection
        from ..gen_sections.robustness_gen import RobustnessGenSection
        from ..gen_sections.stylized_facts import StylizedFactsSection
        from ..gen_sections.title_page_gen import TitlePageGenSection
        from ..gen_sections.toc_gen import TOCGenSection

        sections = [
            TitlePageGenSection(),
            AbstractGenSection(),
            TOCGenSection(),
            IntroductionGenSection(),
            InstitutionalBackgroundSection(),
            DataMethodologyGenSection(),
            StylizedFactsSection(),
            MechanismSection(),
            RobustnessGenSection(),
        ]

        if config.has_section("implications"):
            sections.append(ImplicationsSection())

        sections += [
            ConclusionGenSection(),
            ReferencesGenSection(),
        ]

        if config.include_appendix if hasattr(config, "include_appendix") else True:
            sections.append(AppendixGenSection())

        return [s for s in sections if s.has_content(config)]
