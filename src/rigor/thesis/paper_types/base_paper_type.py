"""Base class for paper type templates."""
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..gen_sections.base_gen_section import BaseGenSection
    from ..gen_thesis_config import GenThesisConfig


class BasePaperType(ABC):
    """
    Defines the ordered list of sections for a given paper type.
    Each PaperType knows which sections are canonical for its structure.
    """

    name: str = ""
    description: str = ""

    @abstractmethod
    def get_sections(self, config: "GenThesisConfig") -> list["BaseGenSection"]:
        """Return ordered list of active sections for this paper type."""
        pass

    # --- Drafting order (sections drafted left to right) ---
    @property
    def drafting_order(self) -> list[str]:
        """
        Section names in the order Claude should draft them.
        Abstract is drafted LAST (after all content sections exist).
        """
        return []
