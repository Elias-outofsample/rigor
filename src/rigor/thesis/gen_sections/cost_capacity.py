"""Type 3 — Economic Magnitude, Transaction Costs & Capacity section."""

from .base_gen_section import BaseGenSection


class CostCapacitySection(BaseGenSection):
    name = "Costs & Capacity"
    config_field = "cost_capacity"

    def render(self, builder, config):
        num = builder.next_section_number()
        builder.add_heading(
            f"{num}. Economic Magnitude, Transaction Costs and Capacity", level=1
        )
        self._render_paragraphs(builder, config.cost_capacity)
        self._render_section_figures(builder, config, "cost_capacity")
        self._render_section_tables(builder, config, "cost_capacity")
