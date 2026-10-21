from __future__ import annotations

from .data import FACT_VERSION, Fact


class FactLookupError(ValueError):
    pass


class FactStore:
    def __init__(self, facts: list[Fact]) -> None:
        self._by_brand: dict[str, list[Fact]] = {}
        for fact in facts:
            self._by_brand.setdefault(fact.brand_id, []).append(fact)

    @property
    def brands(self) -> set[str]:
        return set(self._by_brand)

    def get_brand_facts(self, brand_id: str) -> dict[str, object]:
        if brand_id not in self._by_brand:
            raise FactLookupError(f"Unknown brand: {brand_id}")
        return {
            "fact_version": FACT_VERSION,
            "brand_id": brand_id,
            "facts": [fact.as_dict() for fact in self._by_brand[brand_id]],
        }

    def known_fact_ids(self) -> set[str]:
        return {fact.fact_id for facts in self._by_brand.values() for fact in facts}
