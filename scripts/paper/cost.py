"""Tarifs LLM + accounting. Tous montants en USD par million de tokens.

Les tarifs sont figés — modifier à la main si un provider change. Le
smoke-test warn si le tarif vivant (retourné par l'API) diverge > 20% de
la valeur figée ici.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Price:
    input: float   # USD / 1M prompt tokens
    output: float  # USD / 1M completion tokens


PRICING: Dict[str, Price] = {
    "qwen/qwen3.6-plus":           Price(input=0.50, output=1.50),
    "deepseek/deepseek-v3.2":      Price(input=0.15, output=0.60),
    "codestral-latest":            Price(input=0.30, output=0.90),
    "anthropic/claude-sonnet-4-6": Price(input=3.00, output=15.00),
    "anthropic/claude-opus-4-7":   Price(input=15.00, output=75.00),
    "openai/gpt-5":                Price(input=1.25, output=10.00),
}


def estimate_cost_usd(
    model: str, prompt_tokens: int, completion_tokens: int,
) -> float:
    p = PRICING.get(model)
    if p is None:
        # Modèle inconnu → on ne plante pas, on retourne 0 et le smoke warn.
        return 0.0
    return (
        prompt_tokens * p.input + completion_tokens * p.output
    ) / 1_000_000.0
