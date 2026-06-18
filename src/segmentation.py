"""Translate a predicted P1-P4 risk tier into an underwriting recommendation.

This is the business layer on top of the classifier: the model outputs a tier,
and this module maps that tier to a lending decision and a suggested action.
"""
from __future__ import annotations

from . import config


def recommend(tier: str) -> dict:
    """Return the {risk, decision, action} policy for a P1-P4 tier."""
    tier = str(tier).upper()
    if tier not in config.RECOMMENDATIONS:
        raise ValueError(f"Unknown risk tier {tier!r}; expected one of {config.TIER_ORDER}")
    return config.RECOMMENDATIONS[tier]


def describe(tier: str) -> str:
    """One-line human-readable recommendation for a tier."""
    rec = recommend(tier)
    return f"{tier} ({rec['risk']}) -> {rec['decision']}: {rec['action']}"


def policy_table() -> list[dict]:
    """The full P1-P4 -> recommendation framework as a list of rows."""
    return [{"tier": t, **config.RECOMMENDATIONS[t]} for t in config.TIER_ORDER]


if __name__ == "__main__":
    for tier in config.TIER_ORDER:
        print(describe(tier))
