from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

@dataclass
class RetrievedDocument:
    text: str
    ticker: str | None = None
    company: str | None = None
    fiscal_year: int | str | None = None
    doc_id: str | None = None
    paragraph_id: str | None = None
    score: float | None = None

@dataclass
class BusinessAnalysis:
    ticker: str
    drivers: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    growth_outlook: str = "neutral"
    margin_outlook: str = "stable"
    capital_efficiency: str = "stable"
    evidence: list[RetrievedDocument] = field(default_factory=list)
    raw_summary: str = ""

@dataclass
class CompanyAnalysis:
    ticker: str
    company_name: str = ""
    company_data: dict[str, Any] = field(default_factory=dict)
    profile: dict[str, Any] = field(default_factory=dict)
    valuation_decision: dict[str, Any] = field(default_factory=dict)
    business_analysis: BusinessAnalysis | None = None
    valuation_assumptions: dict[str, Any] = field(default_factory=dict)
    dcf_result: dict[str, Any] = field(default_factory=dict)
    monte_carlo_result: dict[str, Any] = field(default_factory=dict)
    implied_market_expectations: dict[str, Any] = field(default_factory=dict)
    final_report: str = ""
