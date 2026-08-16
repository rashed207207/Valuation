from __future__ import annotations

import numpy as np
from equity_research_ai.data.yahoo import derive_dynamic_valuation_assumptions


class ValuationAnalystAgent:
    """
    ValuationAnalystAgent v2

    Converts:
    - historical financials
    - company lifecycle profile
    - SEC MD&A narrative evidence

    into transparent DCF assumptions.

    This version is still deterministic and explainable.
    It does not force the valuation to match market price.
    """

    def generate_assumptions(
        self,
        company_data: dict,
        profile: dict,
        business_analysis,
        base_erp: float = 0.045,
        risk_free_rate: float = 0.042,
    ):
        auto = derive_dynamic_valuation_assumptions(company_data)

        revenue = np.array(company_data["revenue_history"], dtype=float)
        ebit = np.array(company_data["ebit_history"], dtype=float)

        revenue_ttm = float(revenue[-1])
        current_margin = float(ebit[-1] / revenue[-1]) if revenue[-1] != 0 else 0.10

        revenue_cagr = self._safe_cagr(revenue)
        margin_history = ebit / revenue
        avg_margin = float(np.nanmean(margin_history))
        best_recent_margin = float(np.nanmax(margin_history[-3:]))

        narrative_text = self._collect_narrative_text(business_analysis)

        stage = profile.get("life_cycle_stage", "mature_stable")

        stage_growth_cap = {
            "startup": 0.25,
            "young_growth": 0.20,
            "high_growth": 0.16,
            "mature_growth": 0.11,
            "mature_stable": 0.075,
            "decline": 0.025,
        }.get(stage, 0.075)

        stage_growth_floor = {
            "startup": 0.08,
            "young_growth": 0.06,
            "high_growth": 0.05,
            "mature_growth": 0.035,
            "mature_stable": 0.015,
            "decline": -0.04,
        }.get(stage, 0.015)

        growth_bonus = 0.0
        margin_bonus = 0.0
        sales_to_capital_bonus = 0.0
        risk_penalty = 0.0

        # Growth drivers
        if self._has_any(narrative_text, ["services net sales increased", "services growth", "app store", "cloud services"]):
            growth_bonus += 0.012
            margin_bonus += 0.010
            sales_to_capital_bonus += 0.15

        if self._has_any(narrative_text, ["total net sales increased", "growth in all products and services"]):
            growth_bonus += 0.008

        if self._has_any(narrative_text, ["advertising", "subscription", "recurring"]):
            growth_bonus += 0.006
            margin_bonus += 0.006

        # Margin drivers
        if self._has_any(narrative_text, ["gross margin increased", "cost savings", "different products mix", "higher services net sales"]):
            margin_bonus += 0.012

        if self._has_any(narrative_text, ["productivity", "efficiency", "leverage"]):
            margin_bonus += 0.008

        # Capital intensity
        if self._has_any(narrative_text, ["services", "software", "app store", "cloud services", "advertising"]):
            sales_to_capital_bonus += 0.20

        if self._has_any(narrative_text, ["capital expenditures", "infrastructure", "manufacturing", "data center"]):
            sales_to_capital_bonus -= 0.15

        # Risk and headwinds
        if self._has_any(narrative_text, ["foreign currencies", "foreign exchange", "currency fluctuations"]):
            risk_penalty += 0.004

        if self._has_any(narrative_text, ["macroeconomic", "inflation", "interest rates"]):
            risk_penalty += 0.004

        if self._has_any(narrative_text, ["decreased", "decline", "weakness", "lower net sales"]):
            risk_penalty += 0.006

        # Outlook signals from BusinessAnalystAgent
        if business_analysis.growth_outlook == "positive":
            growth_bonus += 0.010
        elif business_analysis.growth_outlook == "negative":
            risk_penalty += 0.010

        if business_analysis.margin_outlook == "improving":
            margin_bonus += 0.010
        elif business_analysis.margin_outlook == "pressured":
            margin_bonus -= 0.010

        if business_analysis.capital_efficiency == "improving":
            sales_to_capital_bonus += 0.15
        elif business_analysis.capital_efficiency == "capital_intensive":
            sales_to_capital_bonus -= 0.15

        normalized_growth_base = max(revenue_cagr, stage_growth_floor)

        g1_begin = normalized_growth_base + growth_bonus - risk_penalty
        g1_begin = min(max(g1_begin, -0.08), stage_growth_cap)

        g1_end = min(max(g1_begin * 0.80, 0.00), 0.09)

        g2_begin = min(max(g1_end * 0.90, 0.00), 0.075)
        g2_end = min(max(risk_free_rate - 0.004, 0.018), 0.050)

        terminal_growth = min(risk_free_rate - 0.002, 0.040)
        terminal_growth = max(terminal_growth, 0.015)

        # Margin normalization
        base_terminal_margin = max(avg_margin, current_margin)

        # For very profitable companies, do not let one weak year dominate.
        if best_recent_margin > base_terminal_margin:
            base_terminal_margin = 0.50 * base_terminal_margin + 0.50 * best_recent_margin

        terminal_margin = base_terminal_margin + margin_bonus

        # Keep margins within plausible broad bounds.
        terminal_margin = min(max(terminal_margin, 0.03), 0.45)

        current_sales_to_capital = auto["current_sales_to_capital_ratio"]
        terminal_sales_to_capital = max(current_sales_to_capital + sales_to_capital_bonus, 0.5)

        # Avoid extreme sales-to-capital assumptions.
        terminal_sales_to_capital = min(terminal_sales_to_capital, 4.0)

        tax_marginal = 0.24 if company_data.get("country") == "United States" else 0.25

        market_cap = company_data["market_cap"]
        debt = company_data["total_debt"]
        beta = company_data.get("beta", 1.0) or 1.0

        unlevered_beta = beta / (1 + (1 - tax_marginal) * (debt / max(market_cap, 1e-9)))

        # Slight risk premium adjustment if filings show macro or FX risk.
        narrative_risk_erp = 0.002 if risk_penalty >= 0.008 else 0.0

        erp = base_erp + profile.get("country_risk_premium", 0) + narrative_risk_erp

        return {
            "risk_free_rate": risk_free_rate,
            "ERP": round(float(erp), 4),

            "equity_value": market_cap,
            "debt_value": debt,
            "cash_and_non_operating_asset": company_data.get("cash", 0.0),

            "unlevered_beta": round(float(unlevered_beta), 3),
            "terminal_unlevered_beta": round(float(unlevered_beta), 3),
            "year_beta_begins_to_converge_to_terminal_beta": 3,

            "current_pretax_cost_of_debt": auto["current_pretax_cost_of_debt"],
            "terminal_pretax_cost_of_debt": max(auto["current_pretax_cost_of_debt"] * 0.95, risk_free_rate + 0.005),
            "year_cost_of_debt_begins_to_converge_to_terminal_cost_of_debt": 3,

            "current_effective_tax_rate": auto["current_effective_tax_rate"],
            "marginal_tax_rate": tax_marginal,
            "year_effective_tax_rate_begin_to_converge_marginal_tax_rate": 2,

            "revenue_base": revenue_ttm,

            "revenue_growth_rate_cycle1_begin": round(float(g1_begin), 4),
            "revenue_growth_rate_cycle1_end": round(float(g1_end), 4),
            "revenue_growth_rate_cycle2_begin": round(float(g2_begin), 4),
            "revenue_growth_rate_cycle2_end": round(float(g2_end), 4),
            "revenue_growth_rate_cycle3_begin": round(float(g2_end), 4),
            "revenue_growth_rate_cycle3_end": round(float(terminal_growth), 4),

            "revenue_convergance_periods_cycle1": 1,
            "revenue_convergance_periods_cycle2": 1,
            "revenue_convergance_periods_cycle3": 1,

            "length_of_cylcle1": 3,
            "length_of_cylcle2": 4,
            "length_of_cylcle3": 3,

            "current_sales_to_capital_ratio": round(float(current_sales_to_capital), 3),
            "terminal_sales_to_capital_ratio": round(float(terminal_sales_to_capital), 3),
            "year_sales_to_capital_begins_to_converge_to_terminal_sales_to_capital": 2,

            "current_operating_margin": round(float(current_margin), 4),
            "terminal_operating_margin": round(float(terminal_margin), 4),
            "year_operating_margin_begins_to_converge_to_terminal_operating_margin": 2,

            "additional_return_on_cost_of_capital_in_perpetuity": 0.0,
            "asset_liquidation_during_negative_growth": 0,
            "current_invested_capital": "implicit",
        }

    def _safe_cagr(self, series):
        series = np.array(series, dtype=float)

        if len(series) < 2:
            return 0.03

        if series[0] <= 0:
            return 0.03

        return float((series[-1] / series[0]) ** (1 / (len(series) - 1)) - 1)

    def _collect_narrative_text(self, business_analysis):
        parts = []

        parts.extend(getattr(business_analysis, "drivers", []) or [])
        parts.extend(getattr(business_analysis, "risks", []) or [])

        raw_summary = getattr(business_analysis, "raw_summary", "")
        if raw_summary:
            parts.append(raw_summary)

        evidence = getattr(business_analysis, "evidence", []) or []
        for doc in evidence:
            parts.append(getattr(doc, "text", ""))

        return " ".join(parts).lower()

    def _has_any(self, text, phrases):
        return any(phrase.lower() in text for phrase in phrases)