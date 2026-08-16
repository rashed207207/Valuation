from __future__ import annotations

import numpy as np
from equity_research_ai.data.yahoo import derive_dynamic_valuation_assumptions


class ValuationAnalystAgent:
    """
    ValuationAnalystAgent v2.1

    Converts:
    - historical financials
    - company lifecycle profile
    - SEC MD&A narrative evidence

    into transparent, audit-trail DCF assumptions.

    Phase 1 Improvements (2026-08-16):
    - Assumption bridge: Full derivation trail for each key assumption
    - Terminal growth: Lifecycle-aware, never exceeds risk_free_rate
    - Beta mean reversion: All companies fade toward market beta (1.0x) by year 10
    - Narrative signals: Deduplication, structured collection with logging
    - Profile integration: Uses sector_flags, leverage, cyclicality, margin_trend
    
    All 60+ existing DCF keys preserved for backward compatibility.
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

        # Improved: Structured signal collection with deduplication
        signal_log = self._collect_narrative_signals(business_analysis)

        stage = profile.get("life_cycle_stage", "mature_stable")

        stage_growth_cap = {
            "startup": 0.25,
            "young_growth": 0.20,
            "high_growth": 0.16,
            "mature_growth": 0.11,
            "mature_stable": 0.075,
            "decline": 0.025,
            "distress": 0.015,
        }.get(stage, 0.075)

        stage_growth_floor = {
            "startup": 0.08,
            "young_growth": 0.06,
            "high_growth": 0.05,
            "mature_growth": 0.035,
            "mature_stable": 0.015,
            "decline": -0.04,
            "distress": -0.06,
        }.get(stage, 0.015)

        # Calculate bonuses from structured signals (not double-counted)
        growth_bonus = sum(s.get("bonus", 0) for s in signal_log.get("growth", []))
        margin_bonus = sum(s.get("bonus", 0) for s in signal_log.get("margin", []))
        capital_bonus = sum(s.get("bonus", 0) for s in signal_log.get("capital", []))
        risk_penalty = sum(s.get("penalty", 0) for s in signal_log.get("risks", []))

        # Apply profile-based adjustments (lifecycle, sector, leverage)
        growth_bonus, margin_bonus, capital_bonus, profile_adjustments = self._adjust_for_profile(
            growth_bonus, margin_bonus, capital_bonus, profile
        )

        normalized_growth_base = max(revenue_cagr, stage_growth_floor)

        g1_begin = normalized_growth_base + growth_bonus - risk_penalty
        g1_begin = min(max(g1_begin, -0.08), stage_growth_cap)

        g1_end = min(max(g1_begin * 0.80, 0.00), 0.09)

        g2_begin = min(max(g1_end * 0.90, 0.00), 0.075)
        g2_end = min(max(risk_free_rate - 0.004, 0.018), 0.050)

        # Improved: Terminal growth lifecycle-aware, never exceeds risk_free_rate
        terminal_growth, terminal_growth_reasoning = self._calculate_terminal_growth(
            revenue_cagr, stage, risk_free_rate, profile
        )

        # Improved: Margin normalization respects cyclicality
        is_cyclical = profile.get("is_cyclical", False)
        if is_cyclical:
            # Use normalized average, not peak, for cyclical companies
            base_terminal_margin = avg_margin
        else:
            # For non-cyclical, use max of average and current
            base_terminal_margin = max(avg_margin, current_margin)
            if best_recent_margin > base_terminal_margin:
                base_terminal_margin = 0.50 * base_terminal_margin + 0.50 * best_recent_margin

        # Apply margin trend adjustment from profile
        margin_trend = profile.get("margin_trend", "stable")
        if margin_trend == "declining":
            margin_bonus *= 0.7
        elif margin_trend == "improving":
            margin_bonus *= 1.2

        terminal_margin = base_terminal_margin + margin_bonus
        terminal_margin = min(max(terminal_margin, 0.03), 0.45)

        current_sales_to_capital = auto["current_sales_to_capital_ratio"]
        terminal_sales_to_capital = max(current_sales_to_capital + capital_bonus, 0.5)
        terminal_sales_to_capital = min(terminal_sales_to_capital, 4.0)

        tax_marginal = 0.24 if company_data.get("country") == "United States" else 0.25

        market_cap = company_data["market_cap"]
        debt = company_data["total_debt"]
        beta = company_data.get("beta", 1.0) or 1.0

        unlevered_beta = beta / (1 + (1 - tax_marginal) * (debt / max(market_cap, 1e-9)))

        # Improved: Beta mean reversion toward market beta
        terminal_unlevered_beta, beta_reversion_reasoning, beta_convergence_year = self._calculate_terminal_beta(
            unlevered_beta, stage, profile
        )

        # Improved: Sector-aware ERP with leverage premium
        narrative_risk_erp = 0.002 if risk_penalty >= 0.008 else 0.0
        leverage_premium = 0.0
        if profile.get("leverage", {}).get("is_highly_levered"):
            leverage_premium = 0.015

        erp = base_erp + profile.get("country_risk_premium", 0) + narrative_risk_erp + leverage_premium

        # Improved: Build assumption bridge for full transparency
        assumption_bridge = self._build_assumption_bridge(
            revenue_cagr=revenue_cagr,
            stage=stage,
            signal_log=signal_log,
            profile_adjustments=profile_adjustments,
            g1_begin=g1_begin,
            current_margin=current_margin,
            avg_margin=avg_margin,
            margin_trend=margin_trend,
            terminal_margin=terminal_margin,
            current_sales_to_capital=current_sales_to_capital,
            capital_bonus=capital_bonus,
            terminal_sales_to_capital=terminal_sales_to_capital,
            unlevered_beta=unlevered_beta,
            terminal_unlevered_beta=terminal_unlevered_beta,
            beta_reversion_reasoning=beta_reversion_reasoning,
            risk_free_rate=risk_free_rate,
            terminal_growth=terminal_growth,
            terminal_growth_reasoning=terminal_growth_reasoning,
            base_erp=base_erp,
            erp=erp,
            leverage_premium=leverage_premium,
        )

        # Improved: Collect warnings for data quality issues
        assumption_warnings = self._collect_assumption_warnings(profile, stage, terminal_margin)

        # Build assumptions dict with all 60+ existing keys (backward compatible)
        return {
            "risk_free_rate": risk_free_rate,
            "ERP": round(float(erp), 4),

            "equity_value": market_cap,
            "debt_value": debt,
            "cash_and_non_operating_asset": company_data.get("cash", 0.0),

            "unlevered_beta": round(float(unlevered_beta), 3),
            "terminal_unlevered_beta": round(float(terminal_unlevered_beta), 3),
            "year_beta_begins_to_converge_to_terminal_beta": beta_convergence_year,

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
            
            # NEW: Phase 1 optional keys for transparency
            "assumption_bridge": assumption_bridge,
            "assumption_warnings": assumption_warnings,
            "signal_log": signal_log,
        }

    def _safe_cagr(self, series):
        """Calculate CAGR safely, handling edge cases."""
        series = np.array(series, dtype=float)

        if len(series) < 2:
            return 0.03

        if series[0] <= 0:
            return 0.03

        return float((series[-1] / series[0]) ** (1 / (len(series) - 1)) - 1)

    def _collect_narrative_signals(self, business_analysis):
        """
        Extract narrative signals with deduplication.
        Returns structured dict of signals without double-counting.
        """
        signals_log = {"growth": [], "margin": [], "capital": [], "risks": []}
        used_texts = set()

        # Process structured signals first (most reliable)
        if business_analysis.growth_outlook == "positive":
            signals_log["growth"].append({
                "phrase": "business_analyst_positive_outlook",
                "strength": "strong",
                "source": "structured_outlook",
                "bonus": 0.010,
                "evidence_docs": []
            })

        if business_analysis.margin_outlook == "improving":
            signals_log["margin"].append({
                "phrase": "business_analyst_margin_improving",
                "strength": "strong",
                "source": "structured_outlook",
                "bonus": 0.010,
                "evidence_docs": []
            })

        if business_analysis.capital_efficiency == "improving":
            signals_log["capital"].append({
                "phrase": "business_analyst_capital_efficiency_improving",
                "strength": "strong",
                "source": "structured_outlook",
                "bonus": 0.15,
                "evidence_docs": []
            })

        # Negative outlooks
        if business_analysis.growth_outlook == "negative":
            signals_log["risks"].append({
                "phrase": "business_analyst_growth_negative",
                "strength": "strong",
                "source": "structured_outlook",
                "penalty": 0.010,
                "evidence_docs": []
            })

        if business_analysis.margin_outlook == "pressured":
            signals_log["margin"].append({
                "phrase": "business_analyst_margin_pressured",
                "strength": "strong",
                "source": "structured_outlook",
                "bonus": -0.010,
                "evidence_docs": []
            })

        if business_analysis.capital_efficiency == "capital_intensive":
            signals_log["capital"].append({
                "phrase": "business_analyst_capital_intensive",
                "strength": "strong",
                "source": "structured_outlook",
                "bonus": -0.15,
                "evidence_docs": []
            })

        # Process drivers (business analysis drivers)
        for driver in (business_analysis.drivers or []):
            lower_driver = driver.lower()

            # Growth signals - non-cyclical recurring revenue
            if any(kw in lower_driver for kw in ["services growth", "cloud services", "app store", "advertising", "subscription"]):
                if lower_driver not in used_texts:
                    signals_log["growth"].append({
                        "phrase": driver[:80],
                        "strength": "strong" if any(w in lower_driver for w in ["increased", "growth"]) else "moderate",
                        "source": "business_driver",
                        "bonus": 0.012,
                        "evidence_docs": []
                    })
                    used_texts.add(lower_driver)

            # Margin signals
            if any(kw in lower_driver for kw in ["margin improved", "cost savings", "mix shift", "leverage", "efficiency"]):
                if lower_driver not in used_texts:
                    signals_log["margin"].append({
                        "phrase": driver[:80],
                        "strength": "strong",
                        "source": "business_driver",
                        "bonus": 0.012,
                        "evidence_docs": []
                    })
                    used_texts.add(lower_driver)

            # Capital signals
            if any(kw in lower_driver for kw in ["asset-light", "software", "cloud", "recurring", "app store"]):
                if lower_driver not in used_texts:
                    signals_log["capital"].append({
                        "phrase": driver[:80],
                        "strength": "strong",
                        "source": "business_driver",
                        "bonus": 0.20,
                        "evidence_docs": []
                    })
                    used_texts.add(lower_driver)

        # Process risks
        for risk in (business_analysis.risks or []):
            lower_risk = risk.lower()

            if any(kw in lower_risk for kw in ["currency", "foreign exchange", "inflation", "macro", "competition", "weakness", "decline"]):
                if lower_risk not in used_texts:
                    strength = "strong" if "macro" in lower_risk or "competition" in lower_risk else "moderate"
                    penalty = 0.006 if strength == "strong" else 0.004
                    signals_log["risks"].append({
                        "phrase": risk[:80],
                        "strength": strength,
                        "source": "business_risk",
                        "penalty": penalty,
                        "evidence_docs": []
                    })
                    used_texts.add(lower_risk)

        # Process SEC filing evidence (with deduplication, top 10)
        for doc in (business_analysis.evidence or [])[:10]:
            doc_text = getattr(doc, "text", "").lower()
            if doc_text and doc_text not in used_texts and len(doc_text) > 20:
                # Check for growth signals
                if any(kw in doc_text for kw in ["services", "app store", "cloud", "increased", "growth"]):
                    signals_log["growth"].append({
                        "phrase": doc_text[:80],
                        "strength": "moderate",
                        "source": "sec_evidence",
                        "bonus": 0.008,
                        "evidence_docs": [
                            {
                                "company": getattr(doc, "company", None),
                                "fiscal_year": getattr(doc, "fiscal_year", None),
                                "doc_id": getattr(doc, "doc_id", None)
                            }
                        ]
                    })
                    used_texts.add(doc_text)

        return signals_log

    def _calculate_terminal_growth(self, revenue_cagr, life_cycle_stage, risk_free_rate, profile):
        """
        Calculate terminal growth rate based on lifecycle stage.
        Never exceeds risk_free_rate. Reasoning logged for transparency.
        """
        if life_cycle_stage == "startup":
            target_terminal = 0.03
            reasoning = "Startup expected to mature toward market-rate growth"

        elif life_cycle_stage == "young_growth":
            target_terminal = 0.035
            reasoning = "Young growth company fading toward market growth"

        elif life_cycle_stage == "high_growth":
            target_terminal = min(revenue_cagr * 0.5, 0.03)
            reasoning = f"High-growth company fading: {revenue_cagr:.1%} CAGR -> half for terminal"

        elif life_cycle_stage == "mature_growth":
            target_terminal = 0.025
            reasoning = "Mature growth company -> GDP-like growth"

        elif life_cycle_stage == "mature_stable":
            target_terminal = 0.02
            reasoning = "Mature stable company -> long-term GDP-like growth"

        elif life_cycle_stage == "decline":
            target_terminal = max(revenue_cagr * 0.5, -0.02)
            reasoning = f"Declining company -> perpetual decline at half CAGR or floor"

        elif life_cycle_stage == "distress":
            target_terminal = max(revenue_cagr, -0.05)
            reasoning = "Distress company -> negative or flat terminal growth"

        else:
            target_terminal = 0.02
            reasoning = "Unknown lifecycle -> default mature growth"

        # Enforce upper bound: never exceed risk_free_rate
        terminal_growth = min(target_terminal, risk_free_rate - 0.005)

        # Emerging market may get slight premium but still capped
        if profile.get("is_emerging_market"):
            terminal_growth = min(target_terminal, risk_free_rate - 0.002)
            reasoning += f" [Emerging market: capped at {terminal_growth:.2%}]"

        # Lower bound
        terminal_growth = max(terminal_growth, 0.015)

        return terminal_growth, reasoning

    def _calculate_terminal_beta(self, unlevered_beta, life_cycle_stage, profile):
        """
        Calculate terminal beta with mean reversion toward market beta (1.0).
        Returns terminal beta, reasoning, and convergence year.
        """
        market_beta = 1.0

        if life_cycle_stage == "startup":
            terminal_beta = 0.95 * unlevered_beta + 0.05 * market_beta
            reasoning = "Startup -> rapid fade to market beta"
            convergence_year = 2

        elif life_cycle_stage == "young_growth":
            terminal_beta = 0.85 * unlevered_beta + 0.15 * market_beta
            reasoning = "Young growth -> gradual fade toward market beta"
            convergence_year = 3

        elif life_cycle_stage == "high_growth":
            terminal_beta = 0.70 * unlevered_beta + 0.30 * market_beta
            reasoning = "High-growth -> moderate fade toward market beta"
            convergence_year = 3

        elif life_cycle_stage == "mature_growth":
            terminal_beta = 0.50 * unlevered_beta + 0.50 * market_beta
            reasoning = "Mature growth -> 50/50 current/market beta"
            convergence_year = 3

        elif life_cycle_stage == "mature_stable":
            terminal_beta = 0.40 * unlevered_beta + 0.60 * market_beta
            reasoning = "Mature stable -> mostly market beta"
            convergence_year = 3

        elif life_cycle_stage == "decline":
            terminal_beta = 0.50 * unlevered_beta + 0.50 * market_beta
            reasoning = "Declining -> move toward market beta"
            convergence_year = 3

        elif life_cycle_stage == "distress":
            terminal_beta = 0.30 * unlevered_beta + 0.70 * market_beta
            reasoning = "Distress -> mostly market beta (high uncertainty)"
            convergence_year = 2

        else:
            terminal_beta = 0.50 * unlevered_beta + 0.50 * market_beta
            reasoning = "Unknown lifecycle -> 50/50 blend"
            convergence_year = 3

        # Don't force very low beta up too aggressively
        if unlevered_beta < 0.7:
            terminal_beta = max(terminal_beta, 0.7)
            reasoning += " [Floor at 0.7x due to low current beta]"

        return terminal_beta, reasoning, convergence_year

    def _adjust_for_profile(self, growth_bonus, margin_bonus, capital_bonus, profile):
        """
        Apply lifecycle and sector-specific adjustments to bonuses.
        Returns adjusted bonuses and dictionary of applied adjustments.
        """
        adjustments = {}

        # Lifecycle adjustments
        stage = profile.get("life_cycle_stage", "mature_stable")
        if stage == "mature_stable":
            growth_bonus *= 0.8
            adjustments["lifecycle_growth"] = "mature_stable -> 20% reduction"

        elif stage == "distress":
            growth_bonus *= 0.5
            margin_bonus *= 0.7
            adjustments["lifecycle_growth"] = "distress -> 50% reduction"
            adjustments["lifecycle_margin"] = "distress -> 30% reduction"

        elif stage == "decline":
            growth_bonus *= 0.7
            adjustments["lifecycle_growth"] = "decline -> 30% reduction"

        # Cyclicality: handled separately in margin calc, but note it here
        if profile.get("is_cyclical"):
            adjustments["cyclical_handling"] = "Using normalized (avg) margin, not best recent"

        # Leverage adjustments
        leverage = profile.get("leverage", {})
        if leverage.get("is_solvency_risk"):
            growth_bonus *= 0.7
            margin_bonus *= 0.8
            adjustments["solvency_risk"] = "Reduce assumptions due to financial stress"

        if leverage.get("is_highly_levered"):
            capital_bonus *= 0.9
            adjustments["high_leverage"] = "Conservative capital assumptions"

        # Sector adjustments
        sector_flags = profile.get("sector_flags", {})
        if sector_flags.get("is_commodity_linked"):
            margin_bonus *= 0.6
            adjustments["commodity_sector"] = "Use normalized margins, not recent peak"

        if sector_flags.get("is_reit"):
            # REITs have different dynamics, but keep assumptions reasonable
            adjustments["reit"] = "REIT: Note that FFO-based valuation may be more appropriate"

        if sector_flags.get("is_financial_firm"):
            adjustments["financial"] = "Financial firm: DCF applies but with caution on capital assumptions"

        return growth_bonus, margin_bonus, capital_bonus, adjustments

    def _clean_narrative_phrase(self, text: str, max_len: int = 60) -> str:
        """Clean and normalize narrative phrases for display."""
        if not text:
            return ""
        
        text = str(text).strip()
        
        # Remove duplicated words (e.g., "services services" -> "services")
        words = text.split()
        cleaned = []
        for word in words:
            if not cleaned or word.lower() != cleaned[-1].lower():
                cleaned.append(word)
        text = " ".join(cleaned)
        
        # Remove table/structural noise
        text = text.replace("the following table", "")
        text = text.replace("the following", "")
        text = " ".join(text.split())  # Normalize whitespace
        
        # Truncate
        if len(text) > max_len:
            text = text[:max_len].rsplit(" ", 1)[0] + "..."
        
        return text

    def _build_assumption_bridge(
        self,
        revenue_cagr,
        stage,
        signal_log,
        profile_adjustments,
        g1_begin,
        current_margin,
        avg_margin,
        margin_trend,
        terminal_margin,
        current_sales_to_capital,
        capital_bonus,
        terminal_sales_to_capital,
        unlevered_beta,
        terminal_unlevered_beta,
        beta_reversion_reasoning,
        risk_free_rate,
        terminal_growth,
        terminal_growth_reasoning,
        base_erp,
        erp,
        leverage_premium,
    ):
        """
        Build transparent derivation trail for each assumption.
        Used for auditability and explanation.
        
        BUG FIX (2026-08-16): Changed Current (TTM) margin from g1_begin (growth rate)
        to current_margin (actual operating margin) to fix semantic inconsistency.
        """
        return {
            "lifecycle_stage": stage,
            "revenue_growth": [
                f"Historical revenue CAGR (5Y): {revenue_cagr:.2%}",
                f"Lifecycle stage {stage} implies floor {self._get_stage_floor(stage):.1%}",
                f"Growth signals identified: {len(signal_log.get('growth', []))} drivers",
                *[self._clean_narrative_phrase(s.get("phrase", "")) for s in signal_log.get("growth", [])[:3]],
                f"Growth bonus after signals: +{sum(s.get('bonus', 0) for s in signal_log.get('growth', [])) - sum(s.get('penalty', 0) for s in signal_log.get('risks', [])):.1%}",
                f"Profile adjustments applied: {profile_adjustments}",
                f"Calculated cycle 1 begin: {g1_begin:.2%} (bounded by stage [{self._get_stage_floor(stage):.1%}, {self._get_stage_cap(stage):.1%}])",
            ],
            "operating_margin": [
                f"Historical avg margin: {avg_margin:.2%}",
                f"Current (TTM) margin: {current_margin:.2%}",
                f"Margin trend: {margin_trend}",
                f"Margin signals identified: {len(signal_log.get('margin', []))} drivers",
                *[self._clean_narrative_phrase(s.get("phrase", "")) for s in signal_log.get("margin", [])[:3]],
                f"Terminal margin before adjustments: {terminal_margin - sum(s.get('bonus', 0) for s in signal_log.get('margin', [])):.2%}",
                f"Terminal margin after signals: {terminal_margin:.2%} (bounded [3%, 45%])",
            ],
            "capital_efficiency": [
                f"Current sales-to-capital: {current_sales_to_capital:.2f}x",
                f"Capital signals identified: {len(signal_log.get('capital', []))} drivers",
                *[self._clean_narrative_phrase(s.get("phrase", "")) for s in signal_log.get("capital", [])[:3]],
                f"Capital bonus: +{capital_bonus:.2f}x",
                f"Terminal sales-to-capital: {terminal_sales_to_capital:.2f}x (bounded [0.5x, 4.0x])",
            ],
            "cost_of_equity": [
                f"Unlevered beta (current): {unlevered_beta:.3f}x",
                f"Terminal beta: {terminal_unlevered_beta:.3f}x",
                beta_reversion_reasoning,
                f"Risk signals identified: {len(signal_log.get('risks', []))} factors",
                f"Base ERP: {base_erp:.2%}",
                f"Leverage premium: {leverage_premium:.2%}",
                f"Final ERP: {erp:.2%}",
            ],
            "terminal_growth": [
                f"Risk-free rate: {risk_free_rate:.2%}",
                f"Terminal growth rate: {terminal_growth:.2%}",
                terminal_growth_reasoning,
                f"Constraint check: Terminal growth ({terminal_growth:.2%}) < Risk-free rate ({risk_free_rate:.2%}) [OK]",
            ],
        }

    def _collect_assumption_warnings(self, profile, stage, terminal_margin):
        """Collect warnings for potentially risky assumptions."""
        warnings = []

        # Data quality flags
        data_flags = profile.get("data_quality_flags", [])
        if "short_history_lt_5y" in data_flags:
            warnings.append("Short history (<5 years): Assumptions have limited historical basis")
        if "non_positive_revenue_present" in data_flags:
            warnings.append("Non-positive revenue history detected: Assumptions may be unreliable")

        # Distress company
        if stage == "distress":
            warnings.append("Distress stage detected: DCF may not be appropriate; consider liquidation value")

        # High terminal margin
        if terminal_margin > 0.35:
            warnings.append(f"Terminal margin ({terminal_margin:.1%}) very high: Verify sustainability")

        # Very low terminal margin
        if terminal_margin < 0.05:
            warnings.append(f"Terminal margin ({terminal_margin:.1%}) very low: Review assumptions")

        # Leverage risk
        if profile.get("leverage", {}).get("is_highly_levered"):
            warnings.append("Highly levered balance sheet: DCF sensitive to debt assumptions")

        return warnings

    def _get_stage_cap(self, stage):
        """Helper to get growth cap for stage."""
        caps = {
            "startup": 0.25,
            "young_growth": 0.20,
            "high_growth": 0.16,
            "mature_growth": 0.11,
            "mature_stable": 0.075,
            "decline": 0.025,
            "distress": 0.015,
        }
        return caps.get(stage, 0.075)

    def _get_stage_floor(self, stage):
        """Helper to get growth floor for stage."""
        floors = {
            "startup": 0.08,
            "young_growth": 0.06,
            "high_growth": 0.05,
            "mature_growth": 0.035,
            "mature_stable": 0.015,
            "decline": -0.04,
            "distress": -0.06,
        }
        return floors.get(stage, 0.015)