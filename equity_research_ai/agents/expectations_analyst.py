from __future__ import annotations

from equity_research_ai.valuation.dcf import valuator_multi_phase
from equity_research_ai.valuation.implied_expectations import solve_for_implied_assumption


class ExpectationsAgent:
    """
    ExpectationsAgent v2

    Explains what current market value implies.

    If one-variable solving fails, the agent explains that the market price
    requires a combination of assumptions rather than one isolated input change.
    """

    def analyze(self, base_params, market_cap):
        result = {}

        try:
            model_equity_value = valuator_multi_phase(**base_params)["equity_value"]
            result["model_equity_value"] = model_equity_value
            result["market_cap"] = market_cap
            result["valuation_gap"] = market_cap - model_equity_value
            result["valuation_gap_pct"] = (
                market_cap / model_equity_value - 1
                if model_equity_value
                else None
            )
        except Exception as error:
            result["model_equity_value"] = None
            result["model_equity_value_error"] = str(error)
            result["valuation_gap_pct"] = None

        try:
            result["implied_revenue_growth_cycle1_begin"] = solve_for_implied_assumption(
                base_params=base_params,
                target_equity_value=market_cap,
                param_name="revenue_growth_rate_cycle1_begin",
                bounds=(-0.10, 0.80),
            )
            result["implied_revenue_growth_cycle1_begin_error"] = None
        except Exception as error:
            result["implied_revenue_growth_cycle1_begin"] = None
            result["implied_revenue_growth_cycle1_begin_error"] = str(error)

        try:
            result["implied_terminal_operating_margin"] = solve_for_implied_assumption(
                base_params=base_params,
                target_equity_value=market_cap,
                param_name="terminal_operating_margin",
                bounds=(0.02, 0.55),
            )
            result["implied_terminal_operating_margin_error"] = None
        except Exception as error:
            result["implied_terminal_operating_margin"] = None
            result["implied_terminal_operating_margin_error"] = str(error)

        result["assessment"] = self._build_assessment(base_params, result)

        return result

    def _build_assessment(self, base_params, result):
        parts = []

        valuation_gap_pct = result.get("valuation_gap_pct")
        implied_growth = result.get("implied_revenue_growth_cycle1_begin")
        implied_margin = result.get("implied_terminal_operating_margin")

        base_growth = base_params.get("revenue_growth_rate_cycle1_begin")
        base_margin = base_params.get("terminal_operating_margin")

        if valuation_gap_pct is not None:
            if valuation_gap_pct > 1.0:
                parts.append(
                    "The market value is more than double the model-implied equity value under the current assumptions."
                )
            elif valuation_gap_pct > 0.25:
                parts.append(
                    "The market value is materially above the model-implied equity value under the current assumptions."
                )
            elif valuation_gap_pct < -0.25:
                parts.append(
                    "The market value is materially below the model-implied equity value under the current assumptions."
                )
            else:
                parts.append(
                    "The market value is broadly close to the model-implied equity value under the current assumptions."
                )

        if implied_growth is not None:
            if base_growth is not None and implied_growth > base_growth:
                parts.append(
                    "The market implies higher near-term revenue growth than the base case."
                )
            else:
                parts.append(
                    "The market implies similar or lower near-term revenue growth than the base case."
                )
        else:
            parts.append(
                "Changing only near-term revenue growth within the default bounds cannot bridge the valuation gap."
            )

        if implied_margin is not None:
            if base_margin is not None and implied_margin > base_margin:
                parts.append(
                    "The market implies a higher terminal operating margin than the base case."
                )
            else:
                parts.append(
                    "The market implies a similar or lower terminal operating margin than the base case."
                )
        else:
            parts.append(
                "Changing only terminal operating margin within the default bounds cannot bridge the valuation gap."
            )

        if implied_growth is None and implied_margin is None:
            parts.append(
                "This suggests the market price requires a combination of assumptions rather than one isolated variable change."
            )
            parts.append(
                "Possible required drivers include higher long-term Services mix, stronger terminal ROIC, higher sales-to-capital efficiency, lower discount rate, explicit buyback modeling, or a segment-level valuation model."
            )

        return " ".join(parts)