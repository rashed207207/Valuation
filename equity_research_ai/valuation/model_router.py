"""Diagnostic valuation model router.

This layer only chooses and explains the recommended valuation model. It does not
change valuation math or implement specialized valuation models.
"""

from __future__ import annotations


IMPLEMENTED_MODELS = {
    "fcff_dcf",
    "segment_dcf",
    "retail_fcff_dcf",
    "financial_excess_return",
    "reit_ffo_affo",
    "utility_fcfe",
    "distressed_liquidation",
    "commodity_cycle_dcf",
    "telecom_fcfe_or_dividend",
}


def _default_fallback(profile):
    return {
        "recommended_model": "fcff_dcf",
        "secondary_model": "fcff_dcf",
        "fallback_model": "fcff_dcf",
        "model_reasoning": "Default diagnostic fallback: the production valuation remains FCFF DCF until a specialized model is implemented.",
        "model_warning": "Model not implemented yet; fallback to fcff_dcf. Actual valuation math remains unchanged.",
    }


def select_valuation_model(profile):
    """Route a company profile to a diagnostic model label with explanation.

    This router is intentionally limited to routing and rationale. It does not
    alter the selected DCF engine or any existing valuation assumptions.
    """
    if not isinstance(profile, dict):
        return _default_fallback(profile)

    sector_flags = profile.get("sector_flags", {})
    stage = profile.get("life_cycle_stage", "mature_stable")
    leverage = profile.get("leverage", {})
    business_type = profile.get("business_type", "general_company")
    is_financial = bool(sector_flags.get("is_financial_firm"))
    is_reit = bool(sector_flags.get("is_reit"))
    is_utility = bool(sector_flags.get("is_utility"))
    is_commodity = bool(sector_flags.get("is_commodity_linked"))
    is_distress = bool(stage == "distress" or leverage.get("is_solvency_risk"))
    is_high_leverage = bool(leverage.get("is_highly_levered"))

    # Recommended model first; fallback remains FCFF DCF while the specialized
    # model remains diagnostic-only and not yet wired into the valuation engine.
    if is_financial:
        recommended = "financial_excess_return"
        secondary = "fcff_dcf"
        reasoning = (
            "Financial firms are balance-sheet driven, regulated, and do not fit a traditional "
            "FCFF operating cash flow framework as a primary model."
        )
    elif is_reit:
        recommended = "reit_ffo_affo"
        secondary = "fcff_dcf"
        reasoning = (
            "REITs should be framed around FFO/AFFO rather than standard net income or FCFF cash flows."
        )
    elif is_utility:
        recommended = "utility_fcfe"
        secondary = "fcff_dcf"
        reasoning = (
            "Utility cash flows are often regulated and equity-centric; FCFE remains a more natural lens "
            "than unlevered FCFF."
        )
    elif is_distress:
        recommended = "distressed_liquidation"
        secondary = "fcff_dcf"
        reasoning = (
            "Distress indicators such as solvency risk, negative earnings, or weak coverage justify a liquidation "
            "or restructuring-first diagnostic view rather than a standard going-concern DCF."
        )
    elif business_type in {"multi_segment_platform", "consumer_technology_platform"}:
        recommended = "segment_dcf"
        secondary = "fcff_dcf"
        reasoning = (
            "Platform or multi-segment businesses often have materially different operating profiles by segment, "
            "so a segment-level DCF is more diagnostic than a single FCFF aggregate model."
        )
    elif business_type == "mature_retailer":
        recommended = "retail_fcff_dcf"
        secondary = "fcff_dcf"
        reasoning = (
            "Mature retailers have working-capital and store-level operating dynamics that merit a retail-specific "
            "FCFF lens, though the current production model remains the general FCFF DCF."
        )
    elif business_type == "telecom_leverage_heavy":
        recommended = "telecom_fcfe_or_dividend"
        secondary = "fcff_dcf"
        reasoning = (
            "Telecom businesses are often leverage-heavy and dividend-sensitive; FCFE or dividend-oriented valuation "
            "is typically more informative than a pure FCFF approach."
        )
    elif is_commodity:
        recommended = "commodity_cycle_dcf"
        secondary = "fcff_dcf"
        reasoning = (
            "Commodity-linked businesses should be reviewed through a cycle-normalized framework because spot margins "
            "can be distorted by short-term commodity price swings."
        )
    else:
        recommended = "fcff_dcf"
        secondary = "fcff_dcf"
        reasoning = (
            "Default diagnostic selection: the current production methodology is a standard FCFF DCF, and no special-case "
            "valuation model is required for this profile."
        )

    implemented = recommended in IMPLEMENTED_MODELS
    if not implemented:
        fallback = "fcff_dcf"
        warning = (
            f"Selected model '{recommended}' is not yet implemented. Diagnostic routing falls back to '{fallback}' "
            "to preserve the current valuation pipeline and keep the math unchanged."
        )
        return {
            "recommended_model": fallback,
            "secondary_model": fallback,
            "fallback_model": fallback,
            "model_reasoning": reasoning + " " + warning,
            "model_warning": warning,
            "business_type": business_type,
        }

    return {
        "recommended_model": recommended,
        "secondary_model": secondary,
        "fallback_model": "fcff_dcf",
        "model_reasoning": reasoning + " The production valuation remains FCFF DCF until the selected model is implemented.",
        "model_warning": "" if recommended == "fcff_dcf" else "Selected model is diagnostic only; valuation math remains unchanged until a specialized model is implemented.",
        "business_type": business_type,
    }
