from __future__ import annotations

import re


def pct(x):
    try:
        return f"{x:.1%}"
    except Exception:
        return "n/a"


def money(x):
    try:
        return f"${x:,.2f}bn"
    except Exception:
        return "n/a"


def clean_display_text(text: str, max_len: int = 350) -> str:
    """
    Cleans SEC filing text for report display.
    This does not change the underlying retrieval result.
    """

    text = str(text or "")

    text = re.sub(r"\s+", " ", text).strip()

    # Remove common SEC table/page artifacts.
    text = re.sub(r"Apple Inc\.\s*\|\s*\d{4}\s+Form\s+10-K\s*\|\s*\d+", " ", text)
    text = re.sub(r"\|\s*\d{4}\s+Form\s+10-K\s*\|\s*\d+", " ", text)
    text = re.sub(r"\|\s*Form\s+10-K\s*\|", " ", text)
    text = text.replace("|", " ")

    # Remove duplicated headings caused by HTML extraction.
    text = re.sub(r"\bServices Services\b", "Services", text)
    text = re.sub(r"\bProducts Products\b", "Products", text)
    text = re.sub(r"\bMacroeconomic Conditions Macroeconomic conditions\b", "Macroeconomic conditions", text)
    text = re.sub(r"\bServices Gross Margin Services gross margin\b", "Services gross margin", text)
    text = re.sub(r"\bProducts Gross Margin Products gross margin\b", "Products gross margin", text)

    # Shorten table-heavy lead-ins.
    text = re.sub(
        r"Products and Services Performance The following table shows net sales by category for.*?:",
        "Products and Services performance:",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"Segment Operating Performance The following table shows net sales by reportable segment for.*?:",
        "Segment operating performance:",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"Gross Margin Products and Services gross margin and gross margin percentage for.*?:",
        "Gross margin:",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(r"\s+", " ", text).strip()

    if len(text) > max_len:
        text = text[:max_len].rsplit(" ", 1)[0] + "..."

    return text


class ReportWriter:
    def write(self, analysis):
        data = analysis.company_data
        ba = analysis.business_analysis
        val = analysis.dcf_result
        mc = analysis.monte_carlo_result
        ex = analysis.implied_market_expectations
        assumptions = analysis.valuation_assumptions

        shares = data.get("shares_outstanding") or 1

        per_share = (
            val.get("equity_value", 0) / shares
            if shares
            else None
        )

        current_per_share = data.get("current_price")

        if current_per_share is None and shares:
            current_per_share = data.get("market_cap", 0) / shares

        upside = (
            val.get("equity_value", 0) / data.get("market_cap", 1) - 1
            if data.get("market_cap")
            else None
        )

        evidence_lines = []

        if ba:
            for i, doc in enumerate(ba.evidence[:5], 1):
                src = ", ".join(
                    str(x)
                    for x in [
                        getattr(doc, "company", None),
                        getattr(doc, "fiscal_year", None),
                        getattr(doc, "doc_id", None),
                        getattr(doc, "paragraph_id", None),
                    ]
                    if x
                )

                cleaned_text = clean_display_text(getattr(doc, "text", ""))

                evidence_lines.append(
                    f"{i}. {cleaned_text}  \n"
                    f"   Source: {src or 'local filing chunk'}"
                )

        implied_growth = ex.get("implied_revenue_growth_cycle1_begin")
        implied_margin = ex.get("implied_terminal_operating_margin")

        implied_growth_text = (
            pct(implied_growth)
            if implied_growth is not None
            else "No single-variable solution within default bounds"
        )

        implied_margin_text = (
            pct(implied_margin)
            if implied_margin is not None
            else "No single-variable solution within default bounds"
        )

        drivers = ba.drivers if ba else []
        risks = ba.risks if ba else []

        return f"""# {data.get('company_name', analysis.ticker)} ({analysis.ticker}) Equity Research Memo

## Executive View

- **Estimated intrinsic equity value:** {money(val.get('equity_value'))}
- **Current market cap:** {money(data.get('market_cap'))}
- **Implied upside/downside:** {pct(upside) if upside is not None else 'n/a'}
- **Estimated intrinsic value per share:** {money(per_share).replace('bn', '') if per_share is not None else 'n/a'}
- **Current price per share:** {money(current_per_share).replace('bn', '') if current_per_share is not None else 'n/a'}

## Business Drivers

{chr(10).join('- ' + clean_display_text(x, max_len=500) for x in drivers)}

## Key Risks

{chr(10).join('- ' + clean_display_text(x, max_len=500) for x in risks)}

## Valuation Approach

- **Recommended model:** {analysis.valuation_decision.get('primary')}
- **Secondary approach:** {analysis.valuation_decision.get('secondary')}
- **Note:** {analysis.valuation_decision.get('note') or 'None'}

## Core DCF Assumptions

- Year 1 revenue growth: {pct(assumptions.get('revenue_growth_rate_cycle1_begin'))}
- End of cycle 1 revenue growth: {pct(assumptions.get('revenue_growth_rate_cycle1_end'))}
- Terminal growth: {pct(assumptions.get('revenue_growth_rate_cycle3_end'))}
- Current operating margin: {pct(assumptions.get('current_operating_margin'))}
- Terminal operating margin: {pct(assumptions.get('terminal_operating_margin'))}
- ERP: {pct(assumptions.get('ERP'))}
- Risk-free rate: {pct(assumptions.get('risk_free_rate'))}

## Monte Carlo Risk View

- P10 equity value: {money(mc.get('p10'))}
- P50 equity value: {money(mc.get('p50'))}
- P90 equity value: {money(mc.get('p90'))}
- Probability undervalued: {pct(mc.get('probability_undervalued'))}
- Successful scenarios: {mc.get('scenarios')}

## Market-Implied Expectations

- Implied first-cycle starting revenue growth: {implied_growth_text}
- Implied terminal operating margin: {implied_margin_text}
- Assessment: {ex.get('assessment')}

## Selected Filing Evidence

{chr(10).join(evidence_lines) if evidence_lines else 'No local SEC evidence was available. Add sec-mdna-rag JSONL data for stronger narrative support.'}

## Disclaimer

This report is generated by an experimental research system. It is not financial advice. Validate every data point, assumption, and source before making any investment decision.
"""