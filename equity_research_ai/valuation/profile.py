from __future__ import annotations
import warnings
import numpy as np


def build_company_profile(revenue_history, net_income_history, ebit_history,
                           operating_cash_flow_history, capex_history, rd_expense_history,
                           dividends_paid_history, gics_sector, years_since_ipo, country,
                           country_risk_premium_table,
                           total_debt=None, cash_and_equivalents=None,
                           interest_expense_history=None, shares_outstanding=None,
                           market_price_per_share=None):
    """
    Build a company profile used to select an appropriate valuation methodology,
    loosely following Damodaran's life-cycle / approach-selection framework.

    Required *_history arguments must be equal-length, ordered oldest -> most recent.

    Optional inputs (total_debt, cash_and_equivalents, interest_expense_history,
    shares_outstanding, market_price_per_share) sharpen leverage / solvency / size
    classification. Everything degrades gracefully to None if omitted.
    """
    revenue = np.array(revenue_history, dtype=float)
    net_income = np.array(net_income_history, dtype=float)
    ebit = np.array(ebit_history, dtype=float)
    ocf = np.array(operating_cash_flow_history, dtype=float)
    capex = np.array(capex_history, dtype=float)
    rd = np.array(rd_expense_history, dtype=float)
    divs = np.array(dividends_paid_history, dtype=float)

    n = len(revenue)
    if n < 2:
        raise ValueError("At least two years of revenue history are required.")
    for name, arr in [('net_income_history', net_income), ('ebit_history', ebit),
                       ('operating_cash_flow_history', ocf), ('capex_history', capex),
                       ('rd_expense_history', rd), ('dividends_paid_history', divs)]:
        if len(arr) != n:
            raise ValueError(f"{name} must be the same length as revenue_history ({n}).")

    data_quality_flags = []
    if n < 5:
        data_quality_flags.append('short_history_lt_5y')
    if np.any(revenue <= 0):
        data_quality_flags.append('non_positive_revenue_present')

    lookback = min(3, n)  # smoothing window for noisy single-year figures

    sector_l = str(gics_sector).lower()
    is_financial_firm = any(k in sector_l for k in ['financial', 'bank', 'insurance', 'capital markets'])
    is_reit = 'reit' in sector_l  # Only exact REIT keyword, not entire real estate sector
    is_utility = 'utilit' in sector_l
    is_commodity_linked = any(k in sector_l for k in ['energy', 'oil', 'gas', 'mining', 'metals', 'materials'])

    avg_net_income_recent = float(net_income[-lookback:].mean())
    avg_ebit_recent = float(ebit[-lookback:].mean())
    has_positive_reliable_fcf = bool(avg_net_income_recent > 0 and avg_ebit_recent > 0)

    # Earnings quality: average cash conversion over the lookback window rather than a single
    # year, since one year of OCF/NI is easily distorted by working-capital swings or one-offs.
    with np.errstate(divide='ignore', invalid='ignore'):
        eq_ratios = np.where(net_income[-lookback:] != 0, ocf[-lookback:] / net_income[-lookback:], np.nan)
    earnings_quality = np.nanmean(eq_ratios) if not np.all(np.isnan(eq_ratios)) else np.nan

    with np.errstate(divide='ignore', invalid='ignore'):
        ebit_margin_history = np.where(revenue != 0, ebit / revenue, np.nan)
    margin_mean = np.nanmean(ebit_margin_history)
    cyclicality = (np.nanstd(ebit_margin_history) / abs(margin_mean)
                   if margin_mean and not np.isnan(margin_mean) else np.nan)
    is_cyclical = bool(cyclicality > 0.5) if not np.isnan(cyclicality) else False

    # Margin trend in percentage points (recent window vs. earliest window) — robust even
    # when margin_mean is near zero, where a ratio-based comparison would blow up.
    early_margin = np.nanmean(ebit_margin_history[:lookback])
    recent_margin = np.nanmean(ebit_margin_history[-lookback:])
    if np.isnan(early_margin) or np.isnan(recent_margin):
        margin_trend = 'unknown'
    else:
        delta = recent_margin - early_margin
        margin_trend = 'improving' if delta > 0.02 else 'declining' if delta < -0.02 else 'stable'

    revenue_cagr = (revenue[-1] / revenue[0]) ** (1 / (n - 1)) - 1 if revenue[0] > 0 else np.nan
    yoy_growth = revenue[1:] / revenue[:-1] - 1 if np.all(revenue[:-1] > 0) else np.array([])
    revenue_growth_volatility = float(np.std(yoy_growth)) if len(yoy_growth) > 1 else np.nan

    half = n // 2
    growth_first_half = (revenue[half] / revenue[0]) ** (1 / half) - 1 if half > 0 and revenue[0] > 0 else np.nan
    growth_second_half = (revenue[-1] / revenue[half]) ** (1 / (n - half - 1)) - 1 if (n - half - 1) > 0 and revenue[half] > 0 else np.nan
    is_decelerating = bool(growth_second_half < growth_first_half) if not (np.isnan(growth_first_half) or np.isnan(growth_second_half)) else False

    # Smooth reinvestment rate over the lookback window (avoid single-year distortions)
    if ebit[-lookback:].mean() > 0:
        capex_recent = capex[-lookback:].mean()
        rd_recent = rd[-lookback:].mean()
        reinvestment_rate = (capex_recent + rd_recent) / ebit[-lookback:].mean()
    else:
        reinvestment_rate = np.nan
    payout_ratio = abs(divs[-1]) / net_income[-1] if net_income[-1] > 0 else 0.0

    # --- Leverage / solvency (optional inputs) ---
    net_debt = (float(total_debt) - float(cash_and_equivalents or 0.0)) if total_debt is not None else None
    interest_coverage = None
    if interest_expense_history is not None:
        interest_expense = np.abs(np.array(interest_expense_history, dtype=float))
        if len(interest_expense) == n and interest_expense[-lookback:].mean() != 0:
            interest_coverage = float(ebit[-lookback:].mean() / interest_expense[-lookback:].mean())
    is_highly_levered = bool(net_debt is not None and avg_ebit_recent > 0 and (net_debt / avg_ebit_recent) > 5)
    consecutive_loss_years = int((net_income[-lookback:] < 0).sum())
    # A multi-year loss streak is a distress signal for a mature company, but it's the normal,
    # expected state for a young company still investing through its growth phase — so only
    # treat it as a solvency red flag once the company has had time to mature.
    mature_enough_for_loss_signal = years_since_ipo >= 5
    is_solvency_risk = bool(
        (interest_coverage is not None and interest_coverage < 1.5) or
        (consecutive_loss_years >= lookback and mature_enough_for_loss_signal
         and (np.isnan(revenue_cagr) or revenue_cagr < 0.15))
    )

    # --- Size classification (revenue proxy; refined by market cap when supplied) ---
    market_cap = float(shares_outstanding) * float(market_price_per_share) if (shares_outstanding is not None and market_price_per_share is not None) else None
    size_basis = market_cap if market_cap is not None else revenue[-1]
    if size_basis < 300:
        size_category = 'micro_cap'
    elif size_basis < 2_000:
        size_category = 'small_cap'
    elif size_basis < 10_000:
        size_category = 'mid_cap'
    else:
        size_category = 'large_cap'

    # --- Life-cycle scoring ---
    score = 0
    if not np.isnan(revenue_cagr):
        if revenue_cagr > 0.15: score += 2
        elif revenue_cagr > 0.05: score += 1
        elif revenue_cagr < 0: score -= 2
    if not is_decelerating and not np.isnan(revenue_cagr) and revenue_cagr > 0.1: score += 1
    if not np.isnan(reinvestment_rate) and reinvestment_rate > 0.6: score += 1
    if payout_ratio < 0.1: score += 1
    elif payout_ratio > 0.4: score -= 1
    if years_since_ipo < 10: score += 1
    elif years_since_ipo > 20: score -= 1
    if margin_trend == 'improving': score += 1
    elif margin_trend == 'declining': score -= 1

    # Note: "decline" is driven by the actual revenue trajectory, not by the composite score.
    # The score captures growth *intensity* among growing companies; a slow-growing but stable,
    # profitable, low-payout company should land in mature_stable even if age/payout penalties
    # pull its score down — it should not be lumped in with a company whose revenue is shrinking.
    if is_solvency_risk:
        life_cycle_stage = 'distress'
    elif not np.isnan(revenue_cagr) and revenue_cagr < 0:
        life_cycle_stage = 'decline'
    elif score >= 5:
        life_cycle_stage = 'high_growth'
    elif score >= 3:
        life_cycle_stage = 'mature_growth'
    else:
        life_cycle_stage = 'mature_stable'
    if avg_net_income_recent < 0 and not np.isnan(revenue_cagr) and revenue_cagr > 0.15 and life_cycle_stage != 'distress':
        life_cycle_stage = 'young_growth'
    if years_since_ipo < 3 and revenue[0] < 50:
        life_cycle_stage = 'startup'

    crp = country_risk_premium_table.get(country, 0)

    result = {
        'sector_flags': {
            'is_financial_firm': is_financial_firm,
            'is_reit': is_reit,
            'is_utility': is_utility,
            'is_commodity_linked': is_commodity_linked,
        },
        'has_positive_reliable_fcf': has_positive_reliable_fcf,
        'earnings_quality_ratio': round(float(earnings_quality), 2) if not np.isnan(earnings_quality) else None,
        'is_cyclical': is_cyclical,
        'revenue_cagr': round(float(revenue_cagr), 3) if not np.isnan(revenue_cagr) else None,
        'revenue_growth_volatility': round(revenue_growth_volatility, 3) if not np.isnan(revenue_growth_volatility) else None,
        'margin_trend': margin_trend,
        'life_cycle_stage': life_cycle_stage,
        'life_cycle_score': score,
        'reinvestment_rate': round(float(reinvestment_rate), 2) if not np.isnan(reinvestment_rate) else None,
        'payout_ratio': round(float(payout_ratio), 2),
        'leverage': {
            'net_debt': net_debt,
            'interest_coverage': round(interest_coverage, 2) if interest_coverage is not None else None,
            'is_highly_levered': is_highly_levered,
            'consecutive_loss_years': consecutive_loss_years,
            'is_solvency_risk': is_solvency_risk,
        },
        'size_category': size_category,
        'market_cap': market_cap,
        'is_emerging_market': crp > 0.005,
        'country_risk_premium': crp,
        'data_quality_flags': data_quality_flags,
        # Backward-compatible flat keys for legacy code
        'is_financial_firm': is_financial_firm,
        'is_reit': is_reit,
        'is_utility': is_utility,
        'is_commodity_linked': is_commodity_linked,
    }
    return result


def select_valuation_approach(profile, has_enough_comparables):
    """
    Map a profile from build_company_profile() to a recommended primary/secondary
    valuation approach, with a plain-language rationale for each branch.
    """
    flags = profile['sector_flags']
    stage = profile['life_cycle_stage']
    leverage = profile['leverage']

    if flags['is_financial_firm']:
        return {'primary': 'Dividend Discount Model / Excess Return on Equity',
                'secondary': 'P/B multiple vs peers',
                'note': 'Do not use FCFF as the primary model — regulatory capital requirements make free cash flow ill-defined for a financial firm.'}
    if flags['is_reit']:
        return {'primary': 'FFO/AFFO-based Dividend Discount Model',
                'secondary': 'Net Asset Value (NAV) approach',
                'note': 'Use funds from operations rather than net income; depreciation is largely not a real economic cost for a REIT.'}
    if flags['is_utility'] and not leverage['is_solvency_risk']:
        return {'primary': 'DCF (FCFE) on the regulated asset base',
                'secondary': 'Dividend Discount Model',
                'note': 'Regulated, stable cash flows and a high payout ratio make FCFE/DDM more reliable than an unlevered FCFF build.'}
    if stage == 'startup':
        return {'primary': 'Venture Capital Method / Precedent Transactions',
                'secondary': None,
                'note': 'Too few data points and too much uncertainty for a bottom-up DCF to be reliable at this stage.'}
    if stage == 'distress':
        return {'primary': 'Asset-based / Liquidation value, or scenario-weighted Distressed DCF',
                'secondary': 'Comparable distressed-company transactions',
                'note': f"Interest coverage / loss history signal solvency risk (consecutive loss years: {leverage['consecutive_loss_years']}). "
                        "Going-concern assumptions should not be taken for granted — weight restructuring/liquidation scenarios explicitly."}
    if not profile['has_positive_reliable_fcf']:
        return {'primary': 'Revenue-based DCF',
                'secondary': 'EV/Revenue multiple' if has_enough_comparables else None,
                'note': 'Earnings and free cash flow are too unreliable to anchor a multiple on — avoid EV/EBITDA and P/E.'}
    if stage == 'decline':
        return {'primary': 'Asset-based / Liquidation Value',
                'secondary': 'DCF, with caution',
                'note': 'Do not assume aggressive perpetual going-concern growth for a declining business.'}

    note_parts = []
    if profile['is_emerging_market']:
        note_parts.append('Add the country risk premium to the equity risk premium.')
    if flags['is_commodity_linked']:
        note_parts.append('Anchor revenue/margin assumptions to a normalized commodity price deck, not spot prices.')
    if leverage['is_highly_levered']:
        note_parts.append('Leverage is high — cross-check the FCFF value against a levered FCFE build.')
    if profile['is_cyclical']:
        note_parts.append('Margins are cyclical — normalize the terminal-year margin rather than extrapolating the latest year.')
    return {'primary': 'DCF (FCFF)',
            'secondary': 'Market Multiples (EV/EBITDA, P/E)' if has_enough_comparables else 'DCF only',
            'note': ' '.join(note_parts) if note_parts else ''}



def apply_country_risk_premium_to_erp(base_erp, profile):
    return base_erp + profile.get('country_risk_premium', 0) if profile.get('is_emerging_market') else base_erp


def warn_if_dcf_not_recommended(decision):
    if decision['primary'] != 'DCF (FCFF)':
        warnings.warn(f"DCF (FCFF) is not the recommended primary approach: {decision['primary']}. {decision.get('note','')}", stacklevel=2)
        return False
    return True
