from __future__ import annotations
from scipy.optimize import brentq
from equity_research_ai.valuation.dcf import valuator_multi_phase


def solve_for_implied_assumption(base_params, target_equity_value, param_name, bounds):
    lo, hi = bounds
    def objective(x):
        params = dict(base_params)
        params[param_name] = x
        return valuator_multi_phase(**params)['equity_value'] - target_equity_value
    f_lo, f_hi = objective(lo), objective(hi)
    if f_lo * f_hi > 0:
        raise ValueError(f"No solution for {param_name} in bounds {bounds}.")
    return brentq(objective, lo, hi, xtol=1e-6)


def implied_expectations_summary(base_params, market_cap):
    out = {}
    try:
        out['implied_revenue_growth_cycle1_begin'] = solve_for_implied_assumption(base_params, market_cap, 'revenue_growth_rate_cycle1_begin', (-0.10, 0.80))
    except Exception as e:
        out['implied_revenue_growth_cycle1_begin_error'] = str(e)
    try:
        out['implied_terminal_operating_margin'] = solve_for_implied_assumption(base_params, market_cap, 'terminal_operating_margin', (0.02, 0.55))
    except Exception as e:
        out['implied_terminal_operating_margin_error'] = str(e)
    return out
