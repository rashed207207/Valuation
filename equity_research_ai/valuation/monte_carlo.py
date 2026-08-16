from __future__ import annotations
import numpy as np
import pandas as pd
import scipy.stats as st
from equity_research_ai.valuation.dcf import valuator_multi_phase


def gaussian_copula_sample(marginals, corr_pairs, sample_size, seed=42):
    rng = np.random.default_rng(seed)
    names = list(marginals.keys())
    idx = {name: i for i, name in enumerate(names)}
    R = np.eye(len(names))
    for a, b, rho in corr_pairs:
        if a in idx and b in idx:
            i, j = idx[a], idx[b]
            R[i, j] = rho; R[j, i] = rho
    eigvals, eigvecs = np.linalg.eigh(R)
    if np.any(eigvals < 0):
        eigvals = np.clip(eigvals, 1e-8, None)
        R = eigvecs @ np.diag(eigvals) @ eigvecs.T
        d = np.sqrt(np.diag(R)); R = R / np.outer(d, d)
    z = rng.multivariate_normal(np.zeros(len(names)), R, size=sample_size)
    u = st.norm.cdf(z)
    return pd.DataFrame({name: marginals[name].ppf(u[:, idx[name]]) for name in names})


def default_marginals(params):
    return {
        'risk_free_rate': st.norm(params['risk_free_rate'], .003),
        'ERP': st.norm(params['ERP'], .003),
        'unlevered_beta': st.triang(0.5, loc=params['unlevered_beta']*0.85, scale=max(params['unlevered_beta']*0.3, 0.01)),
        'terminal_unlevered_beta': st.triang(0.5, loc=params['terminal_unlevered_beta']*0.85, scale=max(params['terminal_unlevered_beta']*0.3, 0.01)),
        'terminal_operating_margin': st.triang(0.5, loc=max(params['terminal_operating_margin']-0.03, 0.01), scale=0.06),
        'revenue_growth_rate_cycle1_begin': st.norm(params['revenue_growth_rate_cycle1_begin'], .012),
        'terminal_growth_spread': st.halfnorm(loc=0.0, scale=.006),
        'current_sales_to_capital_ratio': st.triang(0.5, loc=max(params['current_sales_to_capital_ratio']-0.3, 0.2), scale=0.6),
        'terminal_sales_to_capital_ratio': st.triang(0.5, loc=max(params['terminal_sales_to_capital_ratio']-0.3, 0.2), scale=0.6),
    }


def monte_carlo_valuator_multi_phase(fixed_params, marginals=None, corr_pairs=None, sample_size=1000, seed=42):
    marginals = marginals or default_marginals(fixed_params)
    corr_pairs = corr_pairs or [
        ['terminal_growth_spread', 'risk_free_rate', -.7],
        ['terminal_operating_margin', 'unlevered_beta', -.2],
        ['unlevered_beta', 'terminal_unlevered_beta', .85],
    ]
    df = gaussian_copula_sample(marginals, corr_pairs, sample_size, seed)
    if 'terminal_growth_spread' in df.columns and 'risk_free_rate' in df.columns:
        df['revenue_growth_rate_cycle3_end'] = df['risk_free_rate'] - df['terminal_growth_spread'].clip(lower=0.0)
    values = []
    for _, row in df.iterrows():
        params = dict(fixed_params)
        params.update(row.to_dict())
        try:
            values.append(valuator_multi_phase(**params)['equity_value'])
        except Exception:
            values.append(np.nan)
    df['equity_valuation'] = values
    return df.dropna(subset=['equity_valuation']).reset_index(drop=True)


def valuation_describer(df_mc, current_market_cap, shares_outstanding=1):
    percentiles = np.arange(0, 110, 10)
    vals = np.percentile(df_mc['equity_valuation'], percentiles)
    df = pd.DataFrame({'percentile': percentiles, 'intrinsic_equity_value': vals})
    df['current_market_cap'] = current_market_cap
    df['price_per_share'] = current_market_cap / shares_outstanding if shares_outstanding else np.nan
    df['intrinsic_value_per_share'] = df['intrinsic_equity_value'] / shares_outstanding if shares_outstanding else np.nan
    df['price_to_value'] = df['current_market_cap'] / df['intrinsic_equity_value']
    df['upside_downside_pct'] = df['intrinsic_equity_value'] / df['current_market_cap'] - 1
    return df


def monte_carlo_summary(df_mc, current_market_cap):
    return {
        'p10': float(np.percentile(df_mc['equity_valuation'], 10)),
        'p50': float(np.percentile(df_mc['equity_valuation'], 50)),
        'p90': float(np.percentile(df_mc['equity_valuation'], 90)),
        'probability_undervalued': float((df_mc['equity_valuation'] > current_market_cap).mean()),
        'scenarios': int(len(df_mc)),
    }
