from __future__ import annotations
import numpy as np
import pandas as pd


def dynamic_converger(current, expected, number_of_steps, period_to_begin_to_converge):
    number_of_steps = int(number_of_steps)
    period_to_begin_to_converge = int(period_to_begin_to_converge)
    if period_to_begin_to_converge < 1:
        raise ValueError("period_to_begin_to_converge must be at least 1")
    if period_to_begin_to_converge > number_of_steps:
        raise ValueError("period_to_begin_to_converge must be less than or equal to number_of_steps")
    phase2_len = number_of_steps - period_to_begin_to_converge + 1
    array_phase1 = np.array([current] * (period_to_begin_to_converge - 1), dtype=float)
    array_phase2 = np.array([expected], dtype=float) if phase2_len == 1 else np.linspace(current, expected, phase2_len)
    return pd.Series(np.concatenate((array_phase1, array_phase2)))


def dynamic_converger_multiple_phase(growth_rates_for_each_cycle, length_of_each_cycle, convergence_periods):
    results = []
    for i in range(len(length_of_each_cycle)):
        results.append(dynamic_converger(
            current=growth_rates_for_each_cycle[i][0],
            expected=growth_rates_for_each_cycle[i][1],
            number_of_steps=length_of_each_cycle[i],
            period_to_begin_to_converge=convergence_periods[i],
        ))
    return pd.concat(results, ignore_index=True)


def revenue_projector_multi_phase(revenue_base, revenue_growth_rate_cycle1_begin, revenue_growth_rate_cycle1_end,
                                  revenue_growth_rate_cycle2_begin, revenue_growth_rate_cycle2_end,
                                  revenue_growth_rate_cycle3_begin, revenue_growth_rate_cycle3_end=0.028,
                                  length_of_cycle1=3, length_of_cycle2=4, length_of_cycle3=3,
                                  revenue_convergence_periods_cycle1=1, revenue_convergence_periods_cycle2=1,
                                  revenue_convergence_periods_cycle3=1, **legacy_kwargs):
    # Backward-compatible misspelled keyword aliases from the notebook.
    length_of_cycle1 = legacy_kwargs.get('length_of_cylcle1', length_of_cycle1)
    length_of_cycle2 = legacy_kwargs.get('length_of_cylcle2', length_of_cycle2)
    length_of_cycle3 = legacy_kwargs.get('length_of_cylcle3', length_of_cycle3)
    revenue_convergence_periods_cycle1 = legacy_kwargs.get('revenue_convergance_periods_cycle1', revenue_convergence_periods_cycle1)
    revenue_convergence_periods_cycle2 = legacy_kwargs.get('revenue_convergance_periods_cycle2', revenue_convergence_periods_cycle2)
    revenue_convergence_periods_cycle3 = legacy_kwargs.get('revenue_convergance_periods_cycle3', revenue_convergence_periods_cycle3)

    growth = dynamic_converger_multiple_phase(
        growth_rates_for_each_cycle=[
            [revenue_growth_rate_cycle1_begin, revenue_growth_rate_cycle1_end],
            [revenue_growth_rate_cycle2_begin, revenue_growth_rate_cycle2_end],
            [revenue_growth_rate_cycle3_begin, revenue_growth_rate_cycle3_end],
        ],
        length_of_each_cycle=[length_of_cycle1, length_of_cycle2, length_of_cycle3],
        convergence_periods=[revenue_convergence_periods_cycle1, revenue_convergence_periods_cycle2, revenue_convergence_periods_cycle3],
    )
    revenues = revenue_base * (1 + growth).cumprod()
    return revenues, growth


def operating_margin_projector(current_operating_margin, terminal_operating_margin, valuation_interval_in_years=10,
                               year_operating_margin_begins_to_converge_to_terminal_operating_margin=5):
    return dynamic_converger(current_operating_margin, terminal_operating_margin, valuation_interval_in_years,
                             year_operating_margin_begins_to_converge_to_terminal_operating_margin)


def tax_rate_projector(current_effective_tax_rate, marginal_tax_rate, valuation_interval_in_years=10,
                       year_effective_tax_rate_begin_to_converge_marginal_tax_rate=5):
    return dynamic_converger(current_effective_tax_rate, marginal_tax_rate, valuation_interval_in_years,
                             year_effective_tax_rate_begin_to_converge_marginal_tax_rate)


def sales_to_capital_projector(current_sales_to_capital_ratio, terminal_sales_to_capital_ratio, valuation_interval_in_years=10,
                               year_sales_to_capital_begins_to_converge_to_terminal_sales_to_capital=3):
    return dynamic_converger(current_sales_to_capital_ratio, terminal_sales_to_capital_ratio, valuation_interval_in_years,
                             year_sales_to_capital_begins_to_converge_to_terminal_sales_to_capital)


def cost_of_capital_projector(unlevered_beta, terminal_unlevered_beta, current_pretax_cost_of_debt, terminal_pretax_cost_of_debt,
                              equity_value, debt_value, marginal_tax_rate=.21, risk_free_rate=0.015, ERP=0.055,
                              valuation_interval_in_years=10, year_beta_begins_to_converge_to_terminal_beta=5,
                              year_cost_of_debt_begins_to_converge_to_terminal_cost_of_debt=5):
    equity_value = max(float(equity_value), 1e-9)
    company_beta = unlevered_beta * (1 + (1 - marginal_tax_rate) * (debt_value / equity_value))
    terminal_beta = terminal_unlevered_beta * (1 + (1 - marginal_tax_rate) * (debt_value / equity_value))
    beta_series = dynamic_converger(company_beta, terminal_beta, valuation_interval_in_years, year_beta_begins_to_converge_to_terminal_beta)
    pretax_debt_series = dynamic_converger(current_pretax_cost_of_debt, terminal_pretax_cost_of_debt, valuation_interval_in_years,
                                           year_cost_of_debt_begins_to_converge_to_terminal_cost_of_debt)
    total_capital = equity_value + debt_value
    equity_to_capital = equity_value / total_capital if total_capital else 1.0
    debt_to_capital = debt_value / total_capital if total_capital else 0.0
    after_tax_debt = pretax_debt_series * (1 - marginal_tax_rate)
    cost_of_equity = risk_free_rate + beta_series * ERP
    wacc = equity_to_capital * cost_of_equity + debt_to_capital * after_tax_debt
    return wacc, beta_series, terminal_beta, cost_of_equity, after_tax_debt


def reinvestment_projector(revenue_base, projected_revenues, sales_to_capital_ratios, asset_liquidation_during_negative_growth=0):
    reinvestment = (pd.concat([pd.Series([revenue_base]), projected_revenues], ignore_index=True).diff().dropna().reset_index(drop=True)
                    / sales_to_capital_ratios.reset_index(drop=True))
    return reinvestment.where(reinvestment > 0, reinvestment * asset_liquidation_during_negative_growth)


def _alias_params(params):
    aliases = {
        'length_of_cycle1': 'length_of_cylcle1',
        'length_of_cycle2': 'length_of_cylcle2',
        'length_of_cycle3': 'length_of_cylcle3',
        'revenue_convergence_periods_cycle1': 'revenue_convergance_periods_cycle1',
        'revenue_convergence_periods_cycle2': 'revenue_convergance_periods_cycle2',
        'revenue_convergence_periods_cycle3': 'revenue_convergance_periods_cycle3',
    }
    out = dict(params)
    for canonical, legacy in aliases.items():
        if canonical in out and legacy not in out:
            out[legacy] = out[canonical]
    return out


def valuator_multi_phase(**kwargs):
    p = _alias_params(kwargs)
    length1, length2, length3 = int(p['length_of_cylcle1']), int(p['length_of_cylcle2']), int(p['length_of_cylcle3'])
    valuation_interval_in_years = length1 + length2 + length3
    terminal_growth_rate = float(p['revenue_growth_rate_cycle3_end'])
    risk_free_rate = float(p['risk_free_rate'])
    if p.get('enforce_terminal_growth_below_risk_free', True) and terminal_growth_rate > risk_free_rate + 1e-9:
        raise ValueError(f"terminal_growth_rate ({terminal_growth_rate:.4f}) exceeds risk_free_rate ({risk_free_rate:.4f}).")

    projected_wacc, projected_beta, terminal_beta, projected_cost_of_equity, projected_after_tax_cost_of_debt = cost_of_capital_projector(
        unlevered_beta=p['unlevered_beta'], terminal_unlevered_beta=p['terminal_unlevered_beta'],
        current_pretax_cost_of_debt=p['current_pretax_cost_of_debt'], terminal_pretax_cost_of_debt=p['terminal_pretax_cost_of_debt'],
        equity_value=p['equity_value'], debt_value=p['debt_value'], marginal_tax_rate=p['marginal_tax_rate'],
        risk_free_rate=risk_free_rate, ERP=p['ERP'], valuation_interval_in_years=valuation_interval_in_years,
        year_beta_begins_to_converge_to_terminal_beta=p['year_beta_begins_to_converge_to_terminal_beta'],
        year_cost_of_debt_begins_to_converge_to_terminal_cost_of_debt=p['year_cost_of_debt_begins_to_converge_to_terminal_cost_of_debt'])

    projected_revenues, projected_revenue_growth = revenue_projector_multi_phase(
        revenue_base=p['revenue_base'], revenue_growth_rate_cycle1_begin=p['revenue_growth_rate_cycle1_begin'],
        revenue_growth_rate_cycle1_end=p['revenue_growth_rate_cycle1_end'], revenue_growth_rate_cycle2_begin=p['revenue_growth_rate_cycle2_begin'],
        revenue_growth_rate_cycle2_end=p['revenue_growth_rate_cycle2_end'], revenue_growth_rate_cycle3_begin=p['revenue_growth_rate_cycle3_begin'],
        revenue_growth_rate_cycle3_end=p['revenue_growth_rate_cycle3_end'], length_of_cylcle1=length1, length_of_cylcle2=length2,
        length_of_cylcle3=length3, revenue_convergance_periods_cycle1=p['revenue_convergance_periods_cycle1'],
        revenue_convergance_periods_cycle2=p['revenue_convergance_periods_cycle2'], revenue_convergance_periods_cycle3=p['revenue_convergance_periods_cycle3'])

    projected_tax_rates = tax_rate_projector(p['current_effective_tax_rate'], p['marginal_tax_rate'], valuation_interval_in_years,
                                             p['year_effective_tax_rate_begin_to_converge_marginal_tax_rate'])
    sales_to_capital_ratios = sales_to_capital_projector(p['current_sales_to_capital_ratio'], p['terminal_sales_to_capital_ratio'],
                                                         valuation_interval_in_years,
                                                         p['year_sales_to_capital_begins_to_converge_to_terminal_sales_to_capital'])
    projected_reinvestment = reinvestment_projector(p['revenue_base'], projected_revenues, sales_to_capital_ratios,
                                                    p.get('asset_liquidation_during_negative_growth', 0))
    current_invested_capital = p.get('current_invested_capital', 'implicit')
    if current_invested_capital == 'implicit':
        current_invested_capital = p['revenue_base'] / p['current_sales_to_capital_ratio']
    invested_capital = projected_reinvestment.copy().reset_index(drop=True)
    invested_capital.iloc[0] = invested_capital.iloc[0] + current_invested_capital
    invested_capital = invested_capital.cumsum()

    margins = operating_margin_projector(p['current_operating_margin'], p['terminal_operating_margin'], valuation_interval_in_years,
                                         p['year_operating_margin_begins_to_converge_to_terminal_operating_margin'])
    ebit = projected_revenues * margins
    after_tax_oi = ebit * (1 - projected_tax_rates)
    fcff = after_tax_oi - projected_reinvestment
    roic = after_tax_oi / invested_capital

    terminal_wacc = float(projected_wacc.iloc[-1])
    terminal_coe = float(projected_cost_of_equity.iloc[-1])
    additional_roic_spread = p.get('additional_return_on_cost_of_capital_in_perpetuity', 0.0)
    terminal_reinvestment_rate = 0 if terminal_growth_rate < 0 else terminal_growth_rate / (terminal_wacc + additional_roic_spread)
    terminal_revenue = float(projected_revenues.iloc[-1]) * (1 + terminal_growth_rate)
    terminal_ebit = terminal_revenue * p['terminal_operating_margin']
    terminal_after_tax_oi = terminal_ebit * (1 - p['marginal_tax_rate'])
    terminal_reinvestment = terminal_after_tax_oi * terminal_reinvestment_rate
    terminal_fcff = terminal_after_tax_oi - terminal_reinvestment
    if terminal_wacc <= terminal_growth_rate:
        raise ValueError("Terminal WACC must exceed terminal growth rate.")
    terminal_value = terminal_fcff / (terminal_wacc - terminal_growth_rate)

    cum_wacc = (1 + projected_wacc).cumprod()
    cum_coe = (1 + projected_cost_of_equity).cumprod()
    terminal_discount_factor = float(cum_wacc.iloc[-1])
    terminal_equity_discount_factor = float(cum_coe.iloc[-1])

    df = pd.DataFrame({
        'cumWACC': pd.concat([cum_wacc, pd.Series([terminal_discount_factor])], ignore_index=True),
        'cumCostOfEquity': pd.concat([cum_coe, pd.Series([terminal_equity_discount_factor])], ignore_index=True),
        'beta': pd.concat([projected_beta, pd.Series([terminal_beta])], ignore_index=True),
        'ERP': p['ERP'],
        'projected_after_tax_cost_of_debt': pd.concat([projected_after_tax_cost_of_debt, pd.Series([projected_after_tax_cost_of_debt.iloc[-1]])], ignore_index=True),
        'revenueGrowth': pd.concat([projected_revenue_growth, pd.Series([terminal_growth_rate])], ignore_index=True),
        'revenues': pd.concat([projected_revenues, pd.Series([terminal_revenue])], ignore_index=True),
        'margins': pd.concat([margins, pd.Series([p['terminal_operating_margin']])], ignore_index=True),
        'ebit': pd.concat([ebit, pd.Series([terminal_ebit])], ignore_index=True),
        'sales_to_capital_ratio': pd.concat([sales_to_capital_ratios, pd.Series([p['terminal_sales_to_capital_ratio']])], ignore_index=True),
        'taxRate': pd.concat([projected_tax_rates, pd.Series([p['marginal_tax_rate']])], ignore_index=True),
        'afterTaxOperatingIncome': pd.concat([after_tax_oi, pd.Series([terminal_after_tax_oi])], ignore_index=True),
        'reinvestment': pd.concat([projected_reinvestment, pd.Series([terminal_reinvestment])], ignore_index=True),
        'invested_capital': pd.concat([invested_capital, pd.Series([np.nan])], ignore_index=True),
        'ROIC': pd.concat([roic, pd.Series([terminal_wacc + additional_roic_spread])], ignore_index=True),
        'FCFF': pd.concat([fcff, pd.Series([terminal_value])], ignore_index=True),
    })
    df['reinvestmentRate'] = df['reinvestment'] / df['afterTaxOperatingIncome'].replace(0, np.nan)
    df['PVFCFF'] = df['FCFF'] / df['cumWACC']
    value_of_operating_assets = float(df['PVFCFF'].sum())
    firm_value = value_of_operating_assets + p.get('cash_and_non_operating_asset', 0.0)
    equity_pv = firm_value - p['debt_value']
    return {
        'valuation': df,
        'firm_value': firm_value,
        'equity_value': equity_pv,
        'cash_and_non_operating_asset': p.get('cash_and_non_operating_asset', 0.0),
        'debt_value': p['debt_value'],
        'value_of_operating_assets': value_of_operating_assets,
        'terminal_value_undiscounted': terminal_value,
    }


def point_estimate_describer(base_case_valuation):
    df = base_case_valuation['valuation'].reset_index(drop=True).copy()
    df['Year'] = np.arange(1, len(df) + 1).astype(object)
    df.loc[df['Year'] == df['Year'].max(), 'Year'] = 'Terminal'
    return df.set_index('Year')
