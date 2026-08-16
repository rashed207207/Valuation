from equity_research_ai.valuation.dcf import dynamic_converger, valuator_multi_phase


def test_dynamic_converger_edge_case():
    edge = dynamic_converger(0.10, 0.03, number_of_steps=5, period_to_begin_to_converge=5)
    assert abs(edge.iloc[-1] - 0.03) < 1e-9


def test_basic_valuation_runs():
    params = dict(
        risk_free_rate=.042, ERP=.045, equity_value=100.0, debt_value=16.0, cash_and_non_operating_asset=3.0,
        unlevered_beta=0.85, terminal_unlevered_beta=0.85, year_beta_begins_to_converge_to_terminal_beta=3,
        current_pretax_cost_of_debt=.055, terminal_pretax_cost_of_debt=.05,
        year_cost_of_debt_begins_to_converge_to_terminal_cost_of_debt=3,
        current_effective_tax_rate=.20, marginal_tax_rate=.24,
        year_effective_tax_rate_begin_to_converge_marginal_tax_rate=2,
        revenue_base=37.0, revenue_growth_rate_cycle1_begin=.08, revenue_growth_rate_cycle1_end=.06,
        revenue_growth_rate_cycle2_begin=.055, revenue_growth_rate_cycle2_end=.045,
        revenue_growth_rate_cycle3_begin=.042, revenue_growth_rate_cycle3_end=.042,
        revenue_convergance_periods_cycle1=1, revenue_convergance_periods_cycle2=1,
        revenue_convergance_periods_cycle3=1, length_of_cylcle1=3, length_of_cylcle2=4, length_of_cylcle3=3,
        current_sales_to_capital_ratio=1.2, terminal_sales_to_capital_ratio=1.3,
        year_sales_to_capital_begins_to_converge_to_terminal_sales_to_capital=2,
        current_operating_margin=.10, terminal_operating_margin=.14,
        year_operating_margin_begins_to_converge_to_terminal_operating_margin=2,
    )
    res = valuator_multi_phase(**params)
    assert 'equity_value' in res
    assert res['valuation'].shape[0] == 11
