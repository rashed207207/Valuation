from equity_research_ai.valuation.monte_carlo import monte_carlo_valuator_multi_phase, monte_carlo_summary

class RiskAgent:
    def run_monte_carlo(self, base_params, current_market_cap, sample_size=1000, seed=42):
        df_mc = monte_carlo_valuator_multi_phase(base_params, sample_size=sample_size, seed=seed)
        summary = monte_carlo_summary(df_mc, current_market_cap)
        summary['sample'] = df_mc
        return summary
