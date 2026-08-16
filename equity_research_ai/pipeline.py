from __future__ import annotations
import argparse
from pathlib import Path
from dotenv import load_dotenv
from equity_research_ai.models import CompanyAnalysis
from equity_research_ai.data.yahoo import fetch_company_data
from equity_research_ai.valuation.profile import build_company_profile, select_valuation_approach
from equity_research_ai.rag.local_retriever import LocalSECRetriever
from equity_research_ai.agents.business_analyst import BusinessAnalystAgent
from equity_research_ai.agents.valuation_analyst import ValuationAnalystAgent
from equity_research_ai.agents.risk_analyst import RiskAgent
from equity_research_ai.agents.expectations_analyst import ExpectationsAgent
from equity_research_ai.agents.report_writer import ReportWriter
from equity_research_ai.valuation.dcf import valuator_multi_phase

COUNTRY_RISK_PREMIUM_TABLE = {'United States': 0.0, 'Egypt': 0.09, 'Saudi Arabia': 0.02}


def run_analysis(ticker: str, sec_data: str | None = None, use_llm: bool = False, sample_size: int = 1000):
    load_dotenv()
    data = fetch_company_data(ticker)
    profile = build_company_profile(
        revenue_history=data['revenue_history'],
        net_income_history=data['net_income_history'],
        ebit_history=data['ebit_history'],
        operating_cash_flow_history=data['operating_cash_flow_history'],
        capex_history=data['capex_history'],
        rd_expense_history=data['rd_expense_history'],
        dividends_paid_history=data['dividends_paid_history'],
        gics_sector=data['gics_sector'],
        years_since_ipo=data['years_since_ipo'],
        country=data['country'],
        country_risk_premium_table=COUNTRY_RISK_PREMIUM_TABLE,
    )
    decision = select_valuation_approach(profile, has_enough_comparables=True)
    retriever = LocalSECRetriever(sec_data) if sec_data else None
    business = BusinessAnalystAgent(retriever=retriever, use_llm=use_llm).analyze(ticker)
    assumptions = ValuationAnalystAgent().generate_assumptions(data, profile, business)
    dcf = valuator_multi_phase(**assumptions)
    mc = RiskAgent().run_monte_carlo(assumptions, data['market_cap'], sample_size=sample_size, seed=42)
    mc_public = {k: v for k, v in mc.items() if k != 'sample'}
    expectations = ExpectationsAgent().analyze(assumptions, data['market_cap'])
    analysis = CompanyAnalysis(
        ticker=ticker.upper(), company_name=data.get('company_name', ticker), company_data=data,
        profile=profile, valuation_decision=decision, business_analysis=business,
        valuation_assumptions=assumptions, dcf_result=dcf,
        monte_carlo_result=mc_public, implied_market_expectations=expectations,
    )
    analysis.final_report = ReportWriter().write(analysis)
    return analysis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ticker')
    parser.add_argument('--sec-data', default=None, help='Path to sec-mdna-rag data/filings JSONL directory')
    parser.add_argument('--use-llm', action='store_true')
    parser.add_argument('--sample-size', type=int, default=1000)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    analysis = run_analysis(args.ticker, args.sec_data, args.use_llm, args.sample_size)
    print(analysis.final_report)
    if args.out:
        Path(args.out).write_text(analysis.final_report, encoding='utf-8')
        print(f"\nSaved report to {args.out}")

if __name__ == '__main__':
    main()
