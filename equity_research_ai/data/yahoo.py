from __future__ import annotations
import datetime
import numpy as np
import pandas as pd

_OFFLINE_SNAPSHOT = {
    'SBUX': {
        'revenue_history': [29.06, 32.25, 35.98, 36.18, 37.18],
        'net_income_history': [4.20, 3.28, 4.12, 3.76, 1.86],
        'ebit_history': [4.87, 4.62, 5.87, 5.41, 3.58],
        'operating_cash_flow_history': [5.02, 4.66, 5.99, 5.30, 4.35],
        'capex_history': [1.83, 2.80, 2.63, 2.44, 1.62],
        'rd_expense_history': [0.0, 0.0, 0.0, 0.0, 0.0],
        'dividends_paid_history': [1.85, 2.11, 2.29, 2.44, 2.65],
        'gics_sector': 'Consumer Cyclical',
        'country': 'United States',
        'years_since_ipo': 33,
        'market_cap': 105.4,
        'total_debt': 16.1,
        'cash': 3.6,
        'beta': 0.95,
        'shares_outstanding': 1.14,
        'current_price': 92.55,
        'invested_capital_book': 4.1,
        'company_name': 'Starbucks Corporation',
        'pretax_income_history': [],
        'tax_provision_history': [],
        'interest_expense_history': [],
    }
}

def _get_row(df, candidates):
    if df is None or getattr(df, 'empty', True):
        return pd.Series(dtype=float)
    for name in candidates:
        if name in df.index:
            return df.loc[name].sort_index().dropna()
    for idx in df.index:
        for name in candidates:
            if name.lower() in str(idx).lower():
                return df.loc[idx].sort_index().dropna()
    return pd.Series(dtype=float)


def _annual_from_quarterly(t, income_index_candidates, years=5):
    q = _get_row(t.quarterly_income_stmt, income_index_candidates)
    if len(q) < 4:
        return pd.Series(dtype=float)
    annual_like = q.sort_index().rolling(4).sum().dropna()
    return annual_like.iloc[::4].tail(years)


def _validate_company_data(data, ticker_symbol):
    problems = []
    if not data['revenue_history'] or any((r is None) or (r != r) or (r <= 0) for r in data['revenue_history']):
        problems.append(f"Revenue data missing or invalid: {data['revenue_history']}")
    if not data.get('shares_outstanding') or data['shares_outstanding'] <= 0:
        problems.append(f"Shares outstanding missing or invalid: {data.get('shares_outstanding')}")
    if not data.get('market_cap') or data['market_cap'] <= 0:
        problems.append(f"Market cap missing or invalid: {data.get('market_cap')}")
    if problems:
        raise ValueError(f"Data for {ticker_symbol} is not reliable enough:\n  - " + "\n  - ".join(problems))


def _fetch_company_data_live(ticker_symbol, years=5):
    import yfinance as yf
    t = yf.Ticker(ticker_symbol)
    info = t.info
    if not info or len(info) < 3:
        raise ConnectionError(f"yfinance returned little or no info for {ticker_symbol}")

    revenue = _get_row(t.income_stmt, ['Total Revenue', 'TotalRevenue', 'Operating Revenue'])
    if len(revenue) < 2:
        revenue = _annual_from_quarterly(t, ['Total Revenue', 'TotalRevenue', 'Operating Revenue'], years=years)
    net_income = _get_row(t.income_stmt, ['Net Income', 'NetIncome', 'Net Income Common Stockholders'])
    ebit = _get_row(t.income_stmt, ['EBIT', 'Operating Income'])
    rd_expense = _get_row(t.income_stmt, ['Research And Development'])
    pretax_income = _get_row(t.income_stmt, ['Pretax Income', 'Income Before Tax'])
    tax_provision = _get_row(t.income_stmt, ['Tax Provision', 'Income Tax Expense', 'Income Tax Expense Benefit'])
    interest_expense = _get_row(t.income_stmt, ['Interest Expense', 'Interest Expense Non Operating', 'Net Interest Income'])
    ocf = _get_row(t.cashflow, ['Operating Cash Flow', 'Cash Flow From Continuing Operating Activities'])
    capex = _get_row(t.cashflow, ['Capital Expenditure', 'Purchase Of PPE'])
    div_paid = _get_row(t.cashflow, ['Cash Dividends Paid', 'Common Stock Dividend Paid'])
    total_debt_bs = _get_row(t.balance_sheet, ['Total Debt'])
    total_equity_bs = _get_row(t.balance_sheet, ['Stockholders Equity', 'Total Equity Gross Minority Interest'])
    cash_bs = _get_row(t.balance_sheet, ['Cash And Cash Equivalents', 'Cash Cash Equivalents And Short Term Investments'])

    core_df = pd.concat({'revenue': revenue, 'net_income': net_income, 'ebit': ebit}, axis=1).sort_index().dropna(how='any')
    n = min(years, len(core_df))
    if n == 0:
        raise ValueError(f"Could not retrieve usable income statement data for {ticker_symbol}.")
    core_df = core_df.tail(n)
    aligned_index = core_df.index

    def _reindex_secondary(series, fallback=0.0):
        if len(series) == 0:
            return pd.Series(fallback, index=aligned_index)
        reindexed = series.reindex(series.index.union(aligned_index)).sort_index().ffill().bfill().reindex(aligned_index)
        return reindexed.fillna(fallback)

    ocf_aligned = _reindex_secondary(ocf, fallback=np.nan).fillna(0.0)
    capex_aligned = _reindex_secondary(capex, fallback=np.nan).fillna(0.0)
    interest_aligned = _reindex_secondary(interest_expense, fallback=np.nan).fillna(0.0)
    rd_aligned = _reindex_secondary(rd_expense, fallback=0.0)
    div_aligned = _reindex_secondary(div_paid, fallback=0.0)

    tax_pair = pd.concat({'pretax': pretax_income, 'tax': tax_provision}, axis=1).sort_index().dropna(how='any')
    if len(tax_pair) > 0:
        tax_pair = tax_pair.tail(min(n, len(tax_pair)))
        pretax_income_history = (tax_pair['pretax'] / 1e9).tolist()
        tax_provision_history = (tax_pair['tax'] / 1e9).tolist()
    else:
        pretax_income_history, tax_provision_history = [], []

    first_trade_epoch = info.get('firstTradeDateEpochUtc')
    years_since_ipo = datetime.datetime.now().year - datetime.datetime.fromtimestamp(first_trade_epoch).year if first_trade_epoch else 15

    invested_capital_book = None
    if len(total_debt_bs) > 0 and len(total_equity_bs) > 0 and len(cash_bs) > 0:
        invested_capital_book = (total_debt_bs.iloc[-1] + total_equity_bs.iloc[-1] - cash_bs.iloc[-1]) / 1e9

    to_bn = lambda s: (s / 1e9).tolist()
    return {
        'revenue_history': to_bn(core_df['revenue']),
        'net_income_history': to_bn(core_df['net_income']),
        'ebit_history': to_bn(core_df['ebit']),
        'operating_cash_flow_history': to_bn(ocf_aligned),
        'capex_history': [abs(v) for v in to_bn(capex_aligned)],
        'rd_expense_history': to_bn(rd_aligned),
        'dividends_paid_history': to_bn(div_aligned),
        'pretax_income_history': pretax_income_history,
        'tax_provision_history': tax_provision_history,
        'interest_expense_history': [abs(v) for v in to_bn(interest_aligned)],
        'gics_sector': info.get('sector', 'Unknown'),
        'country': info.get('country', 'United States'),
        'years_since_ipo': years_since_ipo,
        'market_cap': info.get('marketCap', 0) / 1e9,
        'total_debt': info.get('totalDebt', 0) / 1e9,
        'cash': info.get('totalCash', 0) / 1e9,
        'beta': info.get('beta', 1.0) or 1.0,
        'shares_outstanding': info.get('sharesOutstanding', 0) / 1e9,
        'current_price': info.get('currentPrice', info.get('regularMarketPrice')),
        'invested_capital_book': invested_capital_book,
        'company_name': info.get('longName', ticker_symbol),
    }


def fetch_company_data(ticker_symbol, years=5, allow_offline_fallback=True):
    ticker_upper = ticker_symbol.upper()
    try:
        data = _fetch_company_data_live(ticker_symbol, years=years)
        _validate_company_data(data, ticker_symbol)
        return data
    except Exception as e:
        if allow_offline_fallback and ticker_upper in _OFFLINE_SNAPSHOT:
            print(f"Using offline snapshot for {ticker_upper}; live yfinance failed: {type(e).__name__}: {str(e)[:120]}")
            return dict(_OFFLINE_SNAPSHOT[ticker_upper])
        raise


def derive_dynamic_valuation_assumptions(data, fallback_tax_rate=0.21, fallback_cost_of_debt=0.05, fallback_sales_to_capital=1.2):
    revenue_history = np.array(data['revenue_history'], dtype=float)
    ebit_history = np.array(data['ebit_history'], dtype=float)
    revenue_ttm = revenue_history[-1]
    invested_capital = data.get('invested_capital_book')
    current_sales_to_capital_ratio = revenue_ttm / invested_capital if invested_capital is not None and invested_capital > 0 else fallback_sales_to_capital
    pretax = data.get('pretax_income_history') or []
    tax = data.get('tax_provision_history') or []
    if pretax and tax and pretax[-1] > 0:
        etr = min(max(tax[-1] / pretax[-1], 0.0), 0.50)
    else:
        etr = fallback_tax_rate
    interest = data.get('interest_expense_history') or []
    total_debt = data.get('total_debt')
    if interest and total_debt and total_debt > 0:
        cod = min(max(interest[-1] / total_debt, 0.0), 0.20)
    else:
        cod = fallback_cost_of_debt
    margins = ebit_history[revenue_history > 0] / revenue_history[revenue_history > 0]
    return {
        'current_sales_to_capital_ratio': round(float(current_sales_to_capital_ratio), 3),
        'current_effective_tax_rate': round(float(etr), 4),
        'current_pretax_cost_of_debt': round(float(cod), 4),
        'terminal_operating_margin': round(float(np.mean(margins)), 4),
    }
