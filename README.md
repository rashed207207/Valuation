# Equity Research AI

An MVP platform that combines:

1. **SEC MD&A retrieval** from local JSONL filings.
2. **Business analysis agent** that summarizes drivers, risks, margins, and capital intensity.
3. **Narrative-to-assumptions engine** that converts qualitative signals into DCF inputs.
4. **Multi-stage DCF valuation engine** based on your original model.
5. **Monte Carlo valuation** with Gaussian copula correlations.
6. **Market-implied expectations** using numerical root solving.
7. **Report writer** that produces an equity research memo.

> This is research tooling, not financial advice. Always validate filings, financial data, assumptions, and model outputs manually.

## Quick Start

```bash
cd equity-research-ai
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
cp .env.example .env
```

### Run an analysis

```bash
python -m equity_research_ai.pipeline AAPL --sec-data ./data/sec/filings
```

If you do not have SEC filing data locally, the system still runs with heuristic assumptions from Yahoo Finance, but the business-analysis step will be weaker.

### Use data from `sec-mdna-rag`

Clone the SEC MD&A project placed next to this folder or anywhere locally:

```bash
git clone https://github.com/AndyYTHsiao/sec-mdna-rag.git
```

Then point this project to the filings directory:

```bash
python -m equity_research_ai.pipeline AAPL --sec-data ../sec-mdna-rag/data/filings
```

Expected JSONL fields are flexible, but the retriever works best with:

```json
{
  "ticker": "AAPL",
  "company": "Apple Inc.",
  "fiscal_year": 2023,
  "doc_id": "...",
  "paragraph_id": "...",
  "text": "MD&A paragraph text..."
}
```

### Streamlit app

```bash
streamlit run app/streamlit_app.py
```

## Project layout

```text
equity_research_ai/
  agents/
  data/
  rag/
  valuation/
  models.py
  pipeline.py
app/
  streamlit_app.py
tests/
```

## What is already implemented

- Robust DCF functions from your notebook, refactored into importable modules.
- Company data fetcher using `yfinance`, with a Starbucks offline fallback.
- Local SEC MD&A JSONL retriever using TF-IDF cosine similarity.
- Business analyst agent with optional LLM enrichment.
- Assumption generator with deterministic fallback rules.
- Monte Carlo valuation and tornado sensitivity.
- Market-implied growth and margin solver.
- Markdown report generation.

## Next improvements

- Replace TF-IDF retriever with the hybrid FAISS + BM25 + RRF retriever from `sec-mdna-rag`.
- Add source-level citations to each assumption.
- Add 10-Q and earnings call transcript ingestion.
- Add tests for more edge cases and financial statement data quirks.
