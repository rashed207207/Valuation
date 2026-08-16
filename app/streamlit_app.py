import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from equity_research_ai.pipeline import run_analysis

st.set_page_config(page_title='Equity Research AI', layout='wide')
st.title('Equity Research AI')
st.caption('SEC MD&A RAG + DCF + Monte Carlo + implied expectations')

ticker = st.text_input('Ticker', 'AAPL')
sec_data = st.text_input('SEC JSONL data directory', './data/sec/filings')
use_llm = st.checkbox('Use LLM for business analysis if OPENAI_API_KEY is set', False)
sample_size = st.slider('Monte Carlo scenarios', 200, 5000, 1000, step=100)

if st.button('Run analysis'):
    with st.spinner('Running analysis...'):
        analysis = run_analysis(ticker, sec_data if Path(sec_data).exists() else None, use_llm=use_llm, sample_size=sample_size)
    st.markdown(analysis.final_report)
    st.download_button('Download report.md', analysis.final_report, file_name=f'{ticker.upper()}_research_report.md')
