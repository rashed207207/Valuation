from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from equity_research_ai.models import RetrievedDocument


class LocalSECRetriever:
    """
    Lightweight local SEC MD&A retriever.

    Improvements:
    - Filters by ticker when possible.
    - Uses TF-IDF similarity.
    - Adds a recency boost so 2024 and 2025 filings beat older filings when relevance is similar.
    """

    def __init__(self, data_dir: str | Path | None = None):
        self.data_dir = Path(data_dir) if data_dir else None
        self.df = pd.DataFrame()
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)
        self.matrix = None

        if self.data_dir and self.data_dir.exists():
            self.load()

    def load(self):
        rows = []

        for path in self.data_dir.rglob("*.jsonl"):
            with path.open("r", encoding="utf-8") as file:
                for line in file:
                    if not line.strip():
                        continue

                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if row.get("text"):
                        rows.append(row)

        self.df = pd.DataFrame(rows)

        if len(self.df):
            self.df["text"] = self.df["text"].fillna("").astype(str)

            if "ticker" in self.df.columns:
                self.df["ticker"] = self.df["ticker"].fillna("").astype(str).str.upper()

            if "fiscal_year" in self.df.columns:
                self.df["fiscal_year"] = pd.to_numeric(
                    self.df["fiscal_year"],
                    errors="coerce",
                )

            self.matrix = self.vectorizer.fit_transform(self.df["text"])

        return self

    def query(self, query: str, ticker: str | None = None, top_k: int = 8) -> list[RetrievedDocument]:
        if self.df.empty or self.matrix is None:
            return []

        df = self.df
        idx = df.index

        if ticker and "ticker" in df.columns:
            ticker_upper = ticker.upper()
            ticker_mask = df["ticker"] == ticker_upper
            ticker_idx = df[ticker_mask].index

            if len(ticker_idx) > 0:
                idx = ticker_idx

        qv = self.vectorizer.transform([query])
        cosine_scores = cosine_similarity(qv, self.matrix[idx]).ravel()

        scored_rows = []
        idx_list = list(idx)

        max_year = None
        if "fiscal_year" in df.columns:
            years = df.loc[idx_list, "fiscal_year"].dropna()
            if len(years):
                max_year = int(years.max())

        for local_i, row_index in enumerate(idx_list):
            row = df.loc[row_index]

            base_score = float(cosine_scores[local_i])
            fiscal_year = row.get("fiscal_year")

            recency_boost = 0.0

            if max_year is not None and pd.notna(fiscal_year):
                fiscal_year = int(fiscal_year)
                year_gap = max_year - fiscal_year

                if year_gap == 0:
                    recency_boost = 0.12
                elif year_gap == 1:
                    recency_boost = 0.08
                elif year_gap == 2:
                    recency_boost = 0.04
                else:
                    recency_boost = 0.00

            final_score = base_score + recency_boost

            scored_rows.append(
                {
                    "row_index": row_index,
                    "score": final_score,
                    "base_score": base_score,
                    "recency_boost": recency_boost,
                }
            )

        scored_rows = sorted(scored_rows, key=lambda x: x["score"], reverse=True)
        scored_rows = scored_rows[:top_k]

        docs = []

        for item in scored_rows:
            row = df.loc[item["row_index"]]

            docs.append(
                RetrievedDocument(
                    text=row.get("text", ""),
                    ticker=row.get("ticker"),
                    company=row.get("company"),
                    fiscal_year=row.get("fiscal_year"),
                    doc_id=row.get("doc_id"),
                    paragraph_id=row.get("paragraph_id"),
                    score=float(item["score"]),
                )
            )

        return docs
