from __future__ import annotations

import json
import os
import re
from collections import Counter

from equity_research_ai.models import BusinessAnalysis


def clean_risk_sentence(sentence: str) -> str:
    """
    Converts mixed positive/negative MD&A sentences into cleaner risk bullets.

    This is intentionally a top-level function, not a BusinessAnalystAgent method,
    to avoid class attribute errors.
    """
    sentence = str(sentence).strip()
    lower = sentence.lower()

    if "partially offset by" in lower:
        after = sentence.split("partially offset by", 1)[1].strip()
        after = after.rstrip(".")
        return f"Risk factor: {after}."

    if "could" in lower and "impact" in lower:
        return sentence

    if (
        "foreign currencies" in lower
        or "foreign exchange" in lower
        or "currency fluctuations" in lower
    ):
        return "Foreign currency movements may pressure reported revenue or margins."

    if "inflation" in lower:
        return "Inflation may pressure costs, demand, or margins."

    if "interest rates" in lower:
        return "Changes in interest rates may affect demand, discount rates, or financial conditions."

    if "macroeconomic" in lower:
        return "Macroeconomic conditions may affect demand, operating results, and financial condition."

    if "competition" in lower:
        return "Competitive pressure may affect pricing, market share, or profitability."

    if "regulation" in lower or "regulatory" in lower:
        return "Regulatory developments may affect operations, costs, or business flexibility."

    return sentence


class BusinessAnalystAgent:
    """
    BusinessAnalystAgent v2.

    Retrieves SEC MD&A evidence, extracts business drivers, risks, margin signals,
    and capital intensity signals, then returns a BusinessAnalysis object.
    """

    def __init__(
        self,
        retriever=None,
        use_llm: bool = False,
        model: str = "gpt-4o-mini",
    ):
        self.retriever = retriever
        self.use_llm = use_llm
        self.model = model

    def analyze(self, ticker: str) -> BusinessAnalysis:
        questions = {
            "growth": (
                "revenue growth net sales increased demand growth drivers "
                "segment growth services product sales volume pricing"
            ),
            "margin": (
                "gross margin operating margin profitability margin increased "
                "margin decreased cost leverage pricing mix efficiency"
            ),
            "risks": (
                "risk factors competition regulation foreign exchange inflation "
                "supply chain demand weakness macroeconomic uncertainty"
            ),
            "capital": (
                "capital expenditures research development investment working capital "
                "capital allocation repurchases dividends reinvestment"
            ),
            "headwinds": (
                "net sales decreased decline lower weakness foreign exchange "
                "macroeconomic headwinds inflation supply constraints"
            ),
        }

        evidence = []

        if self.retriever:
            for query in questions.values():
                evidence.extend(self.retriever.query(query, ticker=ticker, top_k=5))

        evidence = self._dedupe_evidence(evidence)

        if self.use_llm and os.getenv("OPENAI_API_KEY") and evidence:
            return self._llm_analyze(ticker, evidence)

        return self._heuristic_analyze(ticker, evidence)

    def _dedupe_evidence(self, evidence):
        seen = set()
        unique = []

        for doc in evidence:
            key = (
                getattr(doc, "doc_id", None),
                getattr(doc, "paragraph_id", None),
                getattr(doc, "text", "")[:120],
            )

            if key in seen:
                continue

            seen.add(key)
            unique.append(doc)

        return unique

    def _sentences(self, text: str):
        text = re.sub(r"\s+", " ", text or "").strip()

        if not text:
            return []

        # Remove common SEC page/table artifacts created during HTML extraction.
        text = re.sub(r"\|\s*\d{4}\s+Form\s+10-K\s*\|\s*\d+", " ", text)
        text = re.sub(r"\|\s*Form\s+10-K\s*\|", " ", text)
        text = re.sub(r"\|\s*\d+\s*\|", " ", text)
        text = text.replace("|", " ")

        # Fix duplicated section labels sometimes created from filing headings.
        text = re.sub(r"\bServices Services\b", "Services", text)
        text = re.sub(r"\bProducts Products\b", "Products", text)
        text = re.sub(r"\bGross Margin Products and Services\b", "Gross margin", text)

        parts = re.split(r"(?<=[.!?])\s+", text)

        cleaned = []

        for part in parts:
            part = part.strip()

            if len(part) <= 40:
                continue

            lower = part.lower()

            # Skip table-like fragments that are not useful as narrative evidence.
            if "dollars in millions" in lower and "were as follows" in lower:
                continue

            if "gross margin and gross margin percentage" in lower:
                continue

            if lower.startswith("form 10-k"):
                continue

            cleaned.append(part)

        return cleaned

    def _find_relevant_sentences(self, evidence, keywords, limit=5):
        results = []

        for doc in evidence:
            for sentence in self._sentences(doc.text):
                lower = sentence.lower()

                if any(keyword in lower for keyword in keywords):
                    results.append((sentence, doc))
                    break

        return results[:limit]

    def _format_source(self, doc):
        company = getattr(doc, "company", None) or "Company"
        year = getattr(doc, "fiscal_year", None) or "Unknown year"
        doc_id = getattr(doc, "doc_id", None) or "Unknown filing"
        paragraph_id = getattr(doc, "paragraph_id", None) or "Unknown paragraph"

        return f"{company}, {year}, {doc_id}, paragraph {paragraph_id}"

    def _to_bullet(self, sentence, doc):
        sentence = str(sentence).strip()

        if len(sentence) > 260:
            sentence = sentence[:260].rsplit(" ", 1)[0] + "..."

        return f"{sentence} Source: {self._format_source(doc)}"

    def _heuristic_analyze(self, ticker, evidence):
        blob = " ".join(d.text.lower() for d in evidence)

        growth_keywords = [
            "net sales increased",
            "revenue increased",
            "growth",
            "higher net sales",
            "demand",
            "services",
            "cloud",
            "subscription",
            "pricing",
            "volume",
        ]

        margin_keywords = [
            "gross margin increased",
            "operating margin",
            "margin increased",
            "higher margin",
            "leverage",
            "efficiency",
            "productivity",
            "mix",
            "cost savings",
        ]

        risk_keywords = [
            "risk",
            "competition",
            "regulation",
            "foreign exchange",
            "inflation",
            "supply",
            "macroeconomic",
            "uncertainty",
            "litigation",
            "china",
            "decrease",
            "decline",
            "weakness",
            "lower",
        ]

        capital_keywords = [
            "research and development",
            "r&d",
            "capital expenditures",
            "investment",
            "infrastructure",
            "data center",
            "working capital",
            "repurchases",
            "dividends",
        ]

        growth_sentences = self._find_relevant_sentences(
            evidence=evidence,
            keywords=growth_keywords,
            limit=5,
        )

        margin_sentences = self._find_relevant_sentences(
            evidence=evidence,
            keywords=margin_keywords,
            limit=4,
        )

        risk_sentences = self._find_relevant_sentences(
            evidence=evidence,
            keywords=risk_keywords,
            limit=5,
        )

        capital_sentences = self._find_relevant_sentences(
            evidence=evidence,
            keywords=capital_keywords,
            limit=4,
        )

        drivers = []

        for sentence, doc in growth_sentences[:3]:
            drivers.append(self._to_bullet(sentence, doc))

        for sentence, doc in margin_sentences[:2]:
            drivers.append(self._to_bullet(sentence, doc))

        risks = []

        for sentence, doc in risk_sentences[:4]:
            cleaned_risk = clean_risk_sentence(sentence)
            risks.append(self._to_bullet(cleaned_risk, doc))

        if not drivers:
            drivers = [
                "No strong company-specific growth or margin driver was found in retrieved SEC evidence."
            ]

        if not risks:
            risks = [
                "No major company-specific risk was found in retrieved SEC evidence."
            ]

        signal_counts = Counter()

        positive_growth_terms = [
            "increased",
            "growth",
            "higher",
            "strong",
            "demand",
            "services",
            "cloud",
            "subscription",
        ]

        negative_growth_terms = [
            "decreased",
            "decline",
            "lower",
            "weakness",
            "headwind",
            "unfavorable",
        ]

        margin_positive_terms = [
            "gross margin increased",
            "operating margin increased",
            "leverage",
            "efficiency",
            "productivity",
            "favorable mix",
        ]

        margin_negative_terms = [
            "gross margin decreased",
            "operating margin decreased",
            "costs increased",
            "inflation",
            "unfavorable mix",
            "foreign exchange",
        ]

        capital_intensive_terms = [
            "capital expenditures",
            "infrastructure",
            "data center",
            "manufacturing",
            "capacity",
        ]

        asset_light_terms = [
            "services",
            "subscription",
            "software",
            "royalty",
            "license",
        ]

        for term in positive_growth_terms:
            if term in blob:
                signal_counts["growth_positive"] += 1

        for term in negative_growth_terms:
            if term in blob:
                signal_counts["growth_negative"] += 1

        for term in margin_positive_terms:
            if term in blob:
                signal_counts["margin_positive"] += 1

        for term in margin_negative_terms:
            if term in blob:
                signal_counts["margin_negative"] += 1

        for term in capital_intensive_terms:
            if term in blob:
                signal_counts["capital_intensive"] += 1

        for term in asset_light_terms:
            if term in blob:
                signal_counts["asset_light"] += 1

        if signal_counts["growth_positive"] >= signal_counts["growth_negative"] + 2:
            growth_outlook = "positive"
        elif signal_counts["growth_negative"] >= signal_counts["growth_positive"] + 2:
            growth_outlook = "negative"
        else:
            growth_outlook = "neutral"

        if signal_counts["margin_positive"] > signal_counts["margin_negative"]:
            margin_outlook = "improving"
        elif signal_counts["margin_negative"] > signal_counts["margin_positive"]:
            margin_outlook = "pressured"
        else:
            margin_outlook = "stable"

        if signal_counts["asset_light"] > signal_counts["capital_intensive"]:
            capital_efficiency = "improving"
        elif signal_counts["capital_intensive"] > signal_counts["asset_light"]:
            capital_efficiency = "capital_intensive"
        else:
            capital_efficiency = "stable"

        raw_summary_parts = [
            f"Growth outlook: {growth_outlook}",
            f"Margin outlook: {margin_outlook}",
            f"Capital efficiency: {capital_efficiency}",
        ]

        if capital_sentences:
            raw_summary_parts.append("Capital allocation evidence:")

            for sentence, doc in capital_sentences[:2]:
                raw_summary_parts.append("- " + self._to_bullet(sentence, doc))

        raw_summary = "\n".join(raw_summary_parts)

        return BusinessAnalysis(
            ticker=ticker,
            drivers=drivers,
            risks=risks,
            growth_outlook=growth_outlook,
            margin_outlook=margin_outlook,
            capital_efficiency=capital_efficiency,
            evidence=evidence[:12],
            raw_summary=raw_summary,
        )

    def _llm_analyze(self, ticker, evidence):
        from openai import OpenAI

        client = OpenAI()

        context = "\n\n".join(
            [
                (
                    (
                        f"[{i + 1}] "
                        f"Company: {getattr(d, 'company', None)} | "
                        f"Year: {getattr(d, 'fiscal_year', None)} | "
                        f"Doc: {getattr(d, 'doc_id', None)} | "
                        f"Paragraph: {getattr(d, 'paragraph_id', None)}\n"
                        f"{d.text}"
                    )
                )
                for i, d in enumerate(evidence[:15])
            ]
        )

        prompt = f"""
You are an equity research analyst analyzing SEC MD&A evidence for {ticker}.

Return JSON only with these exact keys:
- drivers: list of 3 to 6 specific business drivers, each with short citation text.
- risks: list of 3 to 6 specific risks or headwinds, each with short citation text.
- growth_outlook: one of ["positive", "neutral", "negative"].
- margin_outlook: one of ["improving", "stable", "pressured"].
- capital_efficiency: one of ["improving", "stable", "capital_intensive"].
- raw_summary: concise explanation connecting evidence to valuation assumptions.

Rules:
- Be specific.
- Do not use generic wording unless evidence is weak.
- Cite paragraph numbers in each driver and risk when possible.
- Do not invent facts beyond the evidence.

Evidence:
{context}
"""

        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        content = resp.choices[0].message.content.strip()

        if content.startswith("```"):
            content = content.strip("`")
            content = content.replace("json", "", 1).strip()

        data = json.loads(content)

        return BusinessAnalysis(
            ticker=ticker,
            evidence=evidence[:15],
            drivers=data.get("drivers", []),
            risks=data.get("risks", []),
            growth_outlook=data.get("growth_outlook", "neutral"),
            margin_outlook=data.get("margin_outlook", "stable"),
            capital_efficiency=data.get("capital_efficiency", "stable"),
            raw_summary=data.get("raw_summary", ""),
        )