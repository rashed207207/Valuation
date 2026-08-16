from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup


SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"
SEC_ARCHIVES_URL = (
    "https://www.sec.gov/Archives/edgar/data/"
    "{cik_int}/{accession_no_dashes}/{primary_document}"
)


def normalize_cik(cik) -> str:
    return str(cik).strip().lstrip("0")


def cik10(cik) -> str:
    return normalize_cik(cik).zfill(10)


def clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "table"]):
        tag.decompose()

    return clean_text(soup.get_text(" "))


def paragraph_split(text: str, min_chars: int = 80) -> list[str]:
    raw_parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)

    paragraphs = []
    buffer = ""

    for part in raw_parts:
        part = part.strip()

        if not part:
            continue

        if len(buffer) < min_chars:
            buffer = f"{buffer} {part}".strip()
        else:
            paragraphs.append(buffer)
            buffer = part

    if buffer:
        paragraphs.append(buffer)

    return [p for p in paragraphs if len(p) >= min_chars]


def extract_mda(text: str) -> str:
    text = clean_text(text)
    lower_text = text.lower()

    start_patterns = [
        r"item\s+7[\.\s\-–—:]+management[’'`s\s]+discussion\s+and\s+analysis",
        r"item\s+7[\.\s\-–—:]+management\s+discussion\s+and\s+analysis",
        r"item\s+7[\.\s\-–—:]+md&a",
        r"item\s+7[\.\s\-–—:]+",
    ]

    end_patterns = [
        r"item\s+7a[\.\s\-–—:]+",
        r"item\s+8[\.\s\-–—:]+",
    ]

    start = None

    for pattern in start_patterns:
        match = re.search(pattern, lower_text, flags=re.IGNORECASE)

        if match:
            start = match.start()
            break

    if start is None:
        raise ValueError("Could not locate Item 7 / MD&A section.")

    end = None
    search_region = lower_text[start + 50 :]

    for pattern in end_patterns:
        match = re.search(pattern, search_region, flags=re.IGNORECASE)

        if match:
            candidate_end = start + 50 + match.start()

            if candidate_end > start:
                end = candidate_end
                break

    if end is None:
        end = min(len(text), start + 200000)

    return clean_text(text[start:end])


def read_existing_company_map(filings_dir: Path) -> dict:
    companies = {}

    for jsonl in filings_dir.rglob("*.jsonl"):
        try:
            with jsonl.open("r", encoding="utf-8") as file:
                for line in file:
                    if not line.strip():
                        continue

                    row = json.loads(line)
                    cik = normalize_cik(row.get("cik") or jsonl.parent.name)

                    if cik not in companies:
                        companies[cik] = {
                            "cik": cik,
                            "ticker": row.get("ticker"),
                            "company": row.get("company"),
                            "sector": row.get("sector"),
                            "industry": row.get("industry"),
                        }

                    break

        except Exception:
            continue

    return companies


def sec_get_json(url: str, user_agent: str, pause: float = 0.25) -> dict:
    headers = {
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate",
    }

    time.sleep(pause)

    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    return response.json()


def sec_get_text(url: str, user_agent: str, pause: float = 0.25) -> str:
    headers = {
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate",
    }

    time.sleep(pause)

    response = requests.get(url, headers=headers, timeout=60)
    response.raise_for_status()

    return response.text


def find_10k_filings_for_years(
    cik: str,
    years: list[int],
    user_agent: str,
) -> list[dict]:
    url = SEC_SUBMISSIONS_URL.format(cik10=cik10(cik))
    data = sec_get_json(url, user_agent=user_agent)

    recent = data.get("filings", {}).get("recent", {})

    forms = recent.get("form", [])
    accession_numbers = recent.get("accessionNumber", [])
    filing_dates = recent.get("filingDate", [])
    report_dates = recent.get("reportDate", [])
    primary_documents = recent.get("primaryDocument", [])

    target_years = set(int(year) for year in years)
    filings = []

    for form, accession, filing_date, report_date, primary_document in zip(
        forms,
        accession_numbers,
        filing_dates,
        report_dates,
        primary_documents,
    ):
        if form != "10-K":
            continue

        if not report_date:
            continue

        try:
            fiscal_year = int(str(report_date)[:4])
        except ValueError:
            continue

        if fiscal_year not in target_years:
            continue

        filings.append(
            {
                "cik": normalize_cik(cik),
                "fiscal_year": fiscal_year,
                "fiscal_year_end": report_date,
                "filing_date": filing_date,
                "accession_number": accession,
                "primary_document": primary_document,
            }
        )

    return filings


def download_10k_document(filing: dict, user_agent: str) -> str:
    cik_int = normalize_cik(filing["cik"])
    accession_no_dashes = filing["accession_number"].replace("-", "")
    primary_document = filing["primary_document"]

    url = SEC_ARCHIVES_URL.format(
        cik_int=cik_int,
        accession_no_dashes=accession_no_dashes,
        primary_document=primary_document,
    )

    return sec_get_text(url, user_agent=user_agent)


def write_mda_jsonl(
    filings_dir: Path,
    company_meta: dict,
    filing: dict,
    mda_text: str,
    overwrite: bool = False,
) -> Path:
    cik = normalize_cik(filing["cik"])
    ticker = company_meta.get("ticker")
    company = company_meta.get("company")
    sector = company_meta.get("sector")
    industry = company_meta.get("industry")

    out_dir = filings_dir / cik
    out_dir.mkdir(parents=True, exist_ok=True)

    filing_date_compact = filing["filing_date"].replace("-", "")
    out_path = out_dir / f"{filing_date_compact}_10-K_{cik}.jsonl"

    if out_path.exists() and not overwrite:
        print(f"SKIP existing: {out_path}")
        return out_path

    paragraphs = paragraph_split(mda_text)
    doc_id = f"{cik}_{filing['fiscal_year_end']}"

    with out_path.open("w", encoding="utf-8") as file:
        for paragraph_id, paragraph in enumerate(paragraphs, start=1):
            row = {
                "doc_id": doc_id,
                "cik": cik,
                "ticker": ticker,
                "company": company,
                "sector": sector,
                "industry": industry,
                "fiscal_year": filing["fiscal_year"],
                "fiscal_year_end": filing["fiscal_year_end"],
                "section": "MD&A",
                "paragraph_id": paragraph_id,
                "text": paragraph,
                "source": "SEC EDGAR",
                "accession_number": filing["accession_number"],
                "filing_date": filing["filing_date"],
                "primary_document": filing["primary_document"],
            }

            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"WROTE {out_path} paragraphs={len(paragraphs)}")
    return out_path


def update_company(
    filings_dir: Path,
    company_meta: dict,
    years: list[int],
    user_agent: str,
    overwrite: bool = False,
):
    cik = company_meta["cik"]
    ticker = company_meta.get("ticker") or cik

    print(f"\n=== {ticker} CIK {cik} ===")

    sec_filings = find_10k_filings_for_years(
        cik=cik,
        years=years,
        user_agent=user_agent,
    )

    if not sec_filings:
        print(f"No 10-K filings found for target years {years}.")
        return

    for filing in sec_filings:
        print(
            f"Downloading {ticker} FY{filing['fiscal_year']} "
            f"filed {filing['filing_date']} accession {filing['accession_number']}"
        )

        try:
            html = download_10k_document(filing, user_agent=user_agent)
            text = html_to_text(html)
            mda_text = extract_mda(text)

            write_mda_jsonl(
                filings_dir=filings_dir,
                company_meta=company_meta,
                filing=filing,
                mda_text=mda_text,
                overwrite=overwrite,
            )

        except Exception as error:
            print(
                f"FAILED {ticker} FY{filing['fiscal_year']}: "
                f"{type(error).__name__}: {error}"
            )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--filings-dir",
        required=True,
        help="Path to sec-mdna-rag/data/filings",
    )

    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2024, 2025],
        help="Fiscal years to add. Default: 2024 2025",
    )

    parser.add_argument(
        "--tickers",
        nargs="*",
        default=None,
        help="Optional tickers to update, for example: AAPL NVDA AMZN",
    )

    parser.add_argument(
        "--user-agent",
        required=True,
        help='SEC User-Agent, for example: "Rashed Kamal rashed@example.com"',
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing generated JSONL files.",
    )

    args = parser.parse_args()

    filings_dir = Path(args.filings_dir)

    if not filings_dir.exists():
        raise FileNotFoundError(f"Filings directory not found: {filings_dir}")

    companies = read_existing_company_map(filings_dir)

    if not companies:
        raise RuntimeError("Could not read any companies from existing JSONL data.")

    selected = list(companies.values())

    if args.tickers:
        wanted = {ticker.upper() for ticker in args.tickers}

        selected = [
            meta
            for meta in selected
            if str(meta.get("ticker", "")).upper() in wanted
        ]

    print(f"Companies selected: {len(selected)}")
    print(f"Years selected: {args.years}")

    for meta in selected:
        update_company(
            filings_dir=filings_dir,
            company_meta=meta,
            years=args.years,
            user_agent=args.user_agent,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()