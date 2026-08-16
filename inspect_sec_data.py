import json
import pathlib

p = pathlib.Path("../sec-mdna-rag/data/filings")

files = list(p.rglob("*.jsonl"))
print("JSONL files:", len(files))

if not files:
    print("No JSONL files found. Check the path.")
    raise SystemExit

print("\nFirst files:")
for f in files[:10]:
    print(" ", f)

companies = set()
tickers = set()
ciks = set()

print("\nFirst row example:")
with open(files[0], encoding="utf-8") as fh:
    first_line = fh.readline()
    row = json.loads(first_line)
    print(row)
    print("\nKeys:", list(row.keys()))

for f in files:
    with open(f, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue

            row = json.loads(line)

            if row.get("company"):
                companies.add(row["company"])

            if row.get("ticker"):
                tickers.add(row["ticker"])

            if row.get("cik"):
                ciks.add(str(row["cik"]))

            break

print("\nTickers:")
print(sorted(tickers))

print("\nCompanies:")
for c in sorted(companies):
    print(" ", c)

print("\nCIKs:")
for c in sorted(ciks):
    print(" ", c)