"""
One-time test of GDELT's Web NGrams files as a source for the War-Peace Index.
Does NOT change data/war_peace_index.csv.

Step 1 checks which file addresses exist:
  v3: https://data.gdeltproject.org/gdeltv3/webngrams/YYYYMMDDHHMMSS.webngrams.json.gz
      JSON lines: date, ngram, lang, type, pos, pre, post, url   (Web NGrams 3.0, 2020-present)
  v5: https://data.gdeltproject.org/gdeltv5/weblegacy/ngrams/YYYYMMDDHHMM00.ngrams.txt.gz
Step 2 samples one minute per hour from one UTC day using the v3 files, applies the
index's keyword rules per article (URL), and reports size, speed, counts and languages.
"""
from __future__ import annotations

import gzip
import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

V3 = "https://data.gdeltproject.org/gdeltv3/webngrams/{s}.webngrams.json.gz"
V5 = "https://data.gdeltproject.org/gdeltv5/weblegacy/ngrams/{s}.ngrams.txt.gz"
WORKERS = 6
TIME_BUDGET_S = 15 * 60

UKRAINE = {"ukraine", "kyiv"}
RUSSIA = {"russia", "russian"}
PEACE_WORDS = {"ceasefire", "truce", "armistice", "negotiations"}
WAR_WORDS = {"invasion", "offensive", "shelling", "airstrike", "bombardment", "mobilization", "escalation"}
T0 = time.time()


def status(url: str) -> int:
    try:
        return requests.head(url, timeout=30, allow_redirects=True).status_code
    except requests.RequestException as exc:
        print(f"    {exc.__class__.__name__} for {url}")
        return -1


def check_addresses(day: pd.Timestamp) -> list[int]:
    print("Step 1: which file addresses exist?")
    tests = {
        "v5 blog example (2026-06-30 20:16)": V5.format(s="20260630201600"),
        "v5 test day 12:00": V5.format(s=f"{day:%Y%m%d}120000"),
        "v3 blog example (2020-01-01 00:01)": V3.format(s="20200101000100"),
        "v3 2026-10-01 12:00": V3.format(s="20261001120000"),
        "v3 test day 12:00": V3.format(s=f"{day:%Y%m%d}120000"),
        "v3 test day 12:01": V3.format(s=f"{day:%Y%m%d}120100"),
    }
    for name, url in tests.items():
        print(f"  {status(url):>4}  {name}")
    print("  Which minutes between 12:00 and 12:31 have a v3 file?")
    found = [m for m in range(0, 32)
             if status(V3.format(s=f"{day:%Y%m%d}12{m:02d}00")) == 200]
    print(f"    minutes with files: {found}")
    return found


def process_v3(stamp: str) -> dict:
    out = {"stamp": stamp, "status": None, "bytes": 0, "records": 0,
           "docs": {}, "langs": Counter(), "secs": 0.0}
    if time.time() - T0 > TIME_BUDGET_S:
        out["status"] = "skipped (time budget)"; return out
    t = time.time()
    try:
        r = requests.get(V3.format(s=stamp), timeout=120, stream=True)
    except requests.RequestException as exc:
        out["status"] = exc.__class__.__name__; return out
    out["status"] = r.status_code
    if r.status_code != 200:
        return out
    r.raw.decode_content = False
    docs: dict[str, set] = defaultdict(set)
    langs: dict[str, str] = {}
    with gzip.GzipFile(fileobj=r.raw) as gz:
        for line in gz:
            out["records"] += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            url = rec.get("url", "")
            w = str(rec.get("ngram", "")).lower().strip(".,;:!?\"'()")
            fl = docs[url]
            langs.setdefault(url, rec.get("lang", "?"))
            if w in UKRAINE: fl.add("u")
            elif w in RUSSIA: fl.add("r")
            elif w in PEACE_WORDS: fl.add("p")
            elif w in WAR_WORDS: fl.add("w")
            elif w == "peace":
                nxt = str(rec.get("post", "")).lower().lstrip()
                if nxt.startswith(("talks", "deal")): fl.add("p")
    out["bytes"] = r.raw.tell() if hasattr(r.raw, "tell") else 0
    out["docs"] = docs
    out["langs"] = Counter(langs.values())
    out["secs"] = time.time() - t
    return out


def main() -> int:
    day = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() - pd.Timedelta(days=1)
    if len(sys.argv) > 1 and sys.argv[1]:
        day = pd.Timestamp(sys.argv[1])
    found = check_addresses(day)
    minute = found[0] if found else 1
    per_hour = max(1, len([m for m in found if m < 30]) * 2)   # rough files-per-hour from the 30-min scan

    stamps = [(day + pd.Timedelta(hours=h, minutes=minute)).strftime("%Y%m%d%H%M%S") for h in range(24)]
    print(f"\nStep 2: sampling minute :{minute:02d} of each hour on {day.date()} (v3 files)")
    with ThreadPoolExecutor(WORKERS) as ex:
        res = list(ex.map(process_v3, stamps))
    print("  HTTP results:", dict(Counter(str(r["status"]) for r in res)))
    got = [r for r in res if r["status"] == 200]
    if not got:
        print("  No v3 files retrieved."); return 1

    recs = sum(r["records"] for r in got)
    mb = sum(r["bytes"] for r in got) / 1e6
    docs = ctx = peace = war = 0
    langs = Counter()
    for r in got:
        langs.update(r["langs"])
        for fl in r["docs"].values():
            docs += 1
            if "u" in fl and "r" in fl:
                ctx += 1; peace += "p" in fl; war += "w" in fl
    per_file_s = sum(r["secs"] for r in got) / len(got)
    print(f"  Files read: {len(got)}  |  {mb:,.1f} MB compressed  |  {recs:,} records  |  {docs:,} articles")
    print(f"  Avg per file: {mb/len(got):,.1f} MB, {per_file_s:,.1f} s")
    print(f"  Matches: Ukraine+Russia {ctx:,} | peace {peace:,} | war {war:,}")
    n = per_hour * 24
    print(f"  Files per day (from the minute scan): ~{n}")
    print(f"  Estimated full day: ~{mb/len(got)*n/1000:,.1f} GB, "
          f"~{per_file_s*n/WORKERS/60:,.0f} min with {WORKERS} parallel downloads, "
          f"~{docs/len(got)*n:,.0f} articles, ~{ctx/len(got)*n:,.0f} Ukraine+Russia articles")
    print("  Languages:", ", ".join(f"{k}:{v}" for k, v in langs.most_common(12)))
    print(f"  Total run time: {time.time()-T0:,.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
