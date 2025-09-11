import argparse
from pathlib import Path
from typing import Optional
import requests, feedparser, dateparser, pandas as pd
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "macro-multimodal-ai/1.0 (+https://example.local)"}
FEEDS = {
    "fed_press_monetary": "https://www.federalreserve.gov/feeds/press_monetary.xml",
    "fed_speeches": "https://www.federalreserve.gov/feeds/speeches.xml",
    "bls_cpi": "https://www.bls.gov/feed/cpi.rss",
    "bls_empsit": "https://www.bls.gov/feed/empsit.rss",
    "bea_news": "https://apps.bea.gov/rss/rss.xml",  # filtered to Personal Income & Outlays (PCE)
}

def _fetch(url: str, timeout: int = 20) -> Optional[str]:
    try:
        r = requests.get(url, timeout=timeout, headers=HEADERS)
        r.raise_for_status()
        return r.text
    except Exception:
        return None

def _extract_first_paragraph(html: str) -> Optional[str]:
    try:
        soup = BeautifulSoup(html, "lxml")
        for selector in ["article", "main", ".article", ".content", "#content"]:
            node = soup.select_one(selector)
            if node:
                p = node.find("p")
                if p and p.get_text(strip=True):
                    return p.get_text(" ", strip=True)
        p = soup.find("p")
        if p and p.get_text(strip=True):
            return p.get_text(" ", strip=True)
    except Exception:
        pass
    return None

def _parse_date(s: str) -> Optional[pd.Timestamp]:
    # Parse RSS date, convert to tz-naive UTC date (normalize to midnight)
    if not s:
        return None
    dt = dateparser.parse(s)
    if not dt:
        return None
    ts = pd.to_datetime(dt, utc=True).tz_localize(None)  # drop tz
    return ts.normalize()

def _dedupe(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["date","headline","title","link"])
    return (
        df.dropna(subset=["date","headline"])
          .drop_duplicates(subset=["date","headline"])
          .sort_values("date")
          .reset_index(drop=True)
    )

def fetch_feed(name: str, url: str, since: Optional[pd.Timestamp], fetch_pages: bool) -> pd.DataFrame:
    # Fetch XML via requests (custom headers), then parse
    xml = _fetch(url)
    if not xml:
        print(f"[warn] could not fetch feed {url}")
        return pd.DataFrame(columns=["date","headline","title","link"])

    parsed = feedparser.parse(xml)
    rows = []
    for e in parsed.entries:
        dt = _parse_date(e.get("published") or e.get("updated") or e.get("dc_date") or e.get("date"))
        if since and dt is not None and dt < since:
            continue
        title = (e.get("title") or "").strip()
        link = e.get("link")
        headline = title
        if fetch_pages and link:
            html = _fetch(link)
            txt = _extract_first_paragraph(html) if html else None
            if txt and len(txt) > 40:
                headline = txt
        rows.append({"date": dt, "headline": headline, "title": title, "link": link})

    out = pd.DataFrame(rows, columns=["date","headline","title","link"])
    if out.empty:
        return out  # keep columns present

    # BEA: keep Personal Income & Outlays (PCE) only
    if name == "bea_news":
        m = (
            out["title"].str.contains("Personal Income and Outlays|PCE", case=False, na=False) |
            out["headline"].str.contains("Personal Income and Outlays|PCE", case=False, na=False)
        )
        out = out[m]

    return _dedupe(out)

def save_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if df is None or df.empty:
        pd.DataFrame(columns=["date","headline"]).to_csv(path, index=False)
        return
    # ensure we have the two required columns; fallback to title if needed
    cols = set(df.columns)
    if {"date","headline"}.issubset(cols):
        out = df[["date","headline"]]
    elif {"date","title"}.issubset(cols):
        out = df[["date","title"]].rename(columns={"title":"headline"})
    else:
        out = pd.DataFrame(columns=["date","headline"])
    out.to_csv(path, index=False)

def main():
    ap = argparse.ArgumentParser("Fetch official macro text (RSS) and write CSVs (date,headline).")
    ap.add_argument("--since", default="2018-01-01", help="ISO date (e.g., 2018-01-01)")
    ap.add_argument("--outdir", default="mytexts", help="Output directory")
    ap.add_argument("--no-fetch-pages", action="store_true", help="Skip fetching article pages (use RSS titles)")
    args = ap.parse_args()

    since = pd.to_datetime(args.since).normalize() if args.since else None
    outdir = Path(args.outdir)
    outs = {
        "fed_press_monetary": outdir / "fed_statements_minutes.csv",
        "fed_speeches": outdir / "fed_speeches.csv",
        "bls_cpi": outdir / "bls_cpi.csv",
        "bls_empsit": outdir / "bls_nfp.csv",
        "bea_news": outdir / "bea_pce.csv",
    }

    counts = {}
    for key, url in FEEDS.items():
        print(f"[fetch] {key} -> {url}")
        df = fetch_feed(key, url, since, fetch_pages=not args.no_fetch_pages)
        save_csv(df, outs[key])
        counts[key] = 0 if df is None else len(df)

    print("\nWrote:")
    for k, p in outs.items():
        print(f"  {k:18s} -> {p} ({counts.get(k,0)} rows)")

    # Combined preview
    combined = []
    for p in outs.values():
        try:
            combined.append(pd.read_csv(p, parse_dates=["date"]))
        except Exception:
            pass
    if combined:
        merged = (pd.concat(combined, ignore_index=True)
                    .drop_duplicates()
                    .sort_values("date"))
        merged.to_csv(outdir / "official_combined_preview.csv", index=False)
        print(f"\nPreview -> {outdir/'official_combined_preview.csv'} (rows={len(merged)})")

if __name__ == "__main__":
    main()
