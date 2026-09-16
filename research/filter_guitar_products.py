"""
Step 3 of the guitar-domain feasibility plan.

Filters the Amazon Reviews 2023 "Musical Instruments" metadata dump down to
actual guitar instruments (not parts/accessories/strings/amps/effects), then
tries to map those products onto the 113 Maker/Model rows in GuitarModels.csv
at two granularities:
  - brand-level (Maker only)
  - model-level (Maker + Model, fuzzy title match)

Outputs counts at each stage plus sampled matched/unmatched titles for manual
spot-checking, and writes the surviving guitar-instrument product IDs
(parent_asin) to guitar_products.jsonl for the next step (interaction volume
check).
"""
import csv
import gzip
import json
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path

try:
    from rapidfuzz import fuzz
except ImportError:
    print("rapidfuzz not installed; run: pip install rapidfuzz", file=sys.stderr)
    raise

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
META_PATH = DATA_DIR / "meta_Musical_Instruments.jsonl.gz"
META_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Musical_Instruments.jsonl.gz"
GUITAR_MODELS_CSV = ROOT.parent / "GuitarModels.csv"
OUT_PRODUCTS = DATA_DIR / "guitar_products.jsonl"


def ensure_downloaded(path, url):
    """Download `url` to `path` if it isn't already there. No login/API key
    needed - it's a public file on mcauleylab.ucsd.edu."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"{path.name} not found locally, downloading from {url} ...")

    def _progress(block_num, block_size, total_size):
        done = block_num * block_size
        if total_size > 0:
            pct = min(100, done * 100 // total_size)
            print(f"\r  {done / 1e6:.0f} MB / {total_size / 1e6:.0f} MB ({pct}%)", end="", flush=True)

    urllib.request.urlretrieve(url, path, reporthook=_progress)
    print()

# Categories that represent the guitar *instrument itself*, not parts,
# accessories, strings, amps, effects, bags/cases (per category label survey).
GUITAR_INSTRUMENT_CATEGORIES = {
    "Guitars",
    "Electric Guitars",
    "Acoustic Guitars",
    "Bass Guitars",
    "Acoustic-Electric Guitars",
    "Classical & Nylon-String Guitars",
    "Electro-Acoustic Bass Guitars",
    "Lap & Pedal Steel Guitars",
    "Pre-Owned Guitars",
}

MODEL_MATCH_THRESHOLD = 80  # rapidfuzz partial_ratio, 0-100
SAMPLE_SIZE = 25


def load_guitar_models():
    rows = []
    with open(GUITAR_MODELS_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({"maker": row["Maker"], "model": row["Model"]})
    return rows


def normalize(text):
    return re.sub(r"[^a-z0-9 ]", " ", (text or "").lower())


def is_guitar_instrument(categories):
    return any(cat in GUITAR_INSTRUMENT_CATEGORIES for cat in (categories or []))


def best_model_match(title_norm, maker_norm, guitar_models):
    best = None
    best_score = 0
    for gm in guitar_models:
        if gm["maker"].lower() not in title_norm and gm["maker"].lower() != maker_norm:
            continue
        model_norm = normalize(gm["model"])
        score = fuzz.partial_ratio(model_norm, title_norm)
        if score > best_score:
            best_score = score
            best = gm
    if best is not None and best_score >= MODEL_MATCH_THRESHOLD:
        return best, best_score
    return None, best_score


def main():
    ensure_downloaded(META_PATH, META_URL)
    guitar_models = load_guitar_models()
    known_makers = {gm["maker"].lower() for gm in guitar_models}

    stage_counts = Counter()
    matched_model_samples = []
    matched_brand_only_samples = []
    unmatched_guitar_samples = []

    surviving_products = []

    with gzip.open(META_PATH, "rt", encoding="utf-8") as f:
        for line in f:
            stage_counts["raw_musical_instruments"] += 1
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue

            categories = d.get("categories") or []
            if not is_guitar_instrument(categories):
                continue
            stage_counts["guitar_category"] += 1

            title = d.get("title") or ""
            store = d.get("store") or ""
            title_norm = normalize(title)
            maker_norm = normalize(store)

            maker_hit = None
            for m in known_makers:
                if m in title_norm or m == maker_norm:
                    maker_hit = m
                    break

            record = {
                "parent_asin": d.get("parent_asin"),
                "title": title,
                "store": store,
                "categories": categories,
                "average_rating": d.get("average_rating"),
                "rating_number": d.get("rating_number"),
            }

            if maker_hit:
                stage_counts["maker_matched"] += 1
                match, score = best_model_match(title_norm, maker_norm, guitar_models)
                if match:
                    stage_counts["model_matched"] += 1
                    record["matched_maker"] = match["maker"]
                    record["matched_model"] = match["model"]
                    record["match_score"] = score
                    if len(matched_model_samples) < SAMPLE_SIZE:
                        matched_model_samples.append((title, match["maker"], match["model"], score))
                else:
                    record["matched_maker"] = maker_hit
                    record["matched_model"] = None
                    if len(matched_brand_only_samples) < SAMPLE_SIZE:
                        matched_brand_only_samples.append((title, maker_hit))
            else:
                if len(unmatched_guitar_samples) < SAMPLE_SIZE:
                    unmatched_guitar_samples.append((title, store))

            surviving_products.append(record)

    stage_counts["total_guitar_instrument_products"] = len(surviving_products)

    with open(OUT_PRODUCTS, "w", encoding="utf-8") as f:
        for rec in surviving_products:
            f.write(json.dumps(rec) + "\n")

    print("=== Stage counts ===")
    for k in ["raw_musical_instruments", "guitar_category", "maker_matched",
              "model_matched", "total_guitar_instrument_products"]:
        print(f"{k}: {stage_counts[k]}")

    print(f"\n=== Sample: model-level matches ({len(matched_model_samples)}) ===")
    for title, maker, model, score in matched_model_samples:
        print(f"[{score:>3}] {maker} / {model}  <-  {title}")

    print(f"\n=== Sample: brand-matched but no model match ({len(matched_brand_only_samples)}) ===")
    for title, maker in matched_brand_only_samples:
        print(f"{maker}  <-  {title}")

    print(f"\n=== Sample: guitar-category but unknown brand ({len(unmatched_guitar_samples)}) ===")
    for title, store in unmatched_guitar_samples:
        print(f"[{store}]  {title}")

    print(f"\nWrote {len(surviving_products)} guitar-instrument product records to {OUT_PRODUCTS}")


if __name__ == "__main__":
    main()
