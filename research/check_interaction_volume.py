"""
Step 4 of the guitar-domain feasibility plan.

Streams the full Musical_Instruments reviews file once, filtering to the
guitar-instrument parent_asins produced by filter_guitar_products.py, and
reports interaction counts / distinct users / distinct items / sparsity at
three granularities:
  - all guitar-category products (broadest, includes brands outside CSV)
  - brand-matched products (maker present in GuitarModels.csv)
  - model-matched products (fuzzy-matched to a specific GuitarModels.csv row)

Also writes out the filtered review rows for the model-matched and
maker-matched sets, for later use as the actual interaction log.
"""
import gzip
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
REVIEWS_PATH = DATA_DIR / "Musical_Instruments.jsonl.gz"
REVIEWS_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Musical_Instruments.jsonl.gz"
PRODUCTS_PATH = DATA_DIR / "guitar_products.jsonl"


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

OUT_MAKER_REVIEWS = DATA_DIR / "guitar_reviews_maker_matched.jsonl"
OUT_MODEL_REVIEWS = DATA_DIR / "guitar_reviews_model_matched.jsonl"


def load_product_sets():
    all_guitar = set()
    maker_matched = set()
    model_matched = set()
    with open(PRODUCTS_PATH, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            asin = rec["parent_asin"]
            all_guitar.add(asin)
            if rec.get("matched_maker"):
                maker_matched.add(asin)
            if rec.get("matched_model"):
                model_matched.add(asin)
    return all_guitar, maker_matched, model_matched


def stats_for(name, asin_set, reviews_by_asin_count, users_by_asin):
    total_interactions = sum(reviews_by_asin_count[a] for a in asin_set if a in reviews_by_asin_count)
    items_with_reviews = sum(1 for a in asin_set if a in reviews_by_asin_count)
    users = set()
    for a in asin_set:
        users |= users_by_asin.get(a, set())
    print(f"\n=== {name} ===")
    print(f"products in set: {len(asin_set)}")
    print(f"products with >=1 review: {items_with_reviews}")
    print(f"total interactions: {total_interactions}")
    print(f"distinct users: {len(users)}")
    if len(users) and items_with_reviews:
        density = total_interactions / (len(users) * items_with_reviews)
        print(f"sparsity (density): {density:.6f}")


def main():
    ensure_downloaded(REVIEWS_PATH, REVIEWS_URL)
    all_guitar, maker_matched, model_matched = load_product_sets()
    print(f"Loaded product sets: all_guitar={len(all_guitar)}, "
          f"maker_matched={len(maker_matched)}, model_matched={len(model_matched)}")

    reviews_by_asin_count = {}
    users_by_asin_all = {}

    maker_out = open(OUT_MAKER_REVIEWS, "w", encoding="utf-8")
    model_out = open(OUT_MODEL_REVIEWS, "w", encoding="utf-8")

    n_lines = 0
    with gzip.open(REVIEWS_PATH, "rt", encoding="utf-8") as f:
        for line in f:
            n_lines += 1
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            asin = r.get("parent_asin")
            if asin not in all_guitar:
                continue

            reviews_by_asin_count[asin] = reviews_by_asin_count.get(asin, 0) + 1
            users_by_asin_all.setdefault(asin, set()).add(r.get("user_id"))

            if asin in maker_matched:
                maker_out.write(line if line.endswith("\n") else line + "\n")
            if asin in model_matched:
                model_out.write(line if line.endswith("\n") else line + "\n")

    maker_out.close()
    model_out.close()

    print(f"\nScanned {n_lines} total Musical Instruments reviews.")

    stats_for("All guitar-category products", all_guitar, reviews_by_asin_count, users_by_asin_all)
    stats_for("Maker-matched (brand-level, 6 makers)", maker_matched, reviews_by_asin_count, users_by_asin_all)
    stats_for("Model-matched (113-row catalog)", model_matched, reviews_by_asin_count, users_by_asin_all)

    print(f"\nWrote filtered reviews to:\n  {OUT_MAKER_REVIEWS}\n  {OUT_MODEL_REVIEWS}")


if __name__ == "__main__":
    main()
