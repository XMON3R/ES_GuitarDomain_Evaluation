"""
Phase 1 of the implementation plan: turn the feasibility-study artifacts
(research/data/guitar_products.jsonl + Musical_Instruments reviews) into the
final compact items.csv / interactions.csv consumed by the EasyStudy
GuitarDataLoaderWrapper.

Steps:
  1. Re-scan meta_Musical_Instruments.jsonl.gz once to pull image URLs for the
     already-filtered guitar_products.jsonl (not captured in the original
     feasibility pass).
  2. Drop the ~0.36% miscategorized non-instrument titles (tumbler/mug/etc.).
  3. Assign a dense integer item_id per surviving product (sorted by
     parent_asin for reproducibility).
  4. Left-join GuitarModels.csv onto items matched at model-level, adding the
     historical enrichment columns.
  5. Write items.csv.
  6. Re-scan Musical_Instruments.jsonl.gz once to pull reviews for the
     surviving item set, map string user_id -> dense int user_id, write
     interactions.csv.

Run from repo root: python research/build_dataset.py
"""
import csv
import gzip
import json
from pathlib import Path


def _find_project_root(start):
    """Walk upward from `start` until a directory containing an 'EasyStudy'
    folder is found - robust to this script living at any depth/location,
    unlike a fixed ROOT.parent assumption."""
    p = start
    for _ in range(8):
        if (p / "EasyStudy").is_dir():
            return p
        if p.parent == p:
            break
        p = p.parent
    raise RuntimeError(f"Could not locate project root (no 'EasyStudy' directory found in any parent of {start})")


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
META_PATH = DATA_DIR / "meta_Musical_Instruments.jsonl.gz"
REVIEWS_PATH = DATA_DIR / "Musical_Instruments.jsonl.gz"
PRODUCTS_PATH = DATA_DIR / "guitar_products.jsonl"

PROJECT_ROOT = _find_project_root(ROOT)


def _find_guitar_models_csv():
    for candidate in [ROOT.parent / "GuitarModels.csv", PROJECT_ROOT / "GuitarModels.csv", PROJECT_ROOT / "old" / "GuitarModels.csv"]:
        if candidate.exists():
            return candidate
    raise RuntimeError("Could not locate GuitarModels.csv near the project root or under an 'old' archive folder")


GUITAR_MODELS_CSV = _find_guitar_models_csv()

OUT_DIR = PROJECT_ROOT / "EasyStudy" / "server" / "static" / "datasets" / "guitars"
ITEMS_OUT = OUT_DIR / "items.csv"
INTERACTIONS_OUT = OUT_DIR / "interactions.csv"

NOISE_WORDS = ["tumbler", "mug", "sticker", "decal", "keychain", "poster", "shirt", "case only"]


def load_guitar_models_by_key():
    by_key = {}
    with open(GUITAR_MODELS_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_key[(row["Maker"], row["Model"])] = row
    return by_key


def load_surviving_products():
    products = {}
    with open(PRODUCTS_PATH, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            title_lower = rec["title"].lower()
            if any(w in title_lower for w in NOISE_WORDS):
                continue
            products[rec["parent_asin"]] = rec
    return products


def best_image_url(images):
    if not images:
        return ""
    main = next((im for im in images if im.get("variant") == "MAIN"), images[0])
    return main.get("large") or main.get("hi_res") or main.get("thumb") or ""


def main():
    products = load_surviving_products()
    print(f"Surviving products after noise cleanup: {len(products)}")

    # Pull image URLs from metadata for the surviving asins
    n_scanned = 0
    with gzip.open(META_PATH, "rt", encoding="utf-8") as f:
        for line in f:
            n_scanned += 1
            d = json.loads(line)
            asin = d.get("parent_asin")
            if asin in products:
                products[asin]["image_url"] = best_image_url(d.get("images"))
    print(f"Scanned {n_scanned} metadata records for image URLs")

    guitar_models_by_key = load_guitar_models_by_key()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    asin_order = sorted(products.keys())
    asin_to_item_id = {asin: idx for idx, asin in enumerate(asin_order)}

    items_fields = [
        "item_id", "parent_asin", "title", "store", "categories", "image_url",
        "average_rating", "rating_number", "matched_maker", "matched_model",
        "introduced", "notable_user_1", "feature_1", "feature_2", "finish_1",
    ]
    with open(ITEMS_OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=items_fields)
        w.writeheader()
        for asin in asin_order:
            rec = products[asin]
            maker = rec.get("matched_maker")
            model = rec.get("matched_model")
            gm = guitar_models_by_key.get((maker, model)) if maker and model else None
            w.writerow({
                "item_id": asin_to_item_id[asin],
                "parent_asin": asin,
                "title": rec["title"],
                "store": rec.get("store") or "",
                "categories": "|".join(rec.get("categories") or []),
                "image_url": rec.get("image_url", ""),
                "average_rating": rec.get("average_rating") or "",
                "rating_number": rec.get("rating_number") or 0,
                "matched_maker": maker or "",
                "matched_model": model or "",
                "introduced": gm["Introduced"] if gm else "",
                "notable_user_1": gm["NotableUser1"] if gm else "",
                "feature_1": gm["Feature1"] if gm else "",
                "feature_2": gm["Feature2"] if gm else "",
                "finish_1": gm["Finish1"] if gm else "",
            })
    print(f"Wrote {len(asin_order)} items to {ITEMS_OUT}")

    # Now scan reviews once for the surviving item set
    user_to_id = {}
    n_reviews_written = 0
    n_reviews_scanned = 0
    with gzip.open(REVIEWS_PATH, "rt", encoding="utf-8") as fin, \
         open(INTERACTIONS_OUT, "w", newline="", encoding="utf-8") as fout:
        w = csv.writer(fout)
        w.writerow(["user_id", "item_id", "rating", "timestamp", "verified_purchase"])
        for line in fin:
            n_reviews_scanned += 1
            r = json.loads(line)
            asin = r.get("parent_asin")
            if asin not in asin_to_item_id:
                continue
            raw_user = r.get("user_id")
            if raw_user not in user_to_id:
                user_to_id[raw_user] = len(user_to_id)
            w.writerow([
                user_to_id[raw_user],
                asin_to_item_id[asin],
                r.get("rating"),
                r.get("timestamp"),
                1 if r.get("verified_purchase") else 0,
            ])
            n_reviews_written += 1

    print(f"Scanned {n_reviews_scanned} reviews, wrote {n_reviews_written} interactions "
          f"for {len(user_to_id)} distinct users to {INTERACTIONS_OUT}")


if __name__ == "__main__":
    main()
