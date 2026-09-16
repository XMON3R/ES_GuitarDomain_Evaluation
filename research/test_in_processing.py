"""
Demonstration/validation of InProcessingSparseVAECF: trains at the same
scale that already produced legible neurons in guitar_interpretability.ipynb
(hidden=128, latent=16, epochs=20, on the ~137K-interaction guitar slice),
finds the best single neuron and best neuron-combination (via the same
KMeans clustering approach), then measures - averaged over many synthetic
profiles, not just one - whether boosting/suppressing those neurons before
decoding shifts the target category's representation in top-k
recommendations. This is the literal in-processing mechanism check: does
intervening on the latent code (not re-ranking output) actually work?
"""
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np


def _find_project_root(start):
    """Walk upward from `start` until a directory containing an 'EasyStudy'
    folder is found - robust to this script living at any depth/location,
    unlike a fixed ../.. relative path (which also depends on CWD, not
    just this file's own location)."""
    p = start
    for _ in range(8):
        if (p / "EasyStudy").is_dir():
            return p
        if p.parent == p:
            break
        p = p.parent
    raise RuntimeError(f"Could not locate project root (no 'EasyStudy' directory found in any parent of {start})")


SERVER_DIR = _find_project_root(Path(__file__).resolve().parent) / "EasyStudy" / "server"
sys.path.insert(0, str(SERVER_DIR))

from plugins.fastcompare.algo.wrappers.guitar_loader import GuitarDataLoaderWrapper
from plugins.fastcompare.algo.in_processing_sparse_vae_cf import InProcessingSparseVAECF
from plugins.utils.popularity_sampling import PopularitySamplingElicitation

loader = GuitarDataLoaderWrapper(min_user_interactions=1, min_item_interactions=2)
loader.load_data()

HIDDEN_DIM, LATENT_DIM, EPOCHS, BATCH_SIZE = 128, 16, 20, 1024

t0 = time.perf_counter()
algo = InProcessingSparseVAECF(
    loader, positive_threshold=3.0, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM,
    epochs=EPOCHS, learning_rate=1e-3, beta=0.2, sparsity_weight=0.03, alpha=0.3,
    batch_size=BATCH_SIZE,
)
algo.fit()
print(f"Fit took {time.perf_counter() - t0:.1f}s")

M = algo._item_neuron_matrix
n_items, n_latent = M.shape
print(f"item_neuron_matrix shape: {M.shape}")

categories_by_item = {i: [c for c in loader.get_item_index_categories(i) if c != "Guitars"] for i in range(n_items)}
overall = Counter()
for i in range(n_items):
    overall.update(categories_by_item[i])
total = sum(overall.values())

TOP_N, MIN_COUNT = 25, 3


def most_enriched(counts):
    best_key, best_ratio, best_n = None, 0.0, 0
    for key, n in counts.items():
        if n < MIN_COUNT or not key:
            continue
        base_rate = overall.get(key, 1) / total
        ratio = (n / TOP_N) / base_rate if base_rate > 0 else 0
        if ratio > best_ratio:
            best_key, best_ratio, best_n = key, ratio, n
    return best_key, best_ratio, best_n


# Find the single best neuron
best_single = (None, 0.0, None)  # neuron, ratio, category
for neuron in range(n_latent):
    top = np.argsort(-np.abs(M[:, neuron]))[:TOP_N]
    cnt = Counter()
    for i in top:
        cnt.update(categories_by_item[int(i)])
    cat, ratio, n = most_enriched(cnt)
    if ratio > best_single[1]:
        best_single = (neuron, ratio, cat)

print(f"\nBest single neuron: {best_single[0]}, category={best_single[2]!r}, enrichment={best_single[1]:.1f}x")

# Find the best combination via KMeans clustering (same method as the interpretability notebook)
from sklearn.cluster import KMeans

N_CLUSTERS = 2 * LATENT_DIM
kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42)
cluster_labels = kmeans.fit_predict(M)

best_combo = (None, 0.0, None, None)  # neurons, ratio, category, cluster_id
for cluster_id in range(N_CLUSTERS):
    members = np.where(cluster_labels == cluster_id)[0]
    if len(members) < MIN_COUNT:
        continue
    dists = np.linalg.norm(M[members] - kmeans.cluster_centers_[cluster_id], axis=1)
    top = members[np.argsort(dists)][:TOP_N]
    cnt = Counter()
    for i in top:
        cnt.update(categories_by_item[int(i)])
    cat, ratio, n = most_enriched(cnt)
    if ratio > best_combo[1]:
        top_neurons = np.argsort(-np.abs(kmeans.cluster_centers_[cluster_id]))[:3].tolist()
        deltas = {int(nn): float(kmeans.cluster_centers_[cluster_id][nn]) for nn in top_neurons}
        best_combo = (deltas, ratio, cat, cluster_id)

print(f"Best combination cluster {best_combo[3]}: neurons={list(best_combo[0].keys())}, "
      f"category={best_combo[2]!r}, enrichment={best_combo[1]:.1f}x")

# Measure the effect of intervention, averaged over many synthetic profiles
elicitation = PopularitySamplingElicitation(loader.ratings_df, n_samples=6, k=1.0)
rng = np.random.default_rng(123)
profiles = [elicitation.get_initial_data().tolist() for _ in range(40)]

mu_scale = None
with __import__("torch").no_grad():
    import torch as _torch
    uv = np.zeros((1, algo._items_count), dtype=np.float32)
    for i in profiles[0]:
        uv[0, i] = 1.0
    mu, _ = algo._model.encode(_torch.from_numpy(uv))
    mu_scale = mu.std().item()
print(f"\nLatent mu std (for delta calibration): {mu_scale:.4f}")


def match_rate(interventions, category, k=10):
    if interventions:
        algo.set_neuron_interventions(interventions)
    else:
        algo.clear_neuron_interventions()
    matches = []
    for profile in profiles:
        recs = algo.predict(selected_items=profile, filter_out_items=profile, k=k)
        matches.append(sum(1 for i in recs if category in categories_by_item.get(i, [])) / k)
    return float(np.mean(matches))


print("\n=== Single-neuron intervention ===")
neuron, _, cat = best_single
baseline = match_rate({}, cat)
print(f"Category: {cat!r}. Baseline match rate (no intervention): {baseline:.3f}")
for scale in [4, 8, 16, 32]:
    delta = scale * mu_scale
    boosted = match_rate({neuron: delta}, cat)
    suppressed = match_rate({neuron: -delta}, cat)
    print(f"  delta=+{scale}*std ({delta:.2f}): boost={boosted:.3f}  suppress(-)={suppressed:.3f}")

print("\n=== Combination intervention ===")
combo_deltas, _, combo_cat, _ = best_combo
baseline_combo = match_rate({}, combo_cat)
print(f"Category: {combo_cat!r}. Baseline match rate: {baseline_combo:.3f}")
for scale in [4, 8, 16, 32]:
    scaled = {n: scale * mu_scale * np.sign(d) if d != 0 else scale * mu_scale for n, d in combo_deltas.items()}
    boosted = match_rate(scaled, combo_cat)
    print(f"  scale={scale}*std, neurons={scaled}: boost={boosted:.3f}")

algo.clear_neuron_interventions()
print("\nDone.")
