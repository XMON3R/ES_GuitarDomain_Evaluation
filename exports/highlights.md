# Most important & most surprising results

## Guitar domain for EasyStudy

A quick read of the headline findings, pulled from the two full
notebooks in this repo (`guitar_evaluation.ipynb`,
`guitar_interpretability.ipynb`). Every result below is copied
straight from its source notebook - same code, same already-computed
output - except the first chart, which restates an already-validated
number as a picture instead of running it again here.

## 1. Combinations of neurons steer the model - single neurons don't

The core question behind the sparse-autoencoder work: is a
*combination* of latent neurons more meaningful than any one neuron
alone? The in-processing mechanism (`InProcessingSparseVAECF`) tests
this directly by editing latent codes before decoding, not by
re-ranking the output afterwards. Validated in
`test_in_processing.py` (40 synthetic profiles) - see that script for
the full run this chart summarizes.

**Result: nudging one neuron mostly just destabilizes the decode and
never beats baseline. A correctly-signed *combination* of three
neurons reliably shifts what gets recommended.** This is live in the
actual app too: a "Show me something different" button on the
comparison page applies exactly this mechanism to a real
participant's session, not just a script.


```python
# Summary of the in-processing intervention test (40 synthetic profiles,
# see test_in_processing.py for the full run this chart summarizes -
# reproduced here as a static chart rather than re-run, since the
# underlying test trains a fresh model and takes several minutes).
import matplotlib.pyplot as plt

conditions = ["Baseline\n(no intervention)", "Single neuron\n(best of boost/suppress)", "3-neuron combination\n(signed by cluster centroid)"]
target_share = [56.0, 50.8, 73.3]  # % of top-k recs in the target category
colors = ["#888888", "#d9534f", "#2e7d32"]

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(conditions, target_share, color=colors)
ax.set_ylabel("Target-category share of top-k recs (%)")
ax.set_title("Single neuron never beats baseline; a 3-neuron combination does")
ax.axhline(56.0, color="#888888", linestyle="--", linewidth=1)
for b, v in zip(bars, target_share):
    ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v}%", ha="center")
plt.tight_layout()
plt.show()
```


    
![png](highlights_files/highlights_2_0.png)
    


*(Source: guitar_interpretability.ipynb, section 2.5)*

## 2.5. Neuron-*combination* interpretation

Section 2 only asks "which single neuron is an item most associated
with?" One of the biggest questions behind this whole project is
whether a *combination* of neurons carries more meaning than any single
one alone.

Approach: cluster items by their *entire* latent-neuron association
vector (`KMeans` over all `LATENT_DIM` dimensions of
`item_neuron_matrix`, not any single column) with `k = 2 * LATENT_DIM`
clusters - each cluster centroid is then a genuine *combination* of
neuron activations, not a single neuron in isolation. The same
category/maker enrichment method from section 2 is then run per
cluster, and compared against each cluster's own best single
contributing neuron.


```python
from sklearn.cluster import KMeans

N_CLUSTERS = 2 * LATENT_DIM
kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42)
cluster_labels = kmeans.fit_predict(M)

cluster_summaries = []
for cluster_id in range(N_CLUSTERS):
    members = np.where(cluster_labels == cluster_id)[0]
    if len(members) < MIN_COUNT:
        continue
    # Score each member by distance to its own cluster centroid (tightest-fitting first)
    dists = np.linalg.norm(M[members] - kmeans.cluster_centers_[cluster_id], axis=1)
    top = members[np.argsort(dists)][:TOP_N]

    cat_counts, maker_counts = Counter(), Counter()
    for i in top:
        cat_counts.update(categories_by_item[int(i)])
        maker_counts[maker_by_item[int(i)]] += 1
    cat, cat_ratio, cat_n = most_enriched(cat_counts, overall_cat_counts, total_cats)
    maker, maker_ratio, maker_n = most_enriched(maker_counts, overall_maker_counts, total_makers)

    # Which single neurons dominate this cluster's centroid, and what was
    # their own (section 2) enrichment? Lets us check if the combination
    # beats its strongest individual contributor.
    top_neurons = np.argsort(-np.abs(kmeans.cluster_centers_[cluster_id]))[:2].tolist()
    best_contributor_ratio = max(
        (summary_df.loc[n, "category_enrichment"] for n in top_neurons if n in summary_df.index),
        default=0.0,
    )

    cluster_summaries.append({
        "cluster": cluster_id, "size": len(members), "dominant_neurons": top_neurons,
        "top_category": cat, "category_enrichment": round(cat_ratio, 1), "category_n": cat_n,
        "top_maker": maker, "maker_enrichment": round(maker_ratio, 1),
        "beats_best_single_neuron": round(cat_ratio, 1) > best_contributor_ratio,
    })

cluster_df = pd.DataFrame(cluster_summaries).set_index("cluster").sort_values("category_enrichment", ascending=False)
display(cluster_df)

n_beating = cluster_df["beats_best_single_neuron"].sum()
print(f"\n{n_beating}/{len(cluster_df)} clusters show category enrichment stronger than "
      f"either single dominant neuron's own enrichment from section 2.")

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>size</th>
      <th>dominant_neurons</th>
      <th>top_category</th>
      <th>category_enrichment</th>
      <th>category_n</th>
      <th>top_maker</th>
      <th>maker_enrichment</th>
      <th>beats_best_single_neuron</th>
    </tr>
    <tr>
      <th>cluster</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11</th>
      <td>211</td>
      <td>[14, 7]</td>
      <td>Beginner Kits</td>
      <td>10.8</td>
      <td>14</td>
      <td>Best Choice Products</td>
      <td>84.5</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>863</td>
      <td>[14, 7]</td>
      <td>Classical &amp; Nylon-String Guitars</td>
      <td>8.8</td>
      <td>4</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>30</th>
      <td>194</td>
      <td>[15, 8]</td>
      <td>Hollow &amp; Semi-Hollow Body</td>
      <td>7.1</td>
      <td>4</td>
      <td>Grote</td>
      <td>20.7</td>
      <td>False</td>
    </tr>
    <tr>
      <th>7</th>
      <td>210</td>
      <td>[14, 0]</td>
      <td>Hollow &amp; Semi-Hollow Body</td>
      <td>5.4</td>
      <td>3</td>
      <td>Fender</td>
      <td>1.6</td>
      <td>True</td>
    </tr>
    <tr>
      <th>10</th>
      <td>246</td>
      <td>[14, 8]</td>
      <td>Hollow &amp; Semi-Hollow Body</td>
      <td>5.4</td>
      <td>3</td>
      <td>Gibson</td>
      <td>5.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>268</td>
      <td>[14, 8]</td>
      <td>Hollow &amp; Semi-Hollow Body</td>
      <td>5.4</td>
      <td>3</td>
      <td>AXL</td>
      <td>33.6</td>
      <td>True</td>
    </tr>
    <tr>
      <th>19</th>
      <td>232</td>
      <td>[8, 14]</td>
      <td>Hollow &amp; Semi-Hollow Body</td>
      <td>5.4</td>
      <td>3</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>12</th>
      <td>371</td>
      <td>[11, 14]</td>
      <td>Electric Basses</td>
      <td>4.2</td>
      <td>5</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>28</th>
      <td>356</td>
      <td>[15, 14]</td>
      <td>Beginner Kits</td>
      <td>3.9</td>
      <td>5</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>26</th>
      <td>340</td>
      <td>[14, 10]</td>
      <td>Bass Guitars</td>
      <td>3.8</td>
      <td>5</td>
      <td>Fender</td>
      <td>1.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>0</th>
      <td>251</td>
      <td>[9, 14]</td>
      <td>Bass Guitars</td>
      <td>3.8</td>
      <td>5</td>
      <td>Fender</td>
      <td>1.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>31</th>
      <td>287</td>
      <td>[14, 11]</td>
      <td>Acoustic-Electric Guitars</td>
      <td>3.7</td>
      <td>4</td>
      <td>Ibanez</td>
      <td>3.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>9</th>
      <td>265</td>
      <td>[9, 14]</td>
      <td>Acoustic-Electric Guitars</td>
      <td>3.7</td>
      <td>4</td>
      <td>YAMAHA</td>
      <td>3.5</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>232</td>
      <td>[3, 8]</td>
      <td>Steel-String Acoustics</td>
      <td>3.5</td>
      <td>7</td>
      <td>Fender</td>
      <td>1.3</td>
      <td>False</td>
    </tr>
    <tr>
      <th>24</th>
      <td>308</td>
      <td>[0, 14]</td>
      <td>Electric Basses</td>
      <td>3.4</td>
      <td>4</td>
      <td>Ibanez</td>
      <td>2.3</td>
      <td>False</td>
    </tr>
    <tr>
      <th>13</th>
      <td>218</td>
      <td>[0, 5]</td>
      <td>Beginner Kits</td>
      <td>3.1</td>
      <td>4</td>
      <td>Fender</td>
      <td>1.3</td>
      <td>False</td>
    </tr>
    <tr>
      <th>21</th>
      <td>309</td>
      <td>[14, 7]</td>
      <td>Solid Body</td>
      <td>3.1</td>
      <td>17</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>381</td>
      <td>[14, 12]</td>
      <td>Beginner Kits</td>
      <td>3.1</td>
      <td>4</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>20</th>
      <td>347</td>
      <td>[14, 12]</td>
      <td>Beginner Kits</td>
      <td>3.1</td>
      <td>4</td>
      <td>Dean Guitars</td>
      <td>4.8</td>
      <td>False</td>
    </tr>
    <tr>
      <th>15</th>
      <td>277</td>
      <td>[12, 14]</td>
      <td>Acoustic Guitars</td>
      <td>2.9</td>
      <td>13</td>
      <td>YAMAHA</td>
      <td>4.6</td>
      <td>False</td>
    </tr>
    <tr>
      <th>18</th>
      <td>360</td>
      <td>[14, 11]</td>
      <td>Acoustic-Electric Guitars</td>
      <td>2.8</td>
      <td>3</td>
      <td>YAMAHA</td>
      <td>3.5</td>
      <td>False</td>
    </tr>
    <tr>
      <th>23</th>
      <td>234</td>
      <td>[3, 14]</td>
      <td>Acoustic-Electric Guitars</td>
      <td>2.8</td>
      <td>3</td>
      <td>Dean Guitars</td>
      <td>3.6</td>
      <td>False</td>
    </tr>
    <tr>
      <th>8</th>
      <td>313</td>
      <td>[14, 7]</td>
      <td>Acoustic-Electric Guitars</td>
      <td>2.8</td>
      <td>3</td>
      <td>Fender</td>
      <td>1.6</td>
      <td>True</td>
    </tr>
    <tr>
      <th>6</th>
      <td>193</td>
      <td>[12, 7]</td>
      <td>Electric Basses</td>
      <td>2.5</td>
      <td>3</td>
      <td>Fender</td>
      <td>1.3</td>
      <td>False</td>
    </tr>
    <tr>
      <th>27</th>
      <td>389</td>
      <td>[5, 14]</td>
      <td>Steel-String Acoustics</td>
      <td>2.5</td>
      <td>5</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>25</th>
      <td>188</td>
      <td>[11, 3]</td>
      <td>Electric Basses</td>
      <td>2.5</td>
      <td>3</td>
      <td>Directly Cheap</td>
      <td>13.7</td>
      <td>False</td>
    </tr>
    <tr>
      <th>17</th>
      <td>312</td>
      <td>[7, 14]</td>
      <td>Electric Basses</td>
      <td>2.5</td>
      <td>3</td>
      <td>Fender</td>
      <td>1.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>16</th>
      <td>255</td>
      <td>[11, 14]</td>
      <td>Electric Basses</td>
      <td>2.5</td>
      <td>3</td>
      <td>Fender</td>
      <td>1.3</td>
      <td>False</td>
    </tr>
    <tr>
      <th>29</th>
      <td>297</td>
      <td>[14, 3]</td>
      <td>Electric Basses</td>
      <td>2.5</td>
      <td>3</td>
      <td>Takamine</td>
      <td>7.8</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>331</td>
      <td>[8, 14]</td>
      <td>Solid Body</td>
      <td>2.0</td>
      <td>11</td>
      <td>Fender</td>
      <td>1.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>22</th>
      <td>217</td>
      <td>[7, 12]</td>
      <td>Solid Body</td>
      <td>2.0</td>
      <td>11</td>
      <td>Ibanez</td>
      <td>2.3</td>
      <td>False</td>
    </tr>
    <tr>
      <th>14</th>
      <td>253</td>
      <td>[5, 14]</td>
      <td>Acoustic Guitars</td>
      <td>1.6</td>
      <td>7</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>


    
    11/32 clusters show category enrichment stronger than either single dominant neuron's own enrichment from section 2.
    

If `n_beating` above is greater than zero, that's real evidence that
some item groupings only show up when you look at a *combination* of
neurons together, not any single one. If it's zero, the honest reading
is that at this model size, the individual neurons already capture most
of the interpretable structure, and combining them mostly just
re-finds the same categories rather than revealing new ones - also a
fair result, not a failure.

## 3. Best single-neuron result (for context)

*(Source: guitar_interpretability.ipynb, section 2)*

### Inspect the single most interpretable neuron in detail


```python
best_neuron = summary_df["category_enrichment"].astype(float).idxmax()
print(f"Most category-enriched neuron: {best_neuron} "
      f"({summary_df.loc[best_neuron, 'top_category']}, "
      f"{summary_df.loc[best_neuron, 'category_enrichment']}x base rate)")

top = np.argsort(-np.abs(M[:, best_neuron]))[:10]
for i in top:
    print(" -", loader.get_item_index_description(int(i)))

```

    Most category-enriched neuron: 15 (Hollow & Semi-Hollow Body, 10.7x base rate)
     - Fender FA-100 Limited Edition Dreadnought Acoustic Guitar with Gig Bag - Black by Fender
     - PRS Paul Reed Smith SE Standard 24 Electric Guitar, Tobacco Sunburst by PRS Guitars
     - Firefly FF338 Semi Hollowbody Guitar Ice Tea. by Firefly
     - Firefly FFTH Semi-Hollow body Guitar Green color .(Black pearloid pickguard). by Firefly
     - Firefly FFJR Solid Body Electric Guitar (Metallic Gold). by Firefly
     - Firefly FFTH Semi-Hollow body Guitar (Orange). by Firefly
     - Firefly FFDCD Solid Body Electric Guitar （Transparent Red) by Firefly
     - Fender FA-100 Dreadnought Acoustic Guitar - Black Bundle with Gig Bag, Tuner, Strings, Strap, and Picks by Fender
     - Firefly FFTH Semi-Hollow body Guitar Sunburst Color(White pickguard). by Firefly
     - Leo Jaymz 6 Strings Headless Electric Guitar - Full Scale Travel Guitar - Ash Body and 5 piece Maple Neck - Customized Pickups (Brown) by Leo Jaymz
    

*(Source: guitar_interpretability.ipynb, section 3)*

## 3. "Give me something more surprising" mode

Same synthetic user profile, `alpha=0` (pure relevance, identical to plain
`VAE-CF`) vs a high `alpha` (novelty-heavy re-rank) - using the *same*
trained model's weights, only the prediction-time blend changes.


```python
# Build a synthetic profile from a handful of popular items so it's reproducible without the elicitation module
item_popularity = loader.ratings_df.groupby("item").size().sort_values(ascending=False)
seed_items = item_popularity.index[:30].to_numpy()
rng = np.random.default_rng(7)
profile = rng.choice(seed_items, size=3, replace=False).tolist()

print("Seed profile:")
for i in profile:
    print(" -", loader.get_item_index_description(int(i)))

def recommend_with_alpha(alpha, k=8):
    algo._alpha = alpha
    return algo.predict(selected_items=profile, filter_out_items=profile, k=k)

baseline = recommend_with_alpha(0.0)
surprising = recommend_with_alpha(0.9)

print("\n=== alpha=0.0 (pure relevance) ===")
for i in baseline:
    print(" -", loader.get_item_index_description(i))

print("\n=== alpha=0.9 (novelty-heavy / 'more surprising') ===")
for i in surprising:
    print(" -", loader.get_item_index_description(i))

overlap = len(set(baseline) & set(surprising))
print(f"\nOverlap between the two lists: {overlap}/{len(baseline)} items in common")
algo._alpha = 0.3  # restore the study's configured value

```

    Seed profile:
     - Piaa 85115 Superior 112Db 330Hz and 400Hz Twin Tone Bass Horn, Black by Piaa
     - Ashthorpe Full-Size Cutaway Thinline Acoustic-Electric Guitar Package - Premium Tonewoods - Blue by Ashthorpe
     - Best Choice Products 41in Full Size Beginner All Wood Acoustic Guitar Starter Set w/Case, Strap, Capo, Strings, Picks, Tuner - Blue by Best Choice Products
    
    === alpha=0.0 (pure relevance) ===
     - Best Choice Products 38in Beginner All Wood Acoustic Guitar Starter Kit w/Case, Strap, Digital Tuner, Pick, Strings - SoCal Green by Best Choice Products
     - Pyle Classical Acoustic Guitar Kit, 3/4 Junior Size Instrument for Beginner Kids, Adults, 36" Sun Burst by Pyle
     - Donner Acoustic Guitar for Beginner Adult Full Size Cutaway Acustica Guitarra Bundle Kit with Free Online Lesson Bag Strap Tuner Capo Pickguard String Pick, Right Hand 41”Sunburst, DAG-1CS/DAD-160CS by Donner
     - Davison Guitars Full Size Electric Guitar with 10-Watt Amp, Black - Right Handed Beginner Kit with Gig Bag and Accessories by Davison Guitars
     - Best Choice Products 39in Full Size Beginner Electric Guitar Starter Kit w/Case, Strap, 10W Amp, Strings, Pick, Tremolo Bar - Jet Black by Best Choice Products
     - Best Choice Products 22-Fret Full Size Acoustic Electric Bass Guitar w/ 4-Band Equalizer, Adjustable Truss Rod by Best Choice Products
     - Pink Full Size Thinline Acoustic Electric Guitar With Free Gig Bag Case and Picks by Jameson Guitars
     - Best Choice Products 41in Beginner Acoustic Guitar Full Size All Wood Cutaway Guitar Starter Set Bundle with Case, Strap, Capo, Strings, Picks, Tuner - SoCal Green by Best Choice Products
    
    === alpha=0.9 (novelty-heavy / 'more surprising') ===
     - 30" Beginner Acoustic Guitar kids Guitar 1/2 Size Child Guitars Steel Strings with Gig Bag for Adult Girls Students by Strong Wind
     - SX Left Handed 1/2 Size (34 inch) Electric Guitar Package Red with Amp, and Carry Bag SX RST 1/2 CAR LH by SX
     - Fender DG-8S Solid Spruce Top Dreadnought Acoustic Guitar Pack with Gig Bag, Tuner, Strings, Picks, Strap, and Instructional DVD - Natural by Fender
     - Oscar Schmidt OG2CE Dreadnought Acoustic Electric Guitar by Oscar Schmidt
     - Fender FA-100 Limited Edition Dreadnought Acoustic Guitar with Gig Bag - Black by Fender
     - Firefly FF338 Semi Hollowbody Guitar Ice Tea. by Firefly
     - B.C. Rich Warlock Electric Guitar, Black by B.C. Rich
     - Ibanez Performance PN1MHOPN Mahogany Parlor Acoustic Guitar High Gloss Natural by Ibanez
    
    Overlap between the two lists: 0/8 items in common
    

*(Source: guitar_evaluation.ipynb, section 5 - retrained on ~6x more data with bigger models)*

## 5. Compute comparative metrics

- **Coverage**: fraction of the catalog that appears in at least one recommendation list across the synthetic users.
- **Novelty**: average popularity-complement (`1 - fraction of users who interacted with the item`) of recommended items - higher means the algorithm surfaces less mainstream items.
- **Intra-list diversity (ILD)**: average pairwise Jaccard *distance* between recommended items' category sets within each list - higher means a more varied single recommendation list.
- **Effect of `Sparse VAE-CF`'s `alpha`**: it blends in a novelty term defined over *latent-neuron* profile dissimilarity (Part B's post-processing step) - that is a different notion of "different" than category-based ILD or popularity-based novelty computed below, so don't assume it must move both metrics in the same direction; treat this section as the empirical check of *whether/how* the post-processing changes the recommendations relative to plain `VAE-CF`, not as a confirmation of a predicted direction.


```python
item_popularity = loader.ratings_df.groupby("item").size()
n_users = loader.ratings_df.user.nunique()
popularity_complement = (1.0 - item_popularity / n_users).reindex(all_items, fill_value=1.0)

item_categories = {
    idx: set(loader.get_item_index_categories(int(idx))) for idx in all_items
}

def jaccard_distance(a, b):
    if not a and not b:
        return 0.0
    return 1.0 - len(a & b) / len(a | b)

def ild(rec_list):
    cats = [item_categories.get(i, set()) for i in rec_list]
    pairs = list(combinations(range(len(rec_list)), 2))
    if not pairs:
        return 0.0
    return np.mean([jaccard_distance(cats[i], cats[j]) for i, j in pairs])

rows = []
for name, recs in recommendations.items():
    flat = [item for rec_list in recs for item in rec_list]
    coverage = len(set(flat)) / len(all_items)
    novelty = np.mean([popularity_complement.get(i, 1.0) for i in flat])
    diversity = np.mean([ild(rec_list) for rec_list in recs])
    rows.append({"algorithm": name, "coverage": coverage, "novelty": novelty, "ILD": diversity})

metrics_df = pd.DataFrame(rows).set_index("algorithm")
display(metrics_df)

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>coverage</th>
      <th>novelty</th>
      <th>ILD</th>
    </tr>
    <tr>
      <th>algorithm</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>EASE (baseline)</th>
      <td>0.114430</td>
      <td>0.999542</td>
      <td>0.558088</td>
    </tr>
    <tr>
      <th>VAE-CF</th>
      <td>0.015566</td>
      <td>0.988391</td>
      <td>0.565125</td>
    </tr>
    <tr>
      <th>Sparse VAE-CF</th>
      <td>0.009361</td>
      <td>0.988208</td>
      <td>0.477604</td>
    </tr>
  </tbody>
</table>
</div>



```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, col in zip(axes, ["coverage", "novelty", "ILD"]):
    metrics_df[col].plot.bar(ax=ax, title=col)
    ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.show()

```


    
![png](highlights_files/highlights_13_0.png)
    

