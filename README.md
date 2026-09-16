# Guitar domain extension for EasyStudy — notebooks & data pipeline

A guitar e-commerce domain added to [EasyStudy](https://github.com/pdokoupil/EasyStudy)
(`ndbi021` branch), built on the public **Amazon Reviews 2023** dataset
(McAuley Lab, UCSD). This repo holds the data pipeline and the analysis
notebooks; the EasyStudy code itself (algorithms, loader, live UI) is on
the `ndbi021` branch of the forked repo:
[XMON3R/ES_GuitarDomain](https://github.com/XMON3R/ES_GuitarDomain).

## Start here

`research/highlights.ipynb` — curated headline results (also exported to
`exports/highlights.md` / `exports/highlights.pdf`).

## Contents

| File | What it is |
|---|---|
| `research/highlights.ipynb` | Curated "most important / most surprising" results |
| `research/guitar_evaluation.ipynb` | Full online + offline evaluation (executed) |
| `research/guitar_interpretability.ipynb` | Sparse-latent interpretability analysis (executed) |
| `research/filter_guitar_products.py` | Filters Amazon Reviews 2023 metadata down to guitar products |
| `research/check_interaction_volume.py` | Counts/validates the resulting interaction volume |
| `research/build_dataset.py` | Builds the final `items.csv` / `interactions.csv` from the filtered data |
| `research/test_in_processing.py` | Validates the in-processing intervention result (single-neuron vs. neuron-combination) referenced in `highlights.ipynb` |
| `GuitarModels.csv` | Curated historical guitar-model metadata used to enrich the item catalog |
| `exports/` | Markdown + PDF exports of all three notebooks, with graphs |

## Dataset

Public, no login/API key required — pulled directly over HTTPS from
`mcauleylab.ucsd.edu`'s Amazon Reviews 2023 host:
- `meta_Musical_Instruments.jsonl.gz` (product metadata, ~155 MB)
- `Musical_Instruments.jsonl.gz` (reviews, ~459 MB)

Both scripts below download the file they need automatically the first
time they run (into `research/data/`) if it isn't already there — no
separate manual download step.

## Running the pipeline

```bash
cd research
python filter_guitar_products.py      # downloads meta_Musical_Instruments.jsonl.gz if missing
python check_interaction_volume.py    # downloads Musical_Instruments.jsonl.gz if missing
python build_dataset.py               # writes items.csv / interactions.csv
```

`filter_guitar_products.py` and `check_interaction_volume.py` are fully
standalone. `build_dataset.py`, `test_in_processing.py`, and both
`guitar_evaluation.ipynb`/`guitar_interpretability.ipynb` all import from
the EasyStudy codebase and expect an `EasyStudy/` folder as a sibling of
this repo's root (i.e. `EasyStudy/` and this repo checked out next to each
other) — that's where `build_dataset.py` writes its output and where the
others import the algorithm/loader code from
([XMON3R/ES_GuitarDomain](https://github.com/XMON3R/ES_GuitarDomain),
`ndbi021` branch).

```bash
python test_in_processing.py          # validates the headline intervention result (~8-10 min)
```

Then open the notebooks in Jupyter (or read the `exports/` versions directly).
