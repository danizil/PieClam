# Project Context for Claude Code

## What This Project Is

This repository implements **PieClam** and related **CLAM / BigCLAM–style** overlapping community models on graphs, extended with a **learned prior** (RealNVP normalizing flow) and optional **node attributes** and **Lorentz (“lightcone”) geometry** for certain variants.

**Purpose:** Learn latent community-affiliation features `x` so that **edge probability** is expressed through an inner-product structure (and optional flow prior), then use `x` for **link prediction** (with dyad omission), **unsupervised anomaly detection** (likelihood vs. held-out labels), or **distance / synthetic** studies.

**Model variants (boolean flags → names):**

| `vanilla` | `lorenz` | Name     |
|-----------|----------|----------|
| True      | False    | bigclam  |
| True      | True     | ieclam   |
| False     | False    | pclam    |
| False     | True     | pieclam  |

**Core math (simplified):**

- **Messages:** For “active” dyads (`edge_attr` True), use a BigCLAM-like term: inner product \( \langle x_i, B x_j \rangle \) (with vector `B` or diagonal `B` in directed code), then `x_j / (1 - exp(-inner))` (plus `eps`); for **omitted** dyads, skip the denominator branch and pass `x_j` through (see `message` in `clamiter.py` / `clamiter_directed.py`).
- **PCLAM / PieCLAM:** **RealNVP** prior on `x` (and `attr` when `attr_opt`); optimization **alternates** feature updates (MPNN-style `ClamIter` “gradient”) and prior training (`prior_opt`, `back_forth` in YAML).
- **IECLAM / PieCLAM (Lorenz):** Features split into “space/time” halves; **`relu_lightcone`** implements rotated ReLU updates.
- **Directed:** Features split into **sender / receiver** halves; **two** `propagate` passes (flipped `edge_index` vs. swapped feature halves); bilinear inner product `einsum` with matrix **`B`**; optional **degree scaling** on neighbor messages (`normalize_degree`, `deg_in` / `deg_out`, `pow_in` / `pow_out`).

---

## Codebase Architecture

### Core model & training

- **`clamiter.py`** — `ClamIter(MessagePassing)` for **undirected** graphs. `B` is a **1D** tensor (Lorenz: `+1`/`-1` blocks). Single `propagate` per `forward`. `message`: `einsum('ij,ij->i', x_i, self.B*x_j)`. `update`: `self.B*(aggr_out - global_term + x) + global_features - ...`. Imports **`transformation`** (not `_directed`). No `directed` flag.

- **`clamiter_directed.py`** — Same class name **`ClamIter`**, unified for **directed and undirected** via `self.directed` and `graph.is_directed()`. Adds **`directed`**, **`normalize_degree`**; **`B`** as **diagonal matrix** when directed; **two propagations** for directed graphs; **`message`** takes `deg_in`, `deg_out`, `pow_in`, `pow_out` (lists of per-edge tensors for retained vs omitted dyads when `normalize_degree`); **`update`** uses `(aggr_out - global_term) @ self.B + ...` with directed-specific slicing. Also: **`StarProb`**, link-prediction and anomaly helpers, `load_model`, etc. **`degree`** is available via **`from utils.plotting import *`** (plotting re-exports `torch_geometric.utils.degree`). Imports **`transformation_directed`**.

- **`trainer.py`** — **`Trainer`**: loads `hypers/hypers_{task}.yaml` where `task` defaults to **`distance`** if `task is None` (see constructor). Instantiates **`clamiter.ClamIter`** without `directed`. Optional injected `clamiter`. `transform_attributes` on `raw_attr` → `attr`, then deletes `raw_attr`.

- **`trainer_directed.py`** — **`Trainer`** using **`clamiter_directed`**. Hypers file: **`hypers_{task}.yaml`** for `task in ['distance','anomaly_unsupervised','link_prediction']`, else **`hypers_losses.yaml`**. Passes **`directed=self.data.is_directed()`** into `ClamIter`. `clamiter` constructor argument is commented out vs. base trainer.

- **`transformation.py`** — RealNVP (`normflows`), `train_prior`, **`relu_transform`**, **`relu_lightcone`**, hyperbolic / Euclidean maps, activation registry.

- **`transformation_directed.py`** — Parallel module; **`relu_lightcone`** docstring mentions directed sender/receiver; otherwise same structure as `transformation.py`.

- **`datasets/import_dataset.py`** — **`import_dataset`**: Planetoid npz, WebKB, OGB, squirrel, Facebook `.mat`, anomaly `.mat` (photo, reddit, elliptic, …), simulations (`sbm3x3`, bipartite, …). Sets **`edge_attr`** bool, **`remove_isolated_nodes`**, masks **`y`**, **`gt_nomalous`**, train/test fields; **`transform_attributes`** (sparse vs dense handling). **`direct_graph`** helper.

- **`datasets/simulations.py`** — Synthetic graphs for `import_dataset` by name.

- **`datasets/data_utils.py`** — e.g. `intersecting_tensor_from_non_intersecting_vec`.

- **`datasets/anomaly_detection/.../clean_raw.py`** — Elliptic CSV → labeled-only PyG `Data`, `.pt` output.

### `utils/`

- **`utils.py`** — Graph probabilities (`get_prob_graph`), directed/undirected edge scoring, cutnorm, conductance, dyads, `TwoHop`, etc.

- **`pyg_helpers.py`** — Sparse → `edge_index`, coalesce, bidirectional filters, MAT helpers.

- **`link_prediction.py`** — **`omit_dyads_random`**, AUC, dyad sampling with `directed` and `edge_attr`.

- **`plotting.py`** — Feature/state plots; **`from torch_geometric.utils import ... degree`** (side effect: `clamiter_directed` relies on `import *` for `degree`).

- **`printing_utils.py`** — `printd` and formatted debug output.

- **`path_utils.py`** — `get_project_root`.

- **`pring_opt.py`** — Optimization / logging helpers.

### `hypers/` (YAML)

| File | Role |
|------|------|
| **`hypers_link_prediction.yaml`** | `task='link_prediction'` — per-dataset blocks + `GlobalConfigs_{model}` |
| **`hypers_anomaly_unsupervised.yaml`** | Anomaly tasks |
| **`hypers_distance.yaml`** | Distance / synthetic benchmarks |
| **`hypers_losses.yaml`** | Fallback / loss ablations; used when `task` is not one of the three named tasks (`trainer_directed`) |

Each file: **`{dataset}_{model}`** sections (e.g. `cora-ml_pieclam`, `elliptic_pclam`) plus **`GlobalConfigs_bigclam`**, **`GlobalConfigs_ieclam`**, **`GlobalConfigs_pclam`**, **`GlobalConfigs_pieclam`**.

### `experiments/`

- **`optimization_utils.py`** — Undirected-oriented **`SaveRun`**, **`print_folder`**, hyper loaders, batch drivers; link-pred results path **`experiments/results/undirected/...`** (when `directed=False`).

- **`optimization_utils_directed.py`** — Uses **`trainer_directed.Trainer`**; **`SaveRun(..., directed=True)`** writes under **`experiments/results/directed/...`** for link prediction.

- **`scripts/`** — SLURM (`*.sbatch`), `global_config.py`, hypers search, anomaly pipelines (`find_communities`, `densification`, `confidence_intervals`), `test_full_models`.

### Notebooks (`experiments/`)

- **`link_prediction.ipynb`** — Main LP workflows, sweeps, large outputs.
- **`anomaly_unsupervised.ipynb`** — Anomaly benchmarks (photo, reddit, elliptic, …).
- **`distance.ipynb`** — Distance experiments.
- **`shapes_*.ipynb`, `bipartite.ipynb`** — Synthetic demos.
- **`results/.../analysis.ipynb`** — Aggregating JSON results.

### Other

- **`tests/tests.py`** — Tests (often not wired into main flows).
- **`setup.py`** — Package metadata.

---

## Directed Extension (Active Branch)

The **`directed`** line of work is **not** a tiny patch: it is a **parallel stack** (`clamiter_directed.py`, `trainer_directed.py`, `optimization_utils_directed.py`, `experiments/results/directed/`) plus **graphs with `is_directed()` True**.

### `clamiter_directed.py` vs `clamiter.py`

| Aspect | Undirected (`clamiter.py`) | Directed (`clamiter_directed.py`) |
|--------|----------------------------|-----------------------------------|
| **`ClamIter.__init__`** | No `directed`; `B` is 1D vector | `directed: bool`; if directed, `B` is **`torch.diag(...)`**; Lorentz+directed requires **`dim_feat % 4 == 0`** |
| **`forward`** | One `propagate` | If **directed**: prior grad split; **`x_flipped`**; **two** `propagate` calls (flip `edge_index` vs. use `x_flipped`); optional **in/out degree** tensors split into **retained vs omitted** dyads |
| **`message`** | `x_i * (B*x_j)` style inner product; no degree args | Bilinear **`einsum('ij,jk,ik->i', x_i[:,:H], B, x_j[:,H:])`**; **`deg_in`/`deg_out`** as `[ret, omitted]` lists; **`pow_in`/`pow_out`** scale |
| **`update`** | `self.B*(aggr_out - global_term + x) + ...` | **`(aggr_out - global_term) @ self.B + ...`** with directed slicing on **receiver** half and `l1` on appropriate columns |
| **Imports** | `transformation` | `transformation_directed` |

Undirected graphs still go through **`clamiter_directed.py`** when using **`trainer_directed`**, with `directed=False` and a single `propagate` branch.

### `trainer_directed.py` vs `trainer.py`

- Imports **`clamiter_directed`**; builds **`ClamIter(..., directed=self.data.is_directed(), ...)`**.
- Hypers path selection differs from base `trainer.py` when `task` is `None` (base uses `distance`; directed uses **`losses`**).

### `transformation_directed.py` vs `transformation.py`

- Largely duplicated; directed variant documents **sender/receiver** in **`relu_lightcone`**.

### `optimization_utils_directed.py` vs `optimization_utils.py`

- **`SaveRun`**: `directed=True` → `results/directed/link_prediction/...`; anomaly path structure overlaps but link-pred tree is explicit.

---

## Data Format

PyTorch Geometric **`Data`** fields used across tasks:

| Field | Meaning |
|--------|---------|
| **`x`** | Node features `[N, dim_feat]` (affiliations; or full input for some loaders). |
| **`edge_index`** | `[2, E]`; orientation matters for directed graphs. |
| **`edge_attr`** | **`torch.bool`**, length `E`: **`True`** = use full neighbor message (non-omitted dyad); **`False`** = omitted / negative dyad (linear branch in `message`). |
| **`y`** | Node labels where applicable (communities, classes). |
| **`gt_nomalous`** | **Bool** (spelling as in code). MAT anomaly loaders: **`True` ≈ normal**, **`False` ≈ anomaly** (see scoring comments in `clamiter*.py`). Default zeros if missing. **Not** spelled `gt_anomalous`. |
| **`raw_attr` / `attr`** | Raw scipy sparse **attrs** → **`transform_attributes`** → dense **`attr`** for PCLAM/PieCLAM with `attr_opt`; `raw_attr` deleted after transform in `Trainer`. |
| **`num_nodes`** | Set after cleanup in `import_dataset`. |
| **`name`** | Dataset string for YAML keys. |
| **Anomaly masks** | `train_node_mask`, `test_normal_mask`, `test_anomalies_mask`, `train_idx`, `val_idx`, `test_idx`, etc. (from `.mat` pipeline). |
| **Link prediction** | `test_dyads_to_omit`, `val_dyads_to_omit` (OGB); omitted dyads saved as **`omitted_test_dyads.pt`** per split. |

---

## Hyperparameter System

- **Files:** `hypers_link_prediction.yaml`, `hypers_anomaly_unsupervised.yaml`, `hypers_distance.yaml`, `hypers_losses.yaml` (ASCII art section headers are common).

- **Per-dataset keys:** `{dataset}_{model}` (e.g. `citeseer_pieclam`, `elliptic_pclam`).

- **Global defaults:** `GlobalConfigs_bigclam`, `GlobalConfigs_ieclam`, `GlobalConfigs_pclam`, `GlobalConfigs_pieclam` — merged or overridden by dataset-specific triplets.

- **Four nested groups:**

  1. **`clamiter_init`** — `dim_feat`, `dim_attr`, `l1_reg`, `s_reg`, `T`, `hidden_dim`, `num_coupling_blocks`, `num_layers_mlp`, `normalize_degree`, …
  2. **`feat_opt`** — `lr`, `n_iter`, `early_stop`, …
  3. **`prior_opt`** — Prior training: `lr`, `n_iter`, `noise_amp`, `weight_decay`, …
  4. **`back_forth`** — Alternation: `n_back_forth`, `scheduler_*`, `first_func_in_fit` (`fit_feats` vs `fit_prior`), `early_stop_fit`, …

- **`config_triplets`:** `[section, key, value]` overrides applied via `set_config` / `set_multiple_configs` in `trainer*.py`.

---

## Experiments

### Link prediction

- Build graph with **`import_dataset`**; optional **`omit_dyads_random`** to form train/val/test edge masks encoded in **`edge_attr`**.
- Train with **`Trainer`** / **`trainer_directed.Trainer`**, `task='link_prediction'`.
- Metrics: AUC, Hits@K (OGB evaluator where applicable).
- **Storage:** **`SaveRun`** appends to **`acc_configs{HH-MM}_{DD-MM-YY}.json`** (timestamped). Under link prediction: **`experiments/results/directed/.../split_*/{test|valid}/`** (directed) or **`undirected/...`**.

### Anomaly (unsupervised)

- Datasets with attributes + **evaluation** labels; optimization uses **masks**; scoring uses **prior likelihoods** vs **`gt_nomalous`**.
- **`SaveRun`** for `anomaly_unsupervised` uses **`experiments/results/anomaly_unsupervised/{model}/{dataset}/`**.

### Notebooks

- **`link_prediction.ipynb`**, **`anomaly_unsupervised.ipynb`**: primary interactive drivers (large cell outputs / diffs).
- **`distance.ipynb`**: pairs with `hypers_distance.yaml`.
- **Analysis notebooks** under **`results/`**: aggregate JSON.

---

## Current State

**`git log --oneline -20`:** Recent work centers on **directed message passing** (flips, direction fixes), **degree normalization** (sender/receiver degrees, `pow_in`/`pow_out`), **shapes/SBM** experiments, **densification** bugfix, **results/JSON/logs**, merges from **`origin/directed`**, RealNVP / NN param tweaks, SBATCH additions.

**`git status`:** Branch **`directed`**; modified **`clamiter_directed.py`**, **`import_dataset.py`**, large **`link_prediction.ipynb`** / **`anomaly_unsupervised.ipynb`**; untracked new **`acc_configs*.json`** under directed cora-ml/texas.

**Uncommitted diffs (summary):**

- **`clamiter_directed.py`:** Removed erroneous `out_degree = torch.ones(...)` overwrite; replaced single `degree` kwarg with **`deg_in`/`deg_out`/`pow_in`/`pow_out`**; message uses **`deg_ret_j**(-pow_out)*deg_ret_i**(-pow_in)`** instead of a single `**(-0.5)` on one tensor.
- **`import_dataset.py`:** **`transform_attributes`**: sparse-only `nnz` density and `.toarray()` before PCA; dense inputs use **`density_ratio = 1`** and skip `.toarray()`.

**WIP signals:** Degree normalization still has TODOs (`update`, comment on global term scale); notebooks are heavy with outputs; **hidden import** of **`degree`** via `utils.plotting` star-import is fragile.

---

## About Daniel

**Inferred profile (from code, commits, notebooks, and repo structure — not a biography):**

- **Background / level:** Strong **graph modeling** and **optimization** literacy: overlapping communities, BigCLAM-style objectives, **normalizing flows**, **PyTorch Geometric** `MessagePassing`, Lorentz / lightcone featureizations, **einsum**-heavy derivations, OGB link prediction. **Python** is fluent but not “library-purist” (large monolithic files, `import *`, occasional missing explicit imports compensated by side effects). **Research-grade** math notation in comments; **ICML**-era paper codebase.

- **Coding style:** **Single large modules** (`clamiter_*.py` ~1.8k lines) with **parallel** `*_directed` copies rather than heavy abstraction. **ASCII art** in YAML and Python section banners. **First-person / stream-of-consciousness** comments (`#!`, `#?`, `#todo`, `TESTED`, `VARIFIED` typos). **Typos in names** (`gt_nomalous`, `caluclate`) left in place — changing them would break mental map and possibly data.

- **Naming:** Short prefixes (`tbr`, `up`, `ci`), **`printd`** for debug, **`clamiter`** as the MPNN class name, **four-model** naming (`bigclam`, `ieclam`, …).

- **What he cares about:** **Correctness of tricky parts** (asserts on inner product sign) and **empirical sweeps** (JSON configs + accuracies). **Less emphasis** on small-unit tests or CI; **`tests/tests.py`** exists but is peripheral. **Performance** is noted (GPU memory comments) but not micro-optimized everywhere.

- **Workflow:** **Notebook-first** iteration on real experiments (**50k+ line** `link_prediction.ipynb` is a signal of exploratory runs + outputs committed). **Git** messages are informal and diagnostic (“think i found a bug”). **YAML + config_triplets** for systematic hyperparameter deltas. **Cluster** usage (`sbatch`, scripts in `experiments/scripts/`).

- **Collaborating with Claude Code:** Prefer **surgical edits** that match existing patterns; **do not rename** `gt_nomalous` or `clamiter` without explicit ask. **Preserve** dual `trainer`/`trainer_directed` and `optimization_utils` split. When touching **`clamiter_directed`**, remember **`degree`** may come from **`utils.plotting`** star-import. **Ask before** mass-reformatting notebooks or deleting `experiments/results`. He values **honest** assessment of math/bugs over cheerleading.

- **Quirks:** Duplicate logic between `transformation.py` and `transformation_directed.py`; **commented-out alternative** code blocks left in place; **pycache** sometimes tracked or dirty — check `.gitignore` before commits.

---

## Current Goals & Open Questions

**Inferred from branch, commits, and WIP:**

- **Stabilize directed PieClam:** Correct **edge direction**, **flip** semantics, and **two-pass** message passing; recent commits show this was actively debugged.
- **Degree normalization:** Move from ad hoc `0.5` scaling to **explicit sender/receiver** powers (`pow_in`/`pow_out`); `normalize_degree` path still has open TODOs (e.g. global term scale in `update`).
- **Experiments:** **Shapes / SBM** benchmarks added; **directed link prediction** on standard datasets (cora-ml, texas, …) with JSON result accumulation.
- **Data:** **Elliptic** (and dense attribute) pipelines — `transform_attributes` fix for **non-sparse** attrs supports newer loaders.
- **Open:** Whether **degree factors** should apply to **`msg_0`** / omitted dyads; whether **`deg_omitted_*`** tensors in `message` are fully used; cleaning **notebook bloat** vs. keeping run history; potential **explicit import** of `degree` in `clamiter_directed.py` to avoid reliance on `plotting` import side effects.

---
