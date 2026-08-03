# PieClam / BigCLAM — Collaborator Guide

This guide covers the **directed** codebase, which is the active development branch. The undirected code (`clamiter.py`, `trainer.py`, `optimization_utils.py`) is older and largely superseded.

---

## What the project does

Learns latent community-affiliation features `x` per node such that edge probability is expressed via an inner product. Used for **link prediction** and **unsupervised anomaly detection**.

**Four model variants** controlled by `vanilla` and `lorenz` flags:

| Name     | vanilla | lorenz |
|----------|---------|--------|
| bigclam  | True    | False  |
| ieclam   | True    | True   |
| pclam    | False   | False  |
| pieclam  | False   | True   |

`vanilla=True` means no normalizing flow prior. `lorenz=True` uses Lorentz/lightcone geometry (features split into space/time halves).

For **directed** graphs, features are additionally split into sender/receiver halves, and two message-passing passes are run (one per direction).

---

## Key files (directed stack)

| File | Role |
|------|------|
| `clamiter_directed.py` | Core `ClamIter` MessagePassing model. All forward/message/update logic. |
| `trainer_directed.py` | Training loop, loads hypers, alternates feature updates and prior training. |
| `transformation_directed.py` | RealNVP normalizing flow prior, `relu_lightcone`, activations. |
| `experiments/optimization_utils_directed.py` | Hyperparameter search, result saving, `cross_val_link`, `multi_ds_anomaly`. |
| `datasets/import_dataset.py` | Loads all datasets (Planetoid, WebKB, OGB, anomaly .mat files, simulations). |
| `hypers/directed/hypers_link_prediction.yaml` | Hyperparameter defaults for link prediction. |
| `hypers/directed/hypers_anomaly_unsupervised.yaml` | Hyperparameter defaults for anomaly detection. |

---

## Data format

All data is a PyTorch Geometric `Data` object. Key fields:

| Field | Meaning |
|-------|---------|
| `x` | Node features `[N, dim_feat]` — learned during training, initialized from attributes or Gaussian. |
| `edge_index` | `[2, E]` — directed edge list. |
| `edge_attr` | `bool`, length `E` — `True` = real edge, `False` = omitted/negative dyad used for training. |
| `y` | Node labels (communities or anomaly labels). |
| `gt_nomalous` | Bool mask for anomaly datasets — `True` = normal, `False` = anomaly (note: spelling is intentional). |
| `attr` | Dense node attributes used for `from_attr` initialization and `attr_opt`. |

---

## Hyperparameter system

Configs are stored in YAML files under `hypers/directed/`. Each dataset+model pair has a section (e.g. `citeseer_bigclam`), plus global defaults (`GlobalConfigs_bigclam`, etc.).

Config overrides are specified as **triplets**: `[section, key, value]`, e.g.:
```python
[['clamiter_init', 'dim_feat', 32], ['feat_opt', 'lr', 5e-6]]
```

The four config groups are:
- `clamiter_init` — model architecture: `dim_feat`, `l1_reg`, `pow_in`, `pow_out`, `init_type`, …
- `feat_opt` — feature optimizer: `lr`, `n_iter`, `early_stop`
- `prior_opt` — prior (flow) optimizer: `lr`, `n_iter`, `noise_amp` (pclam/pieclam only)
- `back_forth` — alternation schedule between feature and prior updates (pclam/pieclam only)

**`pow_in` / `pow_out`**: degree normalization powers. Setting both to `0.0` disables normalization (since d^0 = 1). Small values like `0.05` apply mild degree scaling.

**`init_type`**: how node features are initialized. Options: `'small_gaus'` (random Gaussian), `'from_attr'` (PCA of node attributes — requires attributes in the dataset).

---

## Running link prediction

```python
import sys
sys.path.insert(0, '..')  # from experiments/
import optimization_utils_directed as ou

# full hyperparameter search
ou.cross_val_link(
    ds_name='citeseer',       # dataset name (matches import_dataset)
    model_name='bigclam',     # bigclam / pclam / ieclam / pieclam
    n_reps=2,                 # repetitions per config
    use_global_config_base=False,
    device='cpu',
    range_triplets=[
        ['clamiter_init', 'dim_feat', [32, 48]],
        ['clamiter_init', 'pow_in',   [0.05, 0.1]],
        ['clamiter_init', 'pow_out',  [0.001, 0.005]],
        ['feat_opt',      'n_iter',   [5000]],
        ['feat_opt',      'lr',       [5e-6]],
    ],
    metric='auc',
    val_p=0.1,        # fraction of edges held out for validation
    test_p=0.0,       # fraction held out for test (0 = val-only mode)
    to_undirected=False,
    remove_self_loops=False,
    acc_every=10,
    plot_every=-1,
)
```

Results are saved to `experiments/results/directed/link_prediction/auc/{ds_name}/{model_name}/split_{timestamp}/valid/acc_configs{timestamp}.json`.

### Continuing from existing results

```python
# run the top 5 configs from a previous search (10 more reps each)
ou.cross_val_link(
    ...,
    from_file='results/directed/link_prediction/auc/citeseer/bigclam/split_.../valid/acc_configs....json',
    from_file_mode='top',       # 'top' or 'remaining'
    from_file_top_n=5,
    from_file_sort_by='val',    # sort by val acc (use 'test' for anomaly)
    from_file_agg='mean',       # 'mean' (matches print_folder) or 'best'
    n_reps=10,
)
```

### Inspecting results

```python
ou.print_folder(
    ds_name='citeseer',
    model_name='bigclam',
    metric='auc',
    test_or_valid='valid',
    task='link_prediction',
    sort_by='val_acc',
    from_date='1-4-26',         # only show files after this date (DD-M-YY)
    show_empty_files=False,
)

# extract top configs programmatically
config_ranges, config_list, scores = ou.top_configs_from_file(
    'results/directed/link_prediction/.../acc_configs....json',
    top_n=5,
    sort_by='val',   # 'val' for link prediction, 'test' for anomaly
    agg='mean',
    return_scores=True,
)
```

---

## Running anomaly detection

```python
ou.multi_ds_anomaly(
    model_name='bigclam',
    ds_names=['disney', 'books'],
    range_triplets=[
        ['clamiter_init', 'dim_feat', [8, 16, 32]],
        ['clamiter_init', 'pow_in',   [0.01, 0.05, 0.1]],
        ['clamiter_init', 'pow_out',  [0.01, 0.05, 0.1]],
        ['feat_opt',      'n_iter',   [5000]],
        ['feat_opt',      'lr',       [1e-4]],
    ],
    metric='auc',
    attr_opt=True,
    n_reps=1,
)

# or continue from existing file
ou.multi_ds_anomaly(
    model_name='bigclam',
    ds_names=['disney', 'books'],
    from_files=['acc_configs....json', 'acc_configs....json'],
    from_files_mode='top',
    from_files_top_n=5,
)
```

Results go to `experiments/results/directed/anomaly_unsupervised/auc/{ds_name}/{model_name}/acc_configs{timestamp}.json`.

---

## Results format

Each JSON file contains:
- `config_ranges`: the search grid definition
- `base_config`: the YAML defaults that were active
- One entry per run: key = `str((test_acc, val_acc))`, value = list of config triplets

`print_folder` aggregates all JSON files in a directory, groups by config, and shows mean ± std sorted by the metric of interest.

---

## Notebooks

### `experiments/link_prediction.ipynb`
The main interactive workspace for link prediction. Use this to:
- Run `cross_val_link` for a broad hyperparameter search
- Inspect loss/accuracy curves during training (`plot_every` controls frequency)
- Run top configs from existing result files (`from_file_mode='top'`)
- Test specific configs manually

Typical workflow: broad search with `n_reps=2` → look at curves → pick promising region → run top configs with more reps.

### `experiments/anomaly_unsupervised.ipynb`
Same as above but for anomaly detection. Uses `multi_ds_anomaly` to run multiple datasets in one call. The anomaly score is the prior likelihood (higher = more normal).

### `experiments/results/directed/link_prediction/analysis.ipynb`
**Result aggregation notebook.** Does not run training — only reads and displays JSON result files. Use `ou.print_folder(...)` here to see ranked configs across all runs for a given dataset/model. Also used to extract top configs for the next round.

### `experiments/results/directed/anomaly_unsupervised/analysis.ipynb`
Same as above for anomaly results. Also contains baseline comparisons (PyGOD detectors).

---

## Folder structure

```
ICML_PieClam_020425/
├── clamiter_directed.py          # core model
├── trainer_directed.py           # training loop
├── transformation_directed.py    # normalizing flow prior
├── hypers/
│   └── directed/
│       ├── hypers_link_prediction.yaml
│       └── hypers_anomaly_unsupervised.yaml
├── datasets/
│   └── import_dataset.py
└── experiments/
    ├── optimization_utils_directed.py   # search functions, SaveRun
    ├── link_prediction.ipynb            # interactive link prediction workspace
    ├── anomaly_unsupervised.ipynb       # interactive anomaly workspace
    ├── results/
    │   └── directed/
    │       ├── link_prediction/
    │       │   └── auc/
    │       │       └── {ds_name}/
    │       │           └── {model_name}/
    │       │               └── split_{HH-MM_DD-MM-YY}/   ← one folder per test split
    │       │                   ├── omitted_test_dyads.pt  ← saved test edges
    │       │                   └── valid/
    │       │                       └── acc_configs{HH-MM_DD-MM-YY}.json
    │       ├── anomaly_unsupervised/
    │       │   └── auc/
    │       │       └── {ds_name}/
    │       │           └── {model_name}/
    │       │               └── acc_configs{HH-MM_DD-MM-YY}.json
    │       └── */analysis.ipynb         ← result aggregation notebooks
    └── scripts/
        ├── link_prediction/
        │   └── hypers_search/
        │       └── hypers_search_splits.py   ← CLI script for cluster runs
        └── anomaly_unsupervised/
            └── directed/
                ├── hypers_search.py          ← CLI script for anomaly search
                └── anomaly_test.sbatch       ← SLURM job submission
```

**Split folders** (`split_{timestamp}`): created the first time a test split is used. If you re-run with the same test edges, the existing split folder is reused — results accumulate in the same directory.

---

## Scripts (for cluster / CLI)

### `experiments/scripts/link_prediction/hypers_search/hypers_search_splits.py`

CLI wrapper around `cross_val_link_splits`. Run from the project root or via SLURM:

```bash
python experiments/scripts/link_prediction/hypers_search/hypers_search_splits.py \
    --model_name bigclam \
    --ds_name citeseer \
    --dim_feats 32 48 \
    --n_iters_feats 5000 \
    --lr_feats 5e-6 \
    --test_p 0.1 \
    --val_p 0.0 \
    --test_only \
    --n_reps 2
```

Key flags: `--model_name`, `--ds_name`, `--dim_feats`, `--lr_feats`, `--n_iters_feats`, `--test_p`, `--val_p`, `--n_reps`, `--random_search`, `--num_draws_random`, `--to_undirected`, `--reverse_test_set_order`.

For pclam/pieclam, also: `--n_iters_prior`, `--lr_prior`, `--noise_amps`, `--num_coupling_blocks`, `--num_layers_mlp`, `--hidden_dim`, `--n_back_forth`.

### `experiments/scripts/anomaly_unsupervised/directed/hypers_search.py`

CLI wrapper around `multi_ds_anomaly`. Same argument style:

```bash
python experiments/scripts/anomaly_unsupervised/directed/hypers_search.py \
    --model_name bigclam \
    --ds_name disney \
    --dim_feats 8 16 32 \
    --lr_feats 1e-4 \
    --pow_in 0.05 \
    --pow_out 0.05 \
    --n_iters_feats 5000
```

### `experiments/scripts/anomaly_unsupervised/directed/anomaly_test.sbatch`

SLURM submission file for running the anomaly script on a GPU cluster. Edit the `#SBATCH` headers and the python call at the bottom to match your job.

---

## Datasets

Loaded via `import_dataset(name, ...)`. Common names:

| Name | Type | Notes |
|------|------|-------|
| `citeseer`, `cora-ml`, `texas`, `cornell` | Link prediction | Directed graphs |
| `disney`, `books`, `amazon` | Anomaly | Small, from PyGOD |
| `elliptic` | Anomaly | Large (200k nodes), Bitcoin transactions |
| `photo`, `reddit` | Anomaly | From PYGOD / mat files |

Setting `to_undirected=False` keeps the graph directed (use for directed models). `remove_self_loops=False` preserves self-loops where present.
