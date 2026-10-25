---
name: pieclam-sweep
description: Run PieClam/CLAM hyperparameter sweeps (broad search then refine-to-10-reps) for link prediction or anomaly detection on directed graphs in this codebase. Use when the user asks to run/resume/queue a hyperparameter search, broad search, refinement, or sweep for bigclam/pclam/ieclam/pieclam on any dataset, or to check/report on results from such a sweep.
---

# PieClam hyperparameter sweeps

This codebase (directed-branch PieClam/CLAM) has two sweep-driving functions:

- **`experiments/optimization_utils_directed.py::cross_val_link`** — link prediction, one dataset per call.
- **`experiments/optimization_utils_directed.py::multi_ds_anomaly`** — anomaly detection, pass `ds_names=[single_dataset]` to treat it the same way (even though it supports multiple datasets per call, this codebase's convention is one at a time).

Both follow the same **two-phase pattern** and share the same results-file machinery (`SaveRun`, `top_configs_from_file`, `remaining_configs_from_file`, `_count_config_reps`). Everything below applies to both unless stated otherwise.

## The two-phase pattern

1. **Broad search**: sweep a grid (`range_triplets`) with `n_reps=2`, `random_search=True`, and `num_draws_random` set to the full grid size (so it's shuffled but not truncated — both functions support this parameter). This produces a new results JSON.
2. **Refine**: take the top N configs from that file and push them to `n_reps=10, max_reps=10` via `from_file=<broad file>, from_file_mode='top', from_file_top_n=20` (adjust N to the grid size — 20 is too many for a 24-config grid, use ~10 there).

To **resume** an interrupted broad search without re-testing finished configs, relaunch with `from_file=<same file>, from_file_mode='remaining'` instead of passing `range_triplets` again. `remaining_configs_from_file` computes the full grid from the file's own `config_ranges` and returns only configs completed fewer than `n_reps` times.

`from_file_mode='top'` (refine) is idempotent to resume too — just rerun the exact same command; `max_reps` + `_count_config_reps` skip whatever's already done.

## Launch discipline

- **Always smoke-test before a real launch**: run the script in the foreground, `sleep 20-25`, check for a traceback, `kill` it, *then* launch for real in the background. New datasets/grids can hit things a smoke test catches immediately (missing yaml block, bad split file, dim mismatch) that would otherwise burn hours silently.
- **Background launches**: `nohup <cmd> > log 2>&1 & disown`, capture the PID, verify it's still alive and error-free ~15-20s later.
- **Respect the user's CPU cap.** Ask if unstated. Set `OMP_NUM_THREADS` and `MKL_NUM_THREADS` per job so `(threads per job) × (concurrent jobs) ≤ cap`. `timeout` is **not available on macOS** — for a wall-clock safety net use a background watchdog instead: `( sleep 86400; kill $PID ) & disown`.
- **Queueing beyond the cap**: write a small watcher shell script that polls `kill -0 $PID` in a loop, and on exit launches the next queued job — chain multiple sequentially in one script if a slot should run several jobs back to back. Always check `$?` after each queued command and log failures distinctly from successes (a crashed job that isn't checked will silently look identical to a finished one).

## Headless plotting (no GUI popups from background jobs)

`plot_every=-1` does **not** disable plotting — it makes the code plot once at the very end of every single training run (see `clamiter_directed.py` around the `plot_every == -1` handling). Combined with a GUI matplotlib backend, this pops a window on every run. Fix: set these env vars on any background launch —

```bash
PIECLAM_HEADLESS=1 PIECLAM_PLOT_PDF=/path/to/loss_sheet.pdf
```

`utils/plotting.py` checks `PIECLAM_HEADLESS` live (no restart needed even mid-session) and, when set, appends each run's diagnostic plot as a new page onto the single PDF at `PIECLAM_PLOT_PDF` (default `/tmp/lp_search_agent/loss_sheet.pdf`), deleting the intermediate file immediately — nothing else accumulates on disk, and the PDF stays valid even if the job is killed mid-run (unlike a naive single open `PdfPages`, which stays corrupt/unreadable until `.close()`, and `.close()` never fires on a plain `kill` since Python doesn't run `atexit` on SIGTERM).

## Config gotchas

- **`use_global_config_base`**: `False` uses the dataset's own `{ds}_{model}` yaml block (`hypers/directed/hypers_{task}.yaml`); `True` falls back to `GlobalConfigs_{model}`. If a dataset has no dedicated block, `use_global_config_base=False` raises `KeyError`. If you find yourself relying on `True` as a workaround, that's a signal to add a dedicated block instead (see below).
- **Config layering**: `Trainer.configs_dict_from_top_list` merges `GlobalConfigs_{model}` → dataset's yaml block → swept `range_triplets`, in that order. `GlobalConfigs_bigclam` defaults `init_type: from_attr`. So a sweep that never explicitly varies `init_type` still *runs* at `from_attr` — it just won't be *recorded* in that entry's config triplets, which will silently poison any analysis that filters by recorded `init_type`. **Always include `['clamiter_init', 'init_type', [...]]` explicitly in `range_triplets`** to avoid this ambiguity.
- **`from_attr` needs enough attributes**: requires `dim_feat // 2` (directed) or `dim_feat` (undirected) ≤ the dataset's actual attribute count. Both sweep functions now pre-check this and skip+log (`printd('Skipping config: from_attr needs N components but only M attributes available...')`) rather than crashing the whole sweep — but a grid with `dim_feat` values too large for a small-attribute dataset will just silently lose coverage at the high end. Check the dataset's `raw_attr`/`attr` shape before picking a `dim_feat` range for `from_attr`.
- **Don't reuse an old split's `test_dyads_path` blindly.** Old `omitted_test_dyads.pt` files are sometimes empty/wrong-dtype (`float32` instead of `int64`), and feeding that through `lp.omit_dyads` corrupts `edge_index`'s dtype downstream, crashing `degree()` deep in `clamiter_directed.forward`. Safest default: omit `test_dyads_path` entirely and let `SaveRun` auto-reuse a matching existing split (it compares shapes/values) or create a fresh one.
- **Adding a dedicated yaml block for a dataset**: match the minimal per-dataset style already used (`hypers/directed/hypers_link_prediction.yaml` / `hypers_anomaly_unsupervised.yaml`) — just `clamiter_init: {dim_feat, l1_reg, pow_in, pow_out}` and `feat_opt: {lr, n_iter}`, populated from the best config found so far. Don't add unused keys (`dim_attr`, `T`, `hidden_dim`, etc. only matter for prior-based models).

## Checking progress / building a results table

```python
import optimization_utils_directed as ou

# top configs in one file, by mean val AUC
config_ranges, config_list, scores = ou.top_configs_from_file(
    file_path, top_n=5, sort_by='val', agg='mean', return_scores=True)

# rep counts per config in a file
counts = ou._count_config_reps(file_path)  # {config_values_tuple: n_reps}
```

Results are keyed as `"{acc}__{unique suffix}"` (not just `str(acc)`) so identical accuracies don't collide and overwrite each other — use `ou._parse_acc_key(k)` before `eval()`-ing a raw key yourself; it's backward compatible with older files that used bare `str(acc)` keys.

**A dataset/init_type can have results scattered across many files in the same split folder** (different exploratory runs over time). To get the true rep count for a config, aggregate across every `acc_configs*.json` in the split's `valid/` folder, not just the most recent file — group by the config-values tuple, sum counts. Only trust a config as "done" once the *aggregate* reaches the target rep count.

**Reporting convention**: only report configs with ≥10 reps as final results. If a config has >14 reps (over-sampled from repeated exploratory runs), report the **best 10 of N** (sorted descending, mean of top 10) rather than the mean of all N — this was the user's explicit convention for this project.

## Reference: existing single-dataset agent

`.Codex/agents/lp-search.md` is a narrower subagent for a single link-prediction dataset (broad search → threshold check → refine). It hardcodes `init_type: small_gaus` as of 2026-08-09 (see project memory `feedback_init_type.md` if available) — check whether that's still current policy before trusting it, since this codebase has flipped the default init_type convention at least once already.
