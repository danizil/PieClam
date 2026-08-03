#!/usr/bin/env python3
"""Runner script for cross_val_link — called by the lp-search agent as a background subprocess.

Usage:
    python run_cross_val_link.py --config /path/to/config.json

The config JSON has the same keys as cross_val_link kwargs, e.g.:
{
    "ds_name": "wisconsin",
    "model_name": "bigclam",
    "n_reps": 2,
    "val_p": 0.1,
    "test_p": 0.0,
    "to_undirected": false,
    "remove_self_loops": false,
    "name": "from_attr",
    "range_triplets": [
        ["clamiter_init", "dim_feat", [24, 32, 40]],
        ["clamiter_init", "init_type", ["from_attr"]],
        ["clamiter_init", "pow_in",  [0.0, 0.001, 0.01, 0.05, 0.1]],
        ["clamiter_init", "pow_out", [0.0, 0.001, 0.01, 0.05, 0.1]],
        ["feat_opt", "n_iter", [5000, 7000]],
        ["feat_opt", "lr",     [1e-6, 5e-6, 1e-5]]
    ]
}
"""

import sys, os, json, argparse

script_dir = os.path.dirname(os.path.realpath(__file__))
for _ in range(4):
    script_dir = os.path.dirname(script_dir)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

os.chdir(os.path.join(script_dir, 'experiments'))

import torch
import experiments.optimization_utils_directed as ou

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to JSON config file')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}', flush=True)

    ou.cross_val_link(
        use_global_config_base=False,
        device=device,
        metric='auc',
        acc_every=cfg.pop('acc_every', 20),
        plot_every=cfg.pop('plot_every', -1),
        **cfg
    )

if __name__ == '__main__':
    main()
