#!/usr/bin/env python
"""Non-invasive launcher for stable-worldmodel's DINO-WM planning eval (eval_wm.py).

eval_wm.py was written against datasets whose `column_names` / `get_row_data`
expose the episode/step index columns. The Lance reader exposes only data
columns (action/pixels/proprio/state) — though `get_col_data('episode_idx')`
and `('step_idx')` work fine. We monkeypatch the Lance reader to surface those
two index columns, then run the UNMODIFIED eval_wm.py via runpy. The DINO-WM /
eval source is left untouched.

Usage:
    python eval_dinowm.py policy=<run>/weights_epoch_N.pt \
        eval.dataset_name=galilai-group--lewm-pusht/pusht_expert_train.lance \
        eval.num_eval=50 [hydra overrides...]
"""

import runpy

import numpy as np

import stable_worldmodel  # noqa: F401  (registers formats)
from stable_worldmodel.data.formats.lance import LanceDataset

# NOTE: we deliberately do NOT touch `column_names`. eval_wm.py's index-column
# heuristic is `'episode_idx' if 'episode_idx' in column_names else 'ep_idx'`;
# since the lance reader lists only data columns it resolves to 'ep_idx'. We
# make 'ep_idx' a working alias for the real 'episode_idx' field in get_col_data
# / get_row_data. Leaving column_names untouched keeps world._extract_init_goal
# (which iterates column_names over episode slices) working on data columns only.

# 1) get_col_data: alias 'ep_idx' -> 'episode_idx'.
_orig_gcd = LanceDataset.get_col_data


def _get_col_data(self, col):
    return _orig_gcd(self, "episode_idx" if col == "ep_idx" else col)


LanceDataset.get_col_data = _get_col_data

# 2) get_row_data: add the index columns eval_wm.py reads off sampled rows.
_orig_rows = LanceDataset.get_row_data


def _get_row_data(self, row_idx):
    out = _orig_rows(self, row_idx)
    idxs = np.asarray(
        row_idx
        if isinstance(row_idx, (list, tuple, np.ndarray))
        else [row_idx],
        dtype=int,
    )
    # Always override ep_idx/step_idx with true integer index columns.
    # Some datasets (e.g. ogbcube) store ep_idx as a float32 data column;
    # _load_slice uses it as an array index and requires a plain integer.
    out["ep_idx"] = np.asarray(_orig_gcd(self, "episode_idx"), dtype=np.int64)[idxs]
    out["step_idx"] = np.asarray(_orig_gcd(self, "step_idx"), dtype=np.int64)[idxs]
    return out


LanceDataset.get_row_data = _get_row_data

EVAL_WM = "/home/colligo/phino/stable-worldmodel/scripts/plan/eval_wm.py"

if __name__ == "__main__":
    runpy.run_path(EVAL_WM, run_name="__main__")
