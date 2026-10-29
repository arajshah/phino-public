"""Object-disjoint dataset splits for physical-property probes."""

from collections.abc import Sequence

import numpy as np


def split_metadata_by_asset(
    metadata: Sequence[dict], split: str, seed: int = 42,
) -> list[dict]:
    """Partition metadata 70/15/15 by asset_id, not physics-variant rows.

    Note: A probe split does not also hold objects out of encoder pretraining.
    """
    if split not in ("train", "val", "test"):
        raise ValueError(f"Unsupported split: {split}")

    if any(not entry.get("asset_id") for entry in metadata):
        raise ValueError("Each entry must have a nonempty asset_id.")

    asset_ids = list(dict.fromkeys(entry["asset_id"] for entry in metadata))
    count = len(asset_ids)
    if count < 3:
        raise ValueError("Three or more unique assets are required.")

    order = np.random.default_rng(seed).permutation(count)
    train_end = min(count - 2, max(1, int(0.70 * count)))
    val_end = train_end + min(count - train_end - 1, max(1, int(0.15 * count)))

    slices = {
        "train": order[:train_end],
        "val": order[train_end:val_end],
        "test": order[val_end:],
    }
    members = {asset_ids[index] for index in slices[split]}
    return [entry for entry in metadata if entry["asset_id"] in members]
