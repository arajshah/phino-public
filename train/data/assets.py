"""
MolmoSpaces asset I/O + rendering.

MolmoSpaces (allenai/molmospaces on HuggingFace) ships MuJoCo object assets in a
packed format: a per-subset parquet index (`path, shard_id, offset, size`) plus
one or more shard `.tar` files. Each indexed entry is a `.tar.zst` blob sitting
at a byte offset inside a shard, so we can fetch just the bytes we need with an
HTTP range request, zstd-decompress, and untar into a local asset directory.

For THOR the object archives are grouped by category bundle (e.g.
"Miscellaneous Objects"); each bundle inflates to
    <Category>/<Class>/Prefabs/<Instance>/<Instance>.xml   (+ meshes, textures)

This module:
  - `download_molmospaces_bundle(...)` : range-fetch + extract one bundle.
  - `discover_assets(root)`            : find the main MJCF per object instance.
  - `render_asset(mjcf, png)`          : offscreen render one canonical view
                                         (the DINO training image + GPT-4o input).
"""

from __future__ import annotations

import io
import logging
import os
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

REPO = "allenai/molmospaces"
SUBSET_BASE = {
    "thor": "mujoco/objects/thor/20251117",
    "objaverse": "mujoco/objects/objaverse/20260131",
}
# Default small THOR bundle (~12 MB) — good for a quick run.
DEFAULT_THOR_BUNDLE = "thor_Miscellaneous Objects.tar.zst"


@dataclass
class AssetRecord:
    asset_id: str
    mjcf_path: str
    category: str = "object"
    metadata: dict = field(default_factory=dict)


# ==========================================================================
# Download / extract (packed shard + offset scheme)
# ==========================================================================
def download_molmospaces_bundle(
    dest_dir: str,
    subset: str = "thor",
    bundle: Optional[str] = None,
) -> Path:
    """Download + extract one MolmoSpaces object bundle into `dest_dir`.

    Returns the directory the bundle was extracted into. Idempotent: if a
    `.done` marker exists the download/extract is skipped.
    """
    import pandas as pd
    import requests
    import zstandard
    from huggingface_hub import hf_hub_download, hf_hub_url

    if subset not in SUBSET_BASE:
        raise ValueError(f"Unknown subset '{subset}'. Options: {list(SUBSET_BASE)}")
    base = SUBSET_BASE[subset]
    bundle = bundle or (DEFAULT_THOR_BUNDLE if subset == "thor" else None)
    if bundle is None:
        raise ValueError(f"A bundle name is required for subset '{subset}'")

    dest = Path(dest_dir)
    marker = dest / f".done_{_safe(bundle)}"
    if marker.exists():
        logger.info(f"Bundle already present: {bundle}")
        return dest
    dest.mkdir(parents=True, exist_ok=True)

    # 1. parquet index (small)
    pq = hf_hub_download(REPO, f"{base}/pkgs-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(pq)
    rows = df[df["path"] == bundle]
    if len(rows) == 0:
        raise ValueError(
            f"Bundle '{bundle}' not found in {subset}. Available examples: "
            f"{list(df['path'].head(20))}"
        )
    row = rows.iloc[0]
    shard_id, offset, size = int(row["shard_id"]), int(row["offset"]), int(row["size"])

    # 2. range-fetch the blob bytes from the shard
    url = hf_hub_url(REPO, f"{base}/shards/{shard_id:05d}.tar", repo_type="dataset")
    logger.info(f"Range-fetching {size/1e6:.1f} MB for '{bundle}' from shard {shard_id}")
    headers = {"Range": f"bytes={offset}-{offset + size - 1}"}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = requests.get(url, headers=headers, timeout=600)
    resp.raise_for_status()
    blob = resp.content

    # 3. zstd-decompress + untar
    raw = zstandard.ZstdDecompressor().decompress(blob, max_output_size=2 * 1024 ** 3)
    with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
        tf.extractall(dest)
    n_xml = len(list(dest.rglob("*.xml")))
    logger.info(f"Extracted '{bundle}' -> {dest} ({n_xml} xml files)")
    marker.write_text("ok")
    return dest


# ==========================================================================
# Discovery
# ==========================================================================
_VARIANT_SUFFIXES = ("_mesh.xml", "_old.xml", "_prim.xml", "_collision.xml")


def discover_assets(root: str, max_assets: Optional[int] = None) -> List[AssetRecord]:
    """Find object MJCFs under a MolmoSpaces asset tree.

    Picks the main `<Instance>.xml` inside each `.../Prefabs/<Instance>/`,
    skipping the `_mesh / _old / _prim` variants.
    """
    root_path = Path(root)
    records: List[AssetRecord] = []
    seen = set()
    for xml in sorted(root_path.rglob("*.xml")):
        name = xml.name
        if any(name.endswith(s) for s in _VARIANT_SUFFIXES):
            continue
        # main MJCF lives in a dir named after the instance (Prefabs/<Inst>/<Inst>.xml)
        if xml.parent.name != xml.stem:
            continue
        asset_id = xml.stem
        if asset_id in seen:
            continue
        seen.add(asset_id)
        category = _category_from_path(xml, root_path)
        metadata = _load_metadata(xml)
        records.append(AssetRecord(asset_id=asset_id, mjcf_path=str(xml),
                                   category=category, metadata=metadata))
        if max_assets and len(records) >= max_assets:
            break
    logger.info(f"Discovered {len(records)} assets under {root}")
    return records


def _category_from_path(xml: Path, root: Path) -> str:
    """Derive a coarse category from the `<Category>/<Class>/Prefabs/...` layout."""
    parts = xml.relative_to(root).parts
    if "Prefabs" in parts:
        i = parts.index("Prefabs")
        if i >= 1:
            return parts[i - 1]  # the object Class (e.g. "TeddyBear")
    return parts[0] if parts else "object"


def _load_metadata(xml: Path) -> dict:
    """Load the sibling `<Instance>.json` (THOR) if present (bbox/scale/meshes)."""
    import json
    cand = xml.parent.parent / f"{xml.stem}.json"
    if cand.exists():
        try:
            return json.loads(cand.read_text())
        except Exception:
            return {}
    return {}


# ==========================================================================
# Rendering
# ==========================================================================
def render_multiview(
    mjcf_path: str,
    out_dir: str,
    asset_id: str,
    n_views: int = 4,
    img_size: int = 224,
    base_azimuth: float = 120.0,
    engine=None,
) -> List[str]:
    """Render `n_views` settled views of an asset from evenly-spaced azimuths.

    Views orbit the object at alternating low/high elevations so GPT-4o can
    judge material, size and hollowness from several angles. View 0 is the
    canonical view used as the DINO training image. Uses the same scene
    composition (asset + plane + light) as the physics engine, so the image
    matches what is simulated. Requires an offscreen GL backend (MUJOCO_GL=egl).
    """
    import mujoco
    import imageio.v3 as iio

    from train.data.mujoco_engine import MuJoCoPhysicsEngine
    eng = engine or MuJoCoPhysicsEngine()

    model, data, info = eng.build_scene(mjcf_path)
    # settle so the object rests on the plane
    mujoco.mj_resetData(model, data)
    data.qpos[info["qadr"] + 2] = eng.spawn_height
    for _ in range(eng.settle_steps):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)

    # frame the camera on the object itself (not the whole scene, which includes
    # the large ground plane and would zoom the object out to a speck).
    bid = info["body_id"]
    center = data.xpos[bid].copy()
    plane_gid = info["plane_gid"]
    obj_radius = 0.0
    for g in range(model.ngeom):
        if g == plane_gid:
            continue
        d_center = float(np.linalg.norm(data.geom_xpos[g] - center))
        obj_radius = max(obj_radius, d_center + float(model.geom_rbound[g]))
    obj_radius = obj_radius or 0.1
    distance = max(0.15, 2.2 * obj_radius)

    out_paths: List[str] = []
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, height=img_size, width=img_size)
    try:
        for k in range(max(1, n_views)):
            cam = mujoco.MjvCamera()
            cam.lookat[:] = center
            cam.distance = distance
            cam.azimuth = base_azimuth + k * (360.0 / max(1, n_views))
            cam.elevation = -20.0 if k % 2 == 0 else -45.0
            renderer.update_scene(data, camera=cam)
            img = renderer.render()
            png = str(Path(out_dir) / f"{_safe(asset_id)}_v{k}.png")
            iio.imwrite(png, img)
            out_paths.append(png)
    finally:
        renderer.close()
    return out_paths


def render_asset(
    mjcf_path: str,
    out_png: str,
    img_size: int = 224,
    engine=None,
) -> str:
    """Render a single canonical view (thin wrapper over render_multiview)."""
    out = Path(out_png)
    paths = render_multiview(mjcf_path, str(out.parent), out.stem,
                             n_views=1, img_size=img_size, engine=engine)
    # render_multiview names the file "<stem>_v0.png"; rename to the requested name
    src = Path(paths[0])
    if src != out:
        src.replace(out)
    return str(out)


def _safe(name: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)
