"""
Re-render the DINO training images (v0) for a generated PHINO dataset.

`scripts/generate_data.py` deletes rendered images by default — renders are a
regenerable cache, while the GPT attribution + trajectories + similarity are the
durable artifacts. Run this before training to recreate the canonical view (v0)
that DINOv2 trains on, reading the asset paths recorded in metadata.json.

Requires an offscreen GL backend (set automatically to egl below) and the
MolmoSpaces assets still present at the `mjcf_path`s recorded in metadata.

Usage:
    python scripts/render_images.py --data_dir data/phino_data
    # after training, optionally reclaim space:
    python scripts/render_images.py --data_dir data/phino_data --delete
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser(description="Re-render PHINO v0 training images")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--delete", action="store_true",
                   help="Instead of rendering, delete the images/ directory.")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    img_dir = data_dir / "images"

    if args.delete:
        shutil.rmtree(img_dir, ignore_errors=True)
        logger.info(f"Deleted {img_dir}")
        return

    from train.data.assets import render_multiview
    from train.data.mujoco_engine import MuJoCoPhysicsEngine
    from train.data.dataset import resolve_image_path

    with open(data_dir / "metadata.json") as f:
        metadata = json.load(f)

    # One render per unique asset (all its physics entries share view0).
    seen, jobs = set(), []
    for e in metadata:
        aid = e["asset_id"]
        if aid in seen:
            continue
        seen.add(aid)
        jobs.append((aid, e["mjcf_path"], e["image_path"]))

    eng = MuJoCoPhysicsEngine()
    img_dir.mkdir(parents=True, exist_ok=True)
    t0, done, skipped = time.time(), 0, 0
    for aid, mjcf_path, image_path in jobs:
        if resolve_image_path(data_dir, image_path) is not None:
            skipped += 1
            continue
        try:
            render_multiview(mjcf_path, str(img_dir), aid,
                             n_views=1, img_size=args.img_size, engine=eng)
            done += 1
        except Exception as e:
            logger.warning(f"Failed to render {aid}: {type(e).__name__}: {e}")
        if (done + skipped) % 100 == 0:
            logger.info(f"  {done+skipped}/{len(jobs)} assets")
    logger.info(f"Rendered {done} assets ({skipped} already present) "
                f"in {time.time()-t0:.1f}s -> {img_dir}")


if __name__ == "__main__":
    main()
