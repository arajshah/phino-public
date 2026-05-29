"""
PHINO smoke test — produce inspectable artifacts for a few assets.

For each of N MolmoSpaces assets:
  - render the multi-view images (kept, not deleted)
  - GPT-4o (or mock) physical attribution -> attribution/<asset_id>.json
  - sample ONE physics config from the attribution
  - for each of the 5 diagnostic actions: simulate in MuJoCo, save
      * a video  (videos/<asset_id>__<action>.mp4)
      * a 6-frame contact sheet (videos/<asset_id>__<action>.png) for quick viewing
      * the trajectory (trajectories/<asset_id>__<action>.npz)
  - write summary.json

Usage:
    # mock (no key):
    python scripts/smoke_test.py --asset_root data/molmospaces_thor/assets --n_assets 3
    # live GPT-4o (Azure/OpenAI creds in env): just omit --mock
    AZURE_OPENAI_API_KEY=... python scripts/smoke_test.py --asset_root ... --n_assets 3
"""

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)


def contact_sheet(frames, n=6):
    """Tile n evenly-spaced frames horizontally into one image for quick viewing."""
    idx = np.linspace(0, len(frames) - 1, n).astype(int)
    return np.concatenate([frames[i] for i in idx], axis=1)


def main():
    p = argparse.ArgumentParser(description="PHINO smoke test")
    p.add_argument("--asset_root", type=str, required=True)
    p.add_argument("--n_assets", type=int, default=3)
    p.add_argument("--n_views", type=int, default=4)
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--mock", action="store_true", help="Force offline mock attribution.")
    p.add_argument("--output_dir", type=str, default="data/smoke")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    import imageio.v3 as iio
    from train.data.assets import discover_assets, render_multiview
    from train.data.attribution import PhysicalAttributor
    from train.data.mujoco_engine import MuJoCoPhysicsEngine, DIAGNOSTIC_ACTIONS
    from train.data.generation import PhysicsConfig

    out = Path(args.output_dir)
    if out.exists():
        shutil.rmtree(out)   # fresh run (so attribution cache isn't stale mock)
    for sub in ("images", "videos", "trajectories"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    eng = MuJoCoPhysicsEngine()
    attributor = PhysicalAttributor(store_path=str(out / "attributions.json"),
                                    backend="mock" if args.mock else "auto", seed=args.seed)
    logger.info(f"Attribution backend: {attributor.backend} ({attributor.provider})")

    recs = discover_assets(args.asset_root, max_assets=args.n_assets)
    summary = {"backend": f"{attributor.backend}:{attributor.provider}", "assets": []}

    for rec in recs:
        logger.info(f"--- {rec.asset_id} ---")
        views = render_multiview(rec.mjcf_path, str(out / "images"), rec.asset_id,
                                 n_views=args.n_views, img_size=args.img_size, engine=eng)
        mass_hint = eng.build_scene(rec.mjcf_path)[0].body_subtreemass[1:].sum()
        attr = attributor.attribute(rec.asset_id, image_paths=views, mass_hint=float(mass_hint))
        physics = PhysicsConfig.sample_from_attribution(attr, rng, category=rec.category)
        logger.info(f"  attribution backend={attr.backend} mass~{attr.mass_kg.mean:.2f}kg "
                    f"| sampled mass={physics.mass:.2f} fric={physics.static_friction:.2f} "
                    f"rest={physics.restitution:.2f}")

        per_action = {}
        for action in DIAGNOSTIC_ACTIONS:
            traj, frames = eng.simulate_action_render(rec.mjcf_path, physics, action,
                                                      img_size=args.img_size)
            stem = f"{rec.asset_id}__{action.name}"
            iio.imwrite(out / "videos" / f"{stem}.mp4", np.stack(frames), fps=args.fps)
            iio.imwrite(out / "videos" / f"{stem}.png", contact_sheet(frames))
            np.savez_compressed(out / "trajectories" / f"{stem}.npz", trajectory=traj)
            per_action[action.name] = {
                "max_disp_m": float(np.abs(traj[:, :3]).max()),
                "final_speed": float(np.linalg.norm(traj[-1, 7:10])),
            }
        summary["assets"].append({
            "asset_id": rec.asset_id, "category": rec.category,
            "attribution": attr.to_dict(),
            "sampled_physics": {"mass": physics.mass,
                                "static_friction": physics.static_friction,
                                "restitution": physics.restitution},
            "per_action": per_action,
        })

    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"=== Smoke test done -> {out} ===")
    logger.info(f"  images/      multi-view renders ({args.n_views}/asset)")
    logger.info(f"  videos/      <asset>__<action>.mp4  + .png contact sheets")
    logger.info(f"  trajectories/ <asset>__<action>.npz")
    logger.info(f"  attribution/ <asset>.json   |   summary.json")


if __name__ == "__main__":
    main()
