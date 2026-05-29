"""
Generate PHINO training data (GPT-4o attribution + MuJoCo trajectories).

For each MolmoSpaces object asset: render a canonical image, ask GPT-4o (or the
offline mock) to predict mass/friction/restitution ranges, sample N physics
configs, simulate the 5 diagnostic actions in MuJoCo, and precompute pairwise
trajectory-MSE similarities used as soft contrastive labels.

Assets are downloaded on demand from the MolmoSpaces HuggingFace dataset
(allenai/molmospaces). Requires an offscreen GL backend for rendering — set
MUJOCO_GL=egl (done automatically below if unset).

Output layout:
    {output_dir}/
        images/                # rendered asset views (DINO input)
        attribution/           # cached GPT-4o predictions (one json per asset)
        fingerprints/          # one .npz trajectory per (asset, physics)
        metadata.json
        similarity_matrix.npz
        invisible_physics_test.json

Examples:
    # offline smoke run on a small THOR bundle (no API key needed)
    python scripts/generate_data.py --download_subset thor \\
        --max_assets 8 --num_physics 3 --num_pairs 500 --mock \\
        --output_dir data/phino_smoke

    # full run on already-downloaded assets, live GPT-4o (OPENAI_API_KEY set)
    python scripts/generate_data.py --asset_root /data/molmospaces_thor \\
        --max_assets 2000 --num_physics 5 --output_dir data/phino_data
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser(description="Generate PHINO training data")
    p.add_argument("--output_dir", type=str, default="data/phino_data")
    p.add_argument("--asset_root", type=str, default=None,
                   help="Directory of already-extracted MolmoSpaces assets. "
                        "If omitted, a bundle is downloaded via --download_subset.")
    p.add_argument("--download_subset", type=str, default=None,
                   choices=["thor", "objaverse"],
                   help="Download+extract a MolmoSpaces object bundle to <output>/assets.")
    p.add_argument("--bundle", type=str, default=None,
                   help="Specific bundle archive name (default: a small THOR bundle).")
    p.add_argument("--max_assets", type=int, default=2000)
    p.add_argument("--num_physics", type=int, default=5,
                   help="Physics configs sampled per asset (within GPT-4o ranges).")
    p.add_argument("--num_pairs", type=int, default=500000,
                   help="Pairwise similarities to precompute.")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--n_views", type=int, default=3,
                   help="Multi-view renders per asset fed to GPT-4o (view 0 = DINO image).")
    p.add_argument("--gpt_model", type=str, default=None,
                   help="VLM model id. Default: gpt-4o (api) or "
                        "Qwen/Qwen2.5-VL-32B-Instruct (provider=qwen).")
    p.add_argument("--provider", type=str, default=None,
                   choices=["azure", "openai", "qwen"],
                   help="Force the attribution provider (else auto-detect from env). "
                        "'qwen' runs a local Qwen2.5-VL model on the GPU(s).")
    p.add_argument("--attribution_store", type=str, default="attributions/attributions.json",
                   help="Single shared, git-tracked JSON of GPT attributions (keyed by "
                        "asset_id). Commit this one file to share with everyone.")
    p.add_argument("--attribution_workers", type=int, default=8,
                   help="Concurrent GPT API calls (GPT is the throughput bottleneck).")
    p.add_argument("--mock", action="store_true",
                   help="Force offline mock attribution (no API calls).")
    p.add_argument("--no_category_hint", action="store_true",
                   help="Do NOT tell the VLM the known object class (pure-visual ablation).")
    p.add_argument("--keep_renders", action="store_true",
                   help="Keep rendered images. Default: delete them after generation "
                        "(regenerate v0 with scripts/render_images.py before training).")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    from train.data.generation import PhinoDataGenerator
    from train.data.assets import download_molmospaces_bundle

    # Resolve asset root: download a bundle if none provided.
    asset_root = args.asset_root
    if asset_root is None:
        subset = args.download_subset or "thor"
        asset_root = str(Path(args.output_dir) / "assets")
        logger.info(f"Downloading MolmoSpaces '{subset}' bundle to {asset_root}")
        download_molmospaces_bundle(asset_root, subset=subset, bundle=args.bundle)

    backend = "mock" if args.mock else "auto"
    # Sensible defaults for the local-Qwen path.
    gpt_model = args.gpt_model or (
        "Qwen/Qwen2.5-VL-32B-Instruct" if args.provider == "qwen" else "gpt-4o")
    # Auto-suffix the store with the model name (so gpt-4o / qwen stores stay
    # separate and it's obvious which model produced each), unless overridden.
    import re
    attribution_store = args.attribution_store
    if attribution_store == "attributions/attributions.json":
        slug = re.sub(r"[^a-z0-9.]+", "-", gpt_model.split("/")[-1].lower()).strip("-")
        attribution_store = f"attributions/attributions_{slug}.json"

    t0 = time.time()
    gen = PhinoDataGenerator(
        output_dir=args.output_dir,
        asset_root=asset_root,
        num_physics_per_asset=args.num_physics,
        max_assets=args.max_assets,
        seed=args.seed,
        attribution_backend=backend,
        gpt_model=gpt_model,
        img_size=args.img_size,
        n_views=args.n_views,
        attribution_store=attribution_store,
        attribution_workers=args.attribution_workers,
        provider=args.provider,
        use_category_hint=not args.no_category_hint,
    )
    gen.generate_all(compute_similarity=True, max_sim_pairs=args.num_pairs,
                     keep_renders=args.keep_renders)
    logger.info(f"Done in {(time.time()-t0)/60:.1f} min -> {args.output_dir}")


if __name__ == "__main__":
    main()
