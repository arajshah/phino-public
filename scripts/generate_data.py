"""
Generate PHINO training data.

For each DomainNet image, samples N physics configurations from
category-grounded material priors, runs the analytical physics engine
for 4 diagnostic actions (push, lift, flick, press), and precomputes
pairwise trajectory-MSE similarities used as soft contrastive labels.

Output layout:
    {output_dir}/
        metadata.json          # N_images * num_physics entries
        fingerprints/          # one .npz per entry (trajectories + physics vector)
        similarity_matrix.npz  # precomputed pairwise similarities
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)

from dynaclip.data.generation import DynaCLIPDataGenerator

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate PHINO training data")
    parser.add_argument("--output_dir", type=str, default="data/phino_data")
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Root containing domainnet/real/<category>/*.jpg")
    parser.add_argument("--max_images", type=int, default=20000)
    parser.add_argument("--num_physics", type=int, default=5,
                        help="Physics configurations sampled per image")
    parser.add_argument("--num_pairs", type=int, default=500000,
                        help="Pairwise similarities to precompute")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t0 = time.time()
    logger.info(f"Generating PHINO data: output={args.output_dir}, "
                f"max_images={args.max_images}, num_physics={args.num_physics}")

    gen = DynaCLIPDataGenerator(
        output_dir=args.output_dir,
        dataset_root=args.dataset_root,
        num_physics_per_image=args.num_physics,
        max_images=args.max_images,
        seed=args.seed,
        similarity_metric="l2",
    )
    gen.generate_all(compute_similarity=True, max_sim_pairs=args.num_pairs)

    logger.info(f"Done in {(time.time()-t0)/60:.1f} min -> {args.output_dir}")


if __name__ == "__main__":
    main()
