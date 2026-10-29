"""
PHINO evaluation — linear probing on physics properties.

Trains a linear probe (mass, friction, restitution regression + category
classification) on frozen features from each backbone, and reports
mean +/- std over multiple seeds.

Usage:
    python scripts/evaluate.py \\
        --checkpoint checkpoints/phino/dynaclip_final.pt \\
        --data_dir data/phino_data \\
        --backbones phino dinov2_vitb14
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train.models.backbones import (
    load_dynaclip,
    load_dynaclip_clsmean,
    load_dinov2_vitb14,
    load_dinov2_vitb14_clsmean,
    load_dinov2_vitl14,
)
from train.data.dataset import create_probe_dataloader
from train.eval.linear_probing import PhysicsLinearProbing

logger = logging.getLogger(__name__)


# Map CLI names to loader callables. "phino_clsmean" uses CLS+Mean (1536d)
# features, which is the default reported in the paper.
LOADERS = {
    "phino": load_dynaclip,
    "phino_clsmean": load_dynaclip_clsmean,
    "dinov2_vitb14": load_dinov2_vitb14,
    "dinov2_vitb14_clsmean": load_dinov2_vitb14_clsmean,
    "dinov2_vitl14": load_dinov2_vitl14,
}


def setup_logging(log_file: str = None):
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)


def load_backbones(checkpoint_path: str, names: list, device: str) -> dict:
    backbones = {}
    for name in names:
        if name not in LOADERS:
            logger.warning(f"Unknown backbone: {name}, skipping. Available: {list(LOADERS.keys())}")
            continue
        t0 = time.time()
        try:
            if name.startswith("phino"):
                if not checkpoint_path:
                    logger.warning(f"{name} requires --checkpoint, skipping")
                    continue
                bb = LOADERS[name](checkpoint_path)
            else:
                bb = LOADERS[name]()
            bb = bb.to(device).eval()
            backbones[name] = bb
            logger.info(f"Loaded {name} ({bb.output_dim}d) in {time.time()-t0:.1f}s")
        except Exception as e:
            logger.error(f"Failed to load {name}: {e}")
    return backbones


def format_results_table(results: dict) -> str:
    lines = ["\n" + "=" * 70, "  Linear Probing Results", "=" * 70]
    all_props = sorted({p for r in results.values() for p in r.keys()})
    header = f"{'Backbone':<24}" + "".join(f" | {p:<20}" for p in all_props)
    lines.append(header)
    lines.append("-" * len(header))
    for bb_name, bb_results in results.items():
        row = f"{bb_name:<24}"
        for prop in all_props:
            if prop in bb_results:
                r = bb_results[prop]
                mean = r.get("mean", r.get("r2", r.get("accuracy", 0)))
                std = r.get("std", 0)
                row += f" | {mean:.4f} +/- {std:.4f} "
            else:
                row += f" | {'N/A':<20}"
        lines.append(row)
    return "\n".join(lines)


def make_serializable(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    return obj


def main():
    parser = argparse.ArgumentParser(description="PHINO linear-probing evaluation")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="PHINO checkpoint (required if probing PHINO backbones)")
    parser.add_argument("--data_dir", type=str, default="data/phino_data")
    parser.add_argument("--output_dir", type=str, default="results/eval")
    parser.add_argument("--backbones", type=str, nargs="+",
                        default=["phino_clsmean", "dinov2_vitb14_clsmean"],
                        help=f"Backbones to evaluate. Available: {list(LOADERS.keys())}")
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(str(output_dir / "eval.log"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    logger.info(f"Backbones: {args.backbones}")

    backbones = load_backbones(args.checkpoint, args.backbones, device)
    if not backbones:
        logger.error("No backbones loaded! Exiting.")
        return

    train_loader = create_probe_dataloader(args.data_dir, split="train",
                                           batch_size=args.batch_size, num_workers=args.num_workers,
                                           seed=args.split_seed)
    val_loader = create_probe_dataloader(args.data_dir, split="val",
                                         batch_size=args.batch_size, num_workers=args.num_workers,
                                           seed=args.split_seed)
    test_loader = create_probe_dataloader(args.data_dir, split="test",
                                          batch_size=args.batch_size, num_workers=args.num_workers,
                                           seed=args.split_seed)

    evaluator = PhysicsLinearProbing(
        backbones=backbones,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_seeds=args.num_seeds,
        num_epochs=args.num_epochs,
        device=device,
        probe_category=True,
    )

    t0 = time.time()
    results = evaluator.run()
    logger.info(f"Linear probing took {time.time()-t0:.0f}s")

    print(format_results_table(results))

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(make_serializable(results), f, indent=2)
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
