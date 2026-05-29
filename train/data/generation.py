"""
PHINO data generation: GPT-4o attribution + MuJoCo trajectories.

Pipeline (per MolmoSpaces object asset):

    1. render the asset to a canonical image  (train/data/assets.render_asset)
    2. GPT-4o chain-of-thought predicts mass / friction / restitution *ranges*
       from that image                        (train/data/attribution)
    3. sample `num_physics` PhysicsConfigs within those ranges
    4. for each config, simulate the 5 diagnostic actions in MuJoCo and record
       a 13-D trajectory fingerprint          (train/data/mujoco_engine)
    5. precompute pairwise full-trajectory MSE similarities -> soft labels

This replaces the old hardcoded category->material priors and the analytical
point-mass engine. The on-disk output format is unchanged, so the dataset /
model / trainer / eval code is reused as-is:

    <output_dir>/
        images/<asset_id>.png        # rendered views (DINO input)
        attribution/<asset_id>.json  # cached GPT-4o predictions
        fingerprints/fp_XXXXXX.npz   # one trajectory per (asset, physics)
        metadata.json
        similarity_matrix.npz
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import numpy as np

# Re-export so existing imports keep working.
from train.data.mujoco_engine import (
    MuJoCoPhysicsEngine,
    DiagnosticAction,
    DIAGNOSTIC_ACTIONS,
    TrajectoryNormalizer,
    compute_dynamics_similarity_mse,
    compute_dynamics_similarity_l2,
    SIMILARITY_METRICS,
)
from train.data.attribution import PhysicalAttributor, AttributionResult

logger = logging.getLogger(__name__)


# ==========================================================================
# Physical property configuration (now sourced from GPT-4o ranges)
# ==========================================================================
@dataclass
class PhysicsConfig:
    mass: float = 1.0
    static_friction: float = 0.5
    dynamic_friction: float = 0.4
    restitution: float = 0.3
    material: str = "object"

    def __post_init__(self):
        self.dynamic_friction = 0.8 * self.static_friction

    @staticmethod
    def sample_from_attribution(
        attr: AttributionResult, rng: np.random.Generator, category: str = "object",
    ) -> "PhysicsConfig":
        """Sample one config within the GPT-4o-predicted ranges.

        Mass is sampled log-uniformly (spans orders of magnitude); friction and
        restitution uniformly. Sampling within each asset's range is what gives
        the same-image/different-physics hard negatives the loss relies on.
        """
        m = attr.mass_kg
        mass = float(np.exp(rng.uniform(np.log(max(m.low, 1e-3)),
                                        np.log(max(m.high, 1e-3)))))
        sf = float(rng.uniform(attr.static_friction.low, attr.static_friction.high))
        rest = float(rng.uniform(attr.restitution.low, attr.restitution.high))
        return PhysicsConfig(mass=mass, static_friction=sf,
                             restitution=rest, material=category)

    @property
    def uid(self) -> str:
        s = f"m{self.mass:.6f}_sf{self.static_friction:.6f}_r{self.restitution:.6f}"
        return hashlib.md5(s.encode()).hexdigest()[:12]

    def to_vector(self) -> np.ndarray:
        return np.array([np.log(self.mass + 1e-6) / np.log(10.0),
                         self.static_friction / 1.5,
                         self.restitution / 0.95], dtype=np.float32)


# ==========================================================================
# Full data generation pipeline
# ==========================================================================
class PhinoDataGenerator:
    def __init__(
        self,
        output_dir: str,
        asset_root: str,
        num_physics_per_asset: int = 5,
        max_assets: int = 2000,
        seed: int = 42,
        attribution_backend: str = "auto",  # auto | live | mock
        gpt_model: str = "gpt-4o",
        img_size: int = 224,
        n_views: int = 4,
        similarity_metric: str = "mse",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.asset_root = asset_root
        self.num_physics = num_physics_per_asset
        self.max_assets = max_assets
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.img_size = img_size
        self.n_views = n_views
        self.sim_fn = SIMILARITY_METRICS.get(similarity_metric, compute_dynamics_similarity_mse)

        self.engine = MuJoCoPhysicsEngine()
        self.attributor = PhysicalAttributor(
            cache_dir=str(self.output_dir / "attribution"),
            backend=attribution_backend,
            model=gpt_model,
            seed=seed,
        )

    def generate_all(self, compute_similarity: bool = True, max_sim_pairs: int = 500000,
                     keep_renders: bool = False):
        from train.data.assets import discover_assets, render_multiview

        logger.info("=== PHINO data generation (GPT-4o + MuJoCo) ===")
        records = discover_assets(self.asset_root, max_assets=self.max_assets)
        if not records:
            raise RuntimeError(f"No assets found under {self.asset_root}")

        img_dir = self.output_dir / "images"
        fp_dir = self.output_dir / "fingerprints"
        img_dir.mkdir(parents=True, exist_ok=True)
        fp_dir.mkdir(parents=True, exist_ok=True)

        metadata: List[dict] = []
        fingerprints: List[np.ndarray] = []
        attributions: dict = {}  # asset_id -> GPT attribution (consolidated)
        image_group = 0  # one group per successfully-processed asset

        for rec in records:
            try:
                # 1. render multi-view images (view 0 = canonical DINO training image)
                views = render_multiview(
                    rec.mjcf_path, str(img_dir), rec.asset_id,
                    n_views=self.n_views, img_size=self.img_size, engine=self.engine,
                )
                img_path = views[0]
                mass_hint = self._geometric_mass(rec.mjcf_path)

                # 2. GPT-4o attribution from all views (cached) -> property ranges
                attr = self.attributor.attribute(
                    rec.asset_id, image_paths=views, mass_hint=mass_hint,
                )
                attributions[rec.asset_id] = attr.to_dict()
                attr_path = str(self.output_dir / "attribution"
                                / f"{_safe(rec.asset_id)}.json")

                # 3-4. sample N physics configs, simulate trajectories
                for _ in range(self.num_physics):
                    physics = PhysicsConfig.sample_from_attribution(
                        attr, self.rng, category=rec.category,
                    )
                    flat = self.engine.simulate_actions(rec.mjcf_path, physics)
                    gidx = len(metadata)
                    fp_path = fp_dir / f"fp_{gidx:06d}.npz"
                    np.savez_compressed(
                        str(fp_path), flat_trajectory=flat,
                        mass=physics.mass, static_friction=physics.static_friction,
                        dynamic_friction=physics.dynamic_friction,
                        restitution=physics.restitution,
                    )
                    fingerprints.append(flat)
                    metadata.append({
                        "image_path": str(img_path),
                        "view_paths": views,
                        "category": rec.category,
                        "material": rec.category,
                        "asset_id": rec.asset_id,
                        "mjcf_path": rec.mjcf_path,
                        "attribution_path": attr_path,
                        "global_idx": gidx,
                        "image_group": image_group,
                        "physics_uid": physics.uid,
                        "mass": float(physics.mass),
                        "static_friction": float(physics.static_friction),
                        "dynamic_friction": float(physics.dynamic_friction),
                        "restitution": float(physics.restitution),
                        "fingerprint_path": str(fp_path),
                    })
                image_group += 1
                if image_group % 50 == 0:
                    logger.info(f"Processed {image_group}/{len(records)} assets")
            except Exception as e:
                logger.warning(f"Skipping asset {rec.asset_id}: {type(e).__name__}: {e}")

        if not metadata:
            raise RuntimeError("No assets simulated successfully")

        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)
        # Consolidated GPT attributions, keyed by asset_id (one row per asset).
        with open(self.output_dir / "attributions.json", "w") as f:
            json.dump(attributions, f, indent=2)
        logger.info(f"Saved metadata for {len(metadata)} entries "
                    f"({image_group} assets x {self.num_physics} physics); "
                    f"attributions for {len(attributions)} assets")

        if compute_similarity:
            self._compute_similarity_matrix(fingerprints, max_pairs=max_sim_pairs)
        self._generate_invisible_physics_testset(metadata, fingerprints)

        # Renders are a regenerable cache (GPT attribution is the durable artifact);
        # delete them unless asked to keep. Re-create the DINO training images later
        # with `scripts/render_images.py` (reads metadata.json -> renders v0).
        if not keep_renders:
            import shutil
            shutil.rmtree(img_dir, ignore_errors=True)
            logger.info(f"Deleted renders at {img_dir}; run scripts/render_images.py "
                        "before training to regenerate the v0 training images.")
        logger.info("=== Data generation complete ===")

    # ----- helpers -------------------------------------------------------
    def _geometric_mass(self, mjcf_path: str) -> float:
        """Asset mass from geometry x default density (used to anchor mock attribution)."""
        try:
            model, _, info = self.engine.build_scene(mjcf_path)
            return float(model.body_subtreemass[info["body_id"]])
        except Exception:
            return 0.0

    def _compute_similarity_matrix(self, fingerprints: list, max_pairs: int = 500000):
        n = len(fingerprints)
        if n < 2:
            logger.warning("Not enough fingerprints for a similarity matrix")
            return
        # Fit + persist the trajectory normalizer (per-action z-score + auto sigma)
        # so labels are well-spread and the dataset can reuse identical stats.
        self._normalizer = TrajectoryNormalizer(
            n_actions=len(DIAGNOSTIC_ACTIONS), timesteps=self.engine.timesteps,
        ).fit(fingerprints, rng=self.rng)
        self._normalizer.save(str(self.output_dir / "traj_norm.npz"))
        logger.info(f"Fitted trajectory normalizer (auto sigma={self._normalizer.sigma:.3f})")

        num_pairs = min(max_pairs, n * (n - 1) // 2)
        logger.info(f"Computing {num_pairs} pairwise similarities for {n} configs")
        pairs = np.zeros((num_pairs, 2), dtype=np.int64)
        sims = np.zeros(num_pairs, dtype=np.float32)
        for k in range(num_pairs):
            if k % 100000 == 0 and k:
                logger.info(f"  pair {k}/{num_pairs}")
            i, j = self.rng.choice(n, size=2, replace=False)
            pairs[k] = [i, j]
            sims[k] = self._normalizer.similarity(fingerprints[i], fingerprints[j])
        np.savez_compressed(self.output_dir / "similarity_matrix.npz",
                            pairs=pairs, similarities=sims)
        logger.info(f"Saved {num_pairs} similarities (mean={sims.mean():.4f}, "
                    f"std={sims.std():.4f}, ==0: {(sims < 1e-3).mean()*100:.1f}%)")

    def _generate_invisible_physics_testset(self, metadata: list, fingerprints: list,
                                            num_pairs: int = 500):
        """Same image, different physics — reuses already-simulated trajectories."""
        groups: dict = {}
        for entry in metadata:
            groups.setdefault(entry["image_group"], []).append(entry["global_idx"])
        multi = [g for g, idxs in groups.items() if len(idxs) >= 2]
        if not multi:
            return
        test_pairs = []
        for _ in range(min(num_pairs, len(multi) * 4)):
            g = multi[self.rng.integers(len(multi))]
            i, j = self.rng.choice(groups[g], size=2, replace=False)
            ei, ej = metadata[i], metadata[j]
            sim_fn = (self._normalizer.similarity
                      if getattr(self, "_normalizer", None) else self.sim_fn)
            dyn_sim = float(sim_fn(fingerprints[i], fingerprints[j]))
            test_pairs.append({
                "image_path": ei["image_path"], "category": ei["category"],
                "material": ei["material"],
                "physics_a": {"mass": ei["mass"], "static_friction": ei["static_friction"],
                              "restitution": ei["restitution"]},
                "physics_b": {"mass": ej["mass"], "static_friction": ej["static_friction"],
                              "restitution": ej["restitution"]},
                "dynamics_similarity": dyn_sim,
            })
        with open(self.output_dir / "invisible_physics_test.json", "w") as f:
            json.dump(test_pairs, f, indent=2)
        logger.info(f"Saved {len(test_pairs)} invisible-physics test pairs")


def _safe(name: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)
