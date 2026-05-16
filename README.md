# PHINO: Physical Intuition via Newtonian Observations

Minimal reference implementation of PHINO — trajectory-based physics
contrastive fine-tuning of DINOv2 for physics-grounded visual
representations.

## Idea

For each image, we sample physics configurations (mass, friction,
restitution) from category-grounded material priors, run an analytical
physics engine to produce short trajectories under a small set of
diagnostic actions, and treat pairwise trajectory MSE as a *soft*
contrastive label. SoftInfoNCE then fine-tunes a DINOv2 backbone so
that frozen features encode physical properties that are otherwise
invisible from a single still image.

## Repo layout

```
phino-minimal/
├── configs/pretrain.yaml          # Training config
├── dynaclip/
│   ├── data/
│   │   ├── generation.py          # Physics engine + data generation
│   │   └── dataset.py             # Contrastive + probing datasets
│   ├── models/
│   │   ├── dynaclip.py            # PHINO model (DINOv2 + projection head)
│   │   └── backbones.py           # Backbone wrappers for probing
│   ├── losses/contrastive.py      # SoftInfoNCE with soft trajectory labels
│   ├── trainers/pretrain.py       # Training loop (bf16, DDP, WiSE-FT optional)
│   ├── eval/linear_probing.py     # Physics-property linear probes
│   └── utils/helpers.py           # Seed, logging, DDP helpers
└── scripts/
    ├── generate_data.py           # Build the physics-grounded dataset
    ├── pretrain.py                # Training entry point
    └── evaluate.py                # Linear-probing entry point
```

## Setup

```bash
conda create -n phino python=3.10 -y
conda activate phino
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e .
```

## 1. Generate training data

Needs DomainNet `real` split at `<dataset_root>/domainnet/real/<category>/*.jpg`
(345 categories).

```bash
python scripts/generate_data.py \
    --dataset_root datasets \
    --output_dir data/phino_data \
    --max_images 20000 \
    --num_physics 5
```

Produces `metadata.json`, per-entry trajectory fingerprints, and a
500K-pair similarity matrix. ~30–60 min on CPU.

## 2. Pretrain

Single GPU:
```bash
python scripts/pretrain.py --config configs/pretrain.yaml --seed 42
```

Multi-GPU (recommended):
```bash
torchrun --nproc_per_node=8 scripts/pretrain.py \
    --config configs/pretrain.yaml --seed 42
```

Default config: 30K steps, batch 1024, SoftInfoNCE + hard mining (30/30),
WiSE-FT off, bf16. ~2–3h on 4–8 modern GPUs. Checkpoint:
`checkpoints/phino/dynaclip_final.pt`.

Override any config value from the CLI, e.g.
`training.total_steps=10000 data.data_dir=data/my_data`.

## 3. Evaluate (linear probing)

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/phino/dynaclip_final.pt \
    --data_dir data/phino_data \
    --backbones phino_clsmean dinov2_vitb14_clsmean
```

Reports mass / friction / restitution R² and category accuracy
(mean ± std over 5 seeds). PHINO is reported with `phino_clsmean` —
concatenated CLS + mean-pooled patch tokens (1536-d) — which is the
default feature extraction used in the paper.

Available backbones for `--backbones`:
`phino`, `phino_clsmean`, `dinov2_vitb14`, `dinov2_vitb14_clsmean`,
`dinov2_vitl14`.

## Notes

- Always evaluate on data generated with the same pipeline as training.
  Cross-generation eval is misleading (different material mappings,
  different similarity computation).
- `data/phino_data` is a placeholder path; pass `--data_dir` to choose
  another location.
