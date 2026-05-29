# PHINO: Physical Intuition via Newtonian Observations

Trajectory-based physics contrastive fine-tuning of DINOv2 for
physics-grounded visual representations.

## Idea

For each 3D object asset we render a canonical image, ask **GPT-4o** (via
chain-of-thought) to estimate its physical properties (mass, friction,
restitution) as plausible ranges, sample several physics configurations within
those ranges, and run **MuJoCo** to produce short trajectories under a fixed set
of diagnostic actions. Pairwise full-trajectory MSE is then used as a *soft*
contrastive label, and SoftInfoNCE fine-tunes a DINOv2 backbone so that frozen
features encode physical properties that are otherwise invisible from a single
still image.

Assets come from **MolmoSpaces** (`allenai/molmospaces`): MuJoCo object assets
(MJCF + meshes) downloaded on demand from HuggingFace.

```
images ──render──▶ GPT-4o CoT ──ranges──▶ sample N physics ──▶ MuJoCo trajectories
                                                                      │
                                                          pairwise MSE similarity
                                                                      │
                                                   SoftInfoNCE fine-tune of DINOv2
```

## Repo layout

```
phino/
├── configs/pretrain.yaml          # Training config
├── train/
│   ├── data/
│   │   ├── assets.py              # MolmoSpaces download/extract + MuJoCo render
│   │   ├── attribution.py        # GPT-4o CoT physical attribution (+ offline mock)
│   │   ├── mujoco_engine.py      # MuJoCo physics + 5 diagnostic actions + similarity
│   │   ├── generation.py         # Orchestration (render→attribute→simulate→similarity)
│   │   └── dataset.py            # Contrastive + probing datasets
│   ├── models/
│   │   ├── dynaclip.py           # PHINO model (DINOv2 + projection head)
│   │   └── backbones.py          # Backbone wrappers for probing
│   ├── losses/contrastive.py     # SoftInfoNCE with soft trajectory labels
│   ├── trainers/pretrain.py      # Training loop (bf16, DDP, WiSE-FT optional)
│   ├── eval/linear_probing.py    # Physics-property linear probes
│   └── utils/helpers.py          # Seed, logging, DDP helpers
└── scripts/
    ├── generate_data.py          # Build the physics-grounded dataset
    ├── pretrain.py               # Training entry point
    └── evaluate.py               # Linear-probing entry point
```

## Setup

```bash
conda create -n phino python=3.10 -y
conda activate phino
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e .

# Offscreen MuJoCo rendering (set automatically by generate_data.py if unset):
export MUJOCO_GL=egl

# Live GPT-4o attribution (optional — omit to use the offline mock).
# Provider is auto-detected from whichever credentials are present:
#   Azure OpenAI:
export AZURE_OPENAI_API_KEY=...
export AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com
export AZURE_OPENAI_DEPLOYMENT=gpt-4o          # your deployment name
export AZURE_OPENAI_API_VERSION=2024-10-21     # optional
#   or standard OpenAI:
export OPENAI_API_KEY=sk-...
export OPENAI_BASE_URL=https://.../v1          # optional (proxy/compatible endpoint)
```

## 1. Generate training data

Assets are downloaded automatically from the MolmoSpaces HuggingFace dataset.
The generator renders each asset, attributes physics with GPT-4o (or the offline
mock if no `OPENAI_API_KEY`), simulates the 5 diagnostic actions in MuJoCo, and
precomputes pairwise trajectory similarities.

```bash
# Offline smoke run on a small THOR bundle (no API key needed):
python scripts/generate_data.py --download_subset thor \
    --max_assets 8 --num_physics 3 --num_pairs 500 --mock \
    --output_dir data/phino_smoke

# Full run with live GPT-4o on already-downloaded assets:
python scripts/generate_data.py --asset_root /data/molmospaces_thor \
    --max_assets 2000 --num_physics 5 --output_dir data/phino_data
```

Produces `images/`, cached `attribution/`, per-entry trajectory `fingerprints/`,
`metadata.json`, a similarity matrix, and an invisible-physics test split.

Key flags: `--download_subset {thor,objaverse}` (auto-download a bundle),
`--asset_root` (use existing assets), `--bundle` (specific archive),
`--num_physics` (configs sampled per asset), `--mock` (force offline
attribution).

## 2. Pretrain

Single GPU:
```bash
python scripts/pretrain.py --config configs/pretrain.yaml --seed 42
```

Multi-GPU:
```bash
torchrun --nproc_per_node=8 scripts/pretrain.py \
    --config configs/pretrain.yaml --seed 42
```

Default config: 30K steps, batch 1024, SoftInfoNCE + hard mining (30/30), bf16.
Checkpoint: `checkpoints/phino/dynaclip_final.pt`. Override any config value from
the CLI, e.g. `training.total_steps=10000 data.data_dir=data/my_data`.

## 3. Evaluate (linear probing)

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/phino/dynaclip_final.pt \
    --data_dir data/phino_data \
    --backbones phino_clsmean dinov2_vitb14_clsmean
```

Reports mass / friction / restitution R² and category accuracy
(mean ± std over seeds). PHINO is reported with `phino_clsmean` — concatenated
CLS + mean-pooled patch tokens (1536-d).

Available backbones: `phino`, `phino_clsmean`, `dinov2_vitb14`,
`dinov2_vitb14_clsmean`, `dinov2_vitl14`.

## Notes

- Always evaluate on data generated with the same pipeline as training.
- Rendering requires an offscreen GL backend (`MUJOCO_GL=egl`, or `osmesa`).
- Restitution is approximated in MuJoCo via the contact `solref` damping ratio
  (MuJoCo has no direct restitution coefficient).
- Without `OPENAI_API_KEY` the attributor falls back to a deterministic offline
  mock (mass anchored to asset geometry), so the full pipeline runs end-to-end
  without any API calls.
