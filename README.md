https://drive.google.com/drive/folders/12a8F8zEfoZ542Ueq9VDHrAlJzYTBsg3w?usp=sharing

# PHINO: Physical Intuition via Newtonian Observations

Teach a DINOv2 vision backbone *physical intuition* by contrastive fine-tuning on
**how objects move**. For each 3D object we ask **GPT-4o** to estimate its
physical properties from rendered images (chain-of-thought), simulate it under a
fixed battery of forces in **MuJoCo**, and use pairwise **trajectory similarity**
as a soft contrastive target — so the frozen features come to encode mass /
friction / restitution that are invisible from a single still image.

```
MolmoSpaces asset ──render──▶ GPT-4o CoT ──mass/friction/restitution ranges──┐
        │                                                                     │
        └──────────────── MuJoCo (5 diagnostic forces) ──▶ trajectories ◀─────┘
                                                               │
                                              pairwise trajectory-MSE similarity
                                                               │
                                              SoftInfoNCE fine-tune of DINOv2
```

---

## Pipeline at a glance

| Step | Command | Produces |
|------|---------|----------|
| 0. Setup | `pip install -e .` | deps |
| 1+2+3. Generate | `python scripts/generate_data.py …` | GPT attributions, trajectories, similarity labels |
| 4. Render (pre-train) | `python scripts/render_images.py …` | the DINO training images (v0) |
| 5. Pretrain | `python scripts/pretrain.py …` | fine-tuned DINOv2 checkpoint |
| 6. Evaluate | `python scripts/evaluate.py …` | mass/friction/restitution R² |
| (any time) | `python scripts/smoke_test.py …` | inspectable images + force videos |

---

## 0. Setup

```bash
pip install -e .
export MUJOCO_GL=egl          # offscreen rendering (or osmesa). generate/render set this automatically.
```
Requires a GPU with an offscreen GL backend for MuJoCo rendering.

### Choose the attribution VLM

The model that reasons about physics is pluggable via `--provider`
(`azure` | `openai` | `qwen`). If omitted, the provider is auto-detected from env
vars. The chain-of-thought prompt is shared across all of them and lives in
`train/data/prompts/physical_attribution_cot.txt` (edit it without touching code).

**Azure OpenAI (GPT-4o)** — only the key is required; endpoint/version/deployment
have defaults (in `train/data/attribution.py`, `AZURE_*_DEFAULT`):
```bash
export AZURE_OPENAI_API_KEY="..."                 # required
export AZURE_OPENAI_ENDPOINT="https://<res>.openai.azure.com"
export AZURE_OPENAI_DEPLOYMENT="gpt-4o"           # deployment name (or AZURE_OPENAI_MODEL)
export AZURE_OPENAI_API_VERSION="2025-04-01-preview"
python scripts/generate_data.py --provider azure ...        # or just omit --provider
```

**Standard OpenAI / compatible proxy (incl. a local vLLM server)**:
```bash
export OPENAI_API_KEY="sk-..."                    # any non-empty string for a local server
export OPENAI_BASE_URL="https://.../v1"           # e.g. http://localhost:8000/v1 for vLLM
python scripts/generate_data.py --provider openai --gpt_model <model-id> ...
```

**Local Qwen2.5-VL (no API, runs on your GPU)** — reasons fully offline with
HuggingFace `transformers`. Default model `Qwen/Qwen2.5-VL-32B-Instruct`, loaded on
a single GPU (`QWEN_DEVICE`, default `cuda:0`; a 32B fits one 80GB card):
```bash
python scripts/generate_data.py --provider qwen ...                       # 32B (default)
python scripts/generate_data.py --provider qwen \
    --gpt_model Qwen/Qwen2.5-VL-72B-Instruct ...                          # 72B (needs more VRAM)
```
Needs `transformers>=4.49`, `accelerate`, `qwen-vl-utils` (`pip install` them).
Throughput ≈ ~30 s/asset on an H100 (`transformers.generate`). For large runs,
serve Qwen with **vLLM** instead and point `--provider openai` at it (paged-attention
+ batching ⇒ far higher throughput):
```bash
pip install vllm   # in a separate venv
vllm serve Qwen/Qwen2.5-VL-32B-Instruct --port 8000 --limit-mm-per-prompt '{"image": 3}'
OPENAI_API_KEY=x OPENAI_BASE_URL=http://localhost:8000/v1 \
  python scripts/generate_data.py --provider openai \
  --gpt_model Qwen/Qwen2.5-VL-32B-Instruct --attribution_workers 16 ...
```

**No provider / no key →** deterministic offline **mock** (mass anchored to asset
geometry) so the pipeline runs end-to-end without any model. Force with `--mock`.

> Tip: keep secrets in `~/.phino_env` (`export AZURE_OPENAI_API_KEY=...`) and
> `source ~/.phino_env` before running, to stay out of shell history.

---

## 1–3. Generate the dataset (`scripts/generate_data.py`)

One command does asset download → render → GPT attribution → MuJoCo simulation →
similarity labels.

```bash
# Live GPT-4o (key in env). Downloads a MolmoSpaces THOR bundle (~12 MB, ~91 objects).
python scripts/generate_data.py --download_subset thor \
    --max_assets 200 --num_physics 5 --n_views 3 \
    --num_pairs 200000 --output_dir data/phino_data

# Offline, no key (mock attribution):
python scripts/generate_data.py --download_subset thor --max_assets 8 --mock \
    --output_dir data/phino_smoke
```

What happens per asset:

1. **Get MolmoSpaces assets** (`train/data/assets.py`). MolmoSpaces
   (`allenai/molmospaces` on HuggingFace) ships MuJoCo objects packed as a parquet
   index + shard tars; we range-fetch just the bytes of one bundle, zstd-decompress,
   and untar to `<output_dir>/assets/`. Use `--asset_root` to point at already-extracted
   assets instead, or `--download_subset {thor,objaverse}` / `--bundle <name>`.
2. **Render** `--n_views` views (`render_multiview`) via MuJoCo offscreen — view 0 is
   the DINO training image; all views are sent to GPT-4o.
3. **GPT-4o attribution** (`train/data/attribution.py`): the views + a CoT prompt →
   `{mass_kg, static_friction, restitution}` each as a `{mean, low, high}` range,
   saved to the shared store (see below).
4. **Sample** `--num_physics` `PhysicsConfig`s within those ranges (same image /
   different physics → the contrastive hard negatives).
5. **Simulate + similarity** (next sections).

Key flags: `--num_physics` (configs/asset), `--num_pairs` (precomputed similarities),
`--n_views`, `--mock`, `--keep_renders` (default deletes renders), `--attribution_store`.

### Where attributions are saved

**One JSON file per model, keyed by `asset_id`**, auto-named after the model so you
can tell them apart (override with `--attribution_store`):

| provider / model | file |
|---|---|
| Azure/OpenAI GPT-4o | `attributions/attributions_gpt-4o.json` |
| local Qwen2.5-VL-32B | `attributions/attributions_qwen2.5-vl-32b-instruct.json` |

They live **outside** the gitignored `data/` so you can **commit and share them** —
anyone who clones reuses them and never re-calls the model:

```json
{
  "Boot_1": {
    "asset_id": "Boot_1",
    "mass_kg":        {"mean": 2.0, "low": 1.5, "high": 3.0},
    "static_friction":{"mean": 0.8, "low": 0.6, "high": 1.0},
    "restitution":    {"mean": 0.5, "low": 0.4, "high": 0.6},
    "reasoning": "hollow leg-shaped molds, likely plastic or rubber…",
    "backend": "live:azure"
  }
}
```

Each store is a cache: an `asset_id` already present is reused (no model call), and a
mock re-run never overwrites a real `live:*` / `qwen:*` entry. The `backend` field
records which model produced each row (`live:azure`, `live:openai`, `qwen:<model>`,
or `mock`). Share with:
```bash
git add attributions/ && git commit -m "Physical attributions (GPT-4o / Qwen)" && git push
```

---

## How trajectories are computed & compared

**Trajectory** (`train/data/mujoco_engine.py: MuJoCoPhysicsEngine.simulate_actions`).
For each (asset, sampled physics):
- Build a scene = asset MJCF + ground plane + light; gravity −9.81; implicit
  integrator (stable). Set **mass** (scale body mass/inertia to target), **friction**
  (`geom_friction`), **restitution** (contact `solref` damping ratio).
- Settle the object onto the plane, then apply **5 diagnostic forces** via
  `xfrc_applied`: `push_x`, `push_y`, `grasp_lift_release`, `lateral_flick`,
  `slow_press_down`.
- Record 50 frames at 20 Hz of a 13-D state `[pos(3), quat(4), linvel(3), angvel(3)]`
  (position relative to start; values clipped to a workspace bound so a near-massless
  object can't fly off and blow up the metric).
- The 5 actions are concatenated → one **(250, 13)** fingerprint, saved to
  `fingerprints/fp_XXXXXX.npz`.

**Similarity / MSE** (`TrajectoryNormalizer`).
- Fingerprints are **standardized per (action, feature)** using dataset statistics, so
  every force contributes comparably (a flick's metre-scale motion doesn't drown out a
  press's millimetres).
- For a pair, `MSE = mean over the standardized (250×13) elements of (zᵢ − zⱼ)²`.
  Note the alignment is **same-force-to-same-force**: `push_x` of i is compared with
  `push_x` of j, never with a different force — then the 5 per-force errors are averaged.
- `similarity = exp(−MSE / σ)`, with **σ auto-calibrated** so the median pairwise
  similarity ≈ 0.5 (well-spread soft labels, no saturation).
- Saved to `similarity_matrix.npz`; normalizer stats to `traj_norm.npz`.

---

## 4. Render the training images (`scripts/render_images.py`)

`generate_data.py` deletes renders by default (they're a regenerable cache; GPT
attribution is the durable artifact). Recreate the v0 training images before training:

```bash
python scripts/render_images.py --data_dir data/phino_data        # re-render v0
python scripts/render_images.py --data_dir data/phino_data --delete   # reclaim space after training
```
(Reads `mjcf_path` from `metadata.json`; the MolmoSpaces assets must still be present.)

---

## 5. Pretrain (`scripts/pretrain.py`)

SoftInfoNCE fine-tuning of DINOv2-B/14 with trajectory-similarity soft labels.

```bash
# single GPU
python scripts/pretrain.py --config configs/pretrain.yaml --seed 42

# multi-GPU
torchrun --nproc_per_node=8 scripts/pretrain.py --config configs/pretrain.yaml --seed 42
```
Override any config from the CLI, e.g. `data.data_dir=data/phino_data training.total_steps=30000`.
Checkpoints → `checkpoints/phino/dynaclip_final.pt`.

---

## 6. Evaluate (`scripts/evaluate.py`)

Linear-probe frozen features for mass / friction / restitution (R²) and category accuracy.

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/phino/dynaclip_final.pt \
    --data_dir data/phino_data \
    --backbones phino_clsmean dinov2_vitb14_clsmean
```

---

## Smoke test (`scripts/smoke_test.py`)

Produce inspectable artifacts for a few assets: multi-view renders, a **video per
force** (+ a 6-frame contact sheet), the trajectory, and the attribution JSON.

```bash
python scripts/smoke_test.py --asset_root data/molmospaces_thor/assets --n_assets 3
# -> data/smoke/{images,videos,trajectories}/  + attributions.json + summary.json
```

---

## Repo layout

```
phino/
├── attributions/attributions.json   # ★ shared GPT-4o attributions (git-tracked, key=asset_id)
├── configs/pretrain.yaml
├── train/data/
│   ├── assets.py          # MolmoSpaces download/extract + MuJoCo multi-view render
│   ├── attribution.py     # GPT-4o (Azure/OpenAI) CoT attribution + offline mock + single-file store
│   ├── prompts/physical_attribution_cot.txt   # the CoT prompt (edit freely)
│   ├── mujoco_engine.py   # MuJoCo physics, 5 diagnostic forces, TrajectoryNormalizer/MSE
│   ├── generation.py      # orchestration
│   └── dataset.py         # contrastive + probing datasets
├── train/{models,losses,trainers,eval,utils}/   # DINOv2 model, SoftInfoNCE, trainer, probes
└── scripts/{generate_data,render_images,pretrain,evaluate,smoke_test}.py
```

### Data layout (per `--output_dir`, gitignored)
```
data/phino_data/
├── metadata.json          # one row per (asset, sampled physics); links asset_id, mjcf_path, image, fingerprint
├── fingerprints/*.npz     # (250,13) MuJoCo trajectories
├── similarity_matrix.npz  # soft contrastive labels
├── traj_norm.npz          # normalizer stats (mean/std/sigma)
├── images/                # v0 training images (regenerable; deleted by default)
└── assets/                # extracted MolmoSpaces MJCF + meshes (if downloaded here)
```
Durable / shareable = `attributions/attributions.json`. Everything under `data/` is
regenerable from the assets + attributions.

---

## Notes
- Restitution is approximated in MuJoCo via the contact `solref` damping ratio
  (MuJoCo has no direct restitution coefficient).
- The Azure jailbreak content filter can reject role-y / strict-format prompts; the
  shipped prompt is phrased to pass it and is sent as a single user turn.
- Always evaluate on data generated with the same pipeline as training.
