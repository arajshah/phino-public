# PHINO × DINO-WM

Benchmark **PHINO** physics-contrastive encoders against a stock DINOv2 encoder
*inside* a **DINO-WM** world model, across several manipulation tasks. The idea:
DINO-WM keeps its visual encoder **frozen** and only learns an action-conditioned
latent predictor — so swapping the frozen encoder lets us measure, downstream,
whether PHINO's physics-aware features (mass / friction / restitution) make the
world model easier to learn and plan with.

```
PHINO fine-tuned DINOv2 (frozen)  ─┐
stock DINOv2-base       (frozen)  ─┼─► DINO-WM predictor (trained) ─► latent rollouts
                                   ┘        (same recipe for every encoder)
```

## What "DINO-WM" is here

DINO-WM is reproduced by **`stable-worldmodel`'s `prejepa`** trainer
(`stable-worldmodel/scripts/train/prejepa.py`; the upstream README states it
"reproduces DINO-WM", arXiv:2411.04983). The encoder is loaded by
`create_backbone(name)` → `transformers.AutoModel.from_pretrained(name)`, where
`name` is an HF alias (`dinov2_base`), an HF id, or a **local directory**. That
is the only seam we touch: **swapping the encoder is pure config — no changes to
the DINO-WM source.**

## Encoders compared (per task)

| tag            | `backbone.name`                                  | what it is                              |
|----------------|--------------------------------------------------|-----------------------------------------|
| `baseline`     | `dinov2_base`                                    | stock frozen DINOv2-ViT-B/14 (768-d)    |
| `phino_full`   | `checkpoints/phino_hf/phino_full`                | PHINO, all forces (mass+friction+restit.) |
| `phino_norest` | `checkpoints/phino_hf/phino_norest`              | PHINO with restitution removed          |

PHINO checkpoints are trained as `DynaCLIP` (DINOv2-ViT-B/14) and saved in the
original **facebookresearch/dinov2 hub layout**. DINO-WM expects a HuggingFace
`Dinov2Model`, so we convert them:

```bash
python phino_wm/convert_phino_to_hf.py downloads/phino_ckpts/dynaclip_final_v2.pt \
       checkpoints/phino_hf/phino_full
python phino_wm/convert_phino_to_hf.py downloads/phino_ckpts/dynaclip_final_v2_norestitution.pt \
       checkpoints/phino_hf/phino_norest
```

The converter remaps keys (fused `attn.qkv` → split q/k/v, `ls*.gamma` →
`layer_scale*.lambda1`, `embeddings.*`) and **verifies the result is bit-exact**
to the original backbone at native 518px (the only difference at 224px is HF's
positional-encoding interpolation, which is applied identically to every encoder,
so comparisons stay apples-to-apples).

## Tasks / datasets

Pulled from the `galilai-group` HF org (lance format), into
`~/.stable_worldmodel/datasets/`:

| task      | dataset                                  | size (lance) |
|-----------|------------------------------------------|--------------|
| `pusht`   | `galilai-group/lewm-pusht`               | ~5 GB        |
| `ogbcube` | `galilai-group/ogb_cube_single`          | ~8–20 GB     |
| `ogbscene`| `galilai-group/ogb_scene_single`         | ~38 GB       |

```bash
hf download galilai-group/lewm-pusht --repo-type dataset \
   --local-dir ~/.stable_worldmodel/datasets/galilai-group--lewm-pusht
```

> **Disk note.** Datasets are large and local disk is limited, so we run **one
> task at a time**: download → train all 3 encoders → delete the dataset → next
> task.

## Layout

```
phino_wm/
├── README.md
├── convert_phino_to_hf.py   # hub-DINOv2 ckpt -> HF Dinov2Model dir (+ verify)
├── train_dinowm.py          # thin launcher: shims an optional missing callback,
│                            #   then runs the UNMODIFIED prejepa.py via runpy
├── run_task.sh              # launch the 3 encoder runs for one task (2x H100, tmux)
└── logs/                    # per-run stdout (gitignored)

stable-worldmodel/scripts/train/config/
└── phino_<task>_<enc>.yaml  # 1 file per (task, encoder); inherits prejepa.yaml
```

`train_dinowm.py` exists only because the released `stable-pretraining` (0.1.7)
lacks `spt.callbacks.CPUOffloadCallback`, which prejepa.py hardcodes. The wrapper
injects a no-op stand-in at runtime so **prejepa.py stays untouched**.

## Run

```bash
# 0. (once) authenticate Weights & Biases
wandb login

# 1. train all 3 encoders for a task (W&B project: phino-dinowm)
phino_wm/run_task.sh pusht                       # default recipe (prejepa.yaml)
phino_wm/run_task.sh pusht trainer.max_epochs=3  # shorter budget

# 2. monitor
tmux ls
tail -f phino_wm/logs/pusht_baseline.log

# 3. when done, free disk and move on
rm -rf ~/.stable_worldmodel/datasets/galilai-group--lewm-pusht
phino_wm/run_task.sh ogbcube
```

Checkpoints are written to `~/.stable_worldmodel/checkpoints/<output_model_name>/`.

## Environment notes

- The local `stable-worldmodel` is used via `PYTHONPATH=stable-worldmodel`
  (not pip-installed). Its deps (`stable-pretraining`, `lightning`, `hydra-core`,
  `lancedb`, …) are installed into the active venv.
- `stable-worldmodel/` here is vendored from
  [galilai-group/stable-worldmodel](https://github.com/galilai-group/stable-worldmodel)
  @ commit `9d853d8`.
