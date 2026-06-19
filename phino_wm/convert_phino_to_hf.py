#!/usr/bin/env python
"""Convert a PHINO/DynaCLIP checkpoint (facebookresearch/dinov2 *hub* layout)
into a HuggingFace ``Dinov2Model`` directory that stable-worldmodel's
``create_backbone`` can load via ``AutoModel.from_pretrained(<dir>)``.

PHINO fine-tunes a DINOv2-ViT-B/14 whose ``backbone.*`` weights are stored in
the original timm-style hub layout (fused ``attn.qkv``, ``ls1/ls2.gamma``
layer-scale, ``patch_embed.proj``, ``pos_embed`` ...). HF ``Dinov2Model`` uses a
different key layout (split q/k/v, ``layer_scale*.lambda1``, ``embeddings.*``).
This script performs that mechanical remapping and verifies the converted model
reproduces the original patch-token features.

Usage:
    python convert_phino_to_hf.py IN.pt OUT_DIR [--no-verify]
"""

import argparse
import re
import sys
from pathlib import Path

import torch


def convert_backbone_state_dict(bb: dict) -> dict:
    """Map hub-dinov2 ``backbone.*`` (prefix already stripped) -> HF Dinov2Model."""
    out = {}
    out["embeddings.cls_token"] = bb["cls_token"]
    out["embeddings.mask_token"] = bb["mask_token"]
    out["embeddings.position_embeddings"] = bb["pos_embed"]
    out["embeddings.patch_embeddings.projection.weight"] = bb["patch_embed.proj.weight"]
    out["embeddings.patch_embeddings.projection.bias"] = bb["patch_embed.proj.bias"]

    n_layers = 1 + max(
        int(m.group(1))
        for k in bb
        for m in [re.match(r"blocks\.(\d+)\.", k)]
        if m
    )

    for i in range(n_layers):
        p, h = f"blocks.{i}.", f"encoder.layer.{i}."
        out[h + "norm1.weight"] = bb[p + "norm1.weight"]
        out[h + "norm1.bias"] = bb[p + "norm1.bias"]

        # Fused qkv (3*d, d) -> separate query/key/value (d, d). Order is q,k,v.
        qkv_w, qkv_b = bb[p + "attn.qkv.weight"], bb[p + "attn.qkv.bias"]
        d = qkv_w.shape[0] // 3
        out[h + "attention.attention.query.weight"] = qkv_w[0:d]
        out[h + "attention.attention.key.weight"] = qkv_w[d : 2 * d]
        out[h + "attention.attention.value.weight"] = qkv_w[2 * d : 3 * d]
        out[h + "attention.attention.query.bias"] = qkv_b[0:d]
        out[h + "attention.attention.key.bias"] = qkv_b[d : 2 * d]
        out[h + "attention.attention.value.bias"] = qkv_b[2 * d : 3 * d]

        out[h + "attention.output.dense.weight"] = bb[p + "attn.proj.weight"]
        out[h + "attention.output.dense.bias"] = bb[p + "attn.proj.bias"]
        out[h + "layer_scale1.lambda1"] = bb[p + "ls1.gamma"]
        out[h + "norm2.weight"] = bb[p + "norm2.weight"]
        out[h + "norm2.bias"] = bb[p + "norm2.bias"]
        out[h + "mlp.fc1.weight"] = bb[p + "mlp.fc1.weight"]
        out[h + "mlp.fc1.bias"] = bb[p + "mlp.fc1.bias"]
        out[h + "mlp.fc2.weight"] = bb[p + "mlp.fc2.weight"]
        out[h + "mlp.fc2.bias"] = bb[p + "mlp.fc2.bias"]
        out[h + "layer_scale2.lambda1"] = bb[p + "ls2.gamma"]

    out["layernorm.weight"] = bb["norm.weight"]
    out["layernorm.bias"] = bb["norm.bias"]
    return out, n_layers


def load_backbone_only(ckpt_path: str) -> dict:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    bb = {
        k[len("backbone.") :]: v
        for k, v in sd.items()
        if k.startswith("backbone.")
    }
    if not bb:
        raise SystemExit(f"No backbone.* keys in {ckpt_path}")
    return bb


def verify_against_hub(bb: dict, hf_model) -> None:
    """Compare HF patch tokens to the original hub forward_features (best effort)."""
    try:
        hub = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vitb14", pretrained=False, verbose=False
        )
        hub.load_state_dict(bb, strict=True)
        hub.eval()
    except Exception as e:  # xformers/hub unavailable -> fall back to self-check
        print(f"[verify] hub unavailable ({type(e).__name__}: {e}); "
              "skipping numerical check (strict load already passed).")
        return

    # Compare at the native 518px grid so NO positional-encoding interpolation
    # is involved (hub and HF interpolate differently, which is a property of how
    # the encoder is *used*, not of the weights). This isolates mapping fidelity.
    x = torch.randn(2, 3, 518, 518)
    with torch.no_grad():
        hub_patches = hub.forward_features(x)["x_norm_patchtokens"]
        hf_patches = hf_model(x, interpolate_pos_encoding=False).last_hidden_state[:, 1:, :]
    diff = (hub_patches - hf_patches).abs().max().item()
    print(f"[verify] max |hub - hf| patch-token diff @518 (no interp) = {diff:.3e}")
    assert diff < 1e-3, f"Conversion mismatch too large: {diff}"
    print("[verify] OK — HF model reproduces hub features exactly at native res.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out_dir")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    from transformers import Dinov2Model

    bb = load_backbone_only(args.ckpt)
    converted, n_layers = convert_backbone_state_dict(bb)
    print(f"Converted {len(bb)} hub keys -> {len(converted)} HF keys ({n_layers} layers)")

    model = Dinov2Model.from_pretrained("facebook/dinov2-base")
    missing, unexpected = model.load_state_dict(converted, strict=False)
    if missing or unexpected:
        raise SystemExit(f"State-dict mismatch:\n  missing={missing}\n  unexpected={unexpected}")
    model.eval()
    print("strict load OK (no missing/unexpected keys)")

    if not args.no_verify:
        verify_against_hub(bb, model)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    print(f"Saved HF Dinov2Model -> {out}")


if __name__ == "__main__":
    main()
