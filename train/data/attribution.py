"""
GPT-4o chain-of-thought physical attribution.

For each MolmoSpaces asset we ask GPT-4o to look at a rendered image and reason
step-by-step about the object's material, size and structure, then emit a strict
JSON estimate of three physical properties as ranges:

    mass_kg, static_friction, restitution    each = {mean, low, high}

The range (rather than a single value) lets the data generator sample N physics
configurations per asset, preserving the "same image / different physics"
hard-negative structure the contrastive loss relies on.

Two backends:
  - live : GPT-4o vision. Auto-detects the provider from env vars:
      * Azure OpenAI  -> AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT
                         (+ AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_API_VERSION)
      * OpenAI        -> OPENAI_API_KEY (+ optional OPENAI_BASE_URL)
  - mock : deterministic, offline. Seeds friction/restitution from the asset id
           and anchors mass to a geometry-derived hint when available. Lets the
           whole pipeline run with no API key (used by the smoke test).

Every result is cached as one JSON per asset under <cache_dir>/<asset_id>.json,
so re-runs and live calls are never repeated.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# Azure OpenAI defaults — only AZURE_OPENAI_API_KEY must be set in the env (it is
# the secret). Endpoint / api-version / deployment are non-secret and default
# here; override any of them with the matching AZURE_OPENAI_* env var.
AZURE_ENDPOINT_DEFAULT = "https://cml-amine.openai.azure.com"
AZURE_API_VERSION_DEFAULT = "2025-04-01-preview"
AZURE_DEPLOYMENT_DEFAULT = "gpt-4o"   # cheap vision model; your Azure deployment name


# Plausible physical bounds used to clamp predictions (and bound the mock).
MASS_BOUNDS = (0.01, 50.0)        # kg
FRICTION_BOUNDS = (0.05, 1.5)     # static friction coefficient
RESTITUTION_BOUNDS = (0.0, 0.95)  # coefficient of restitution


@dataclass
class PropertyRange:
    mean: float
    low: float
    high: float

    def clamp(self, lo: float, hi: float) -> "PropertyRange":
        mean = float(np.clip(self.mean, lo, hi))
        low = float(np.clip(min(self.low, self.mean), lo, hi))
        high = float(np.clip(max(self.high, self.mean), lo, hi))
        if low > high:
            low, high = high, low
        return PropertyRange(mean, low, high)


@dataclass
class AttributionResult:
    asset_id: str
    mass_kg: PropertyRange
    static_friction: PropertyRange
    restitution: PropertyRange
    reasoning: str = ""
    backend: str = "mock"

    def to_dict(self) -> dict:
        return {
            "asset_id": self.asset_id,
            "mass_kg": asdict(self.mass_kg),
            "static_friction": asdict(self.static_friction),
            "restitution": asdict(self.restitution),
            "reasoning": self.reasoning,
            "backend": self.backend,
        }

    @staticmethod
    def from_dict(d: dict) -> "AttributionResult":
        return AttributionResult(
            asset_id=d["asset_id"],
            mass_kg=PropertyRange(**d["mass_kg"]),
            static_friction=PropertyRange(**d["static_friction"]),
            restitution=PropertyRange(**d["restitution"]),
            reasoning=d.get("reasoning", ""),
            backend=d.get("backend", "mock"),
        )


# The chain-of-thought system prompt lives in a standalone text file so it can be
# edited / version-controlled without touching code.
PROMPT_PATH = Path(__file__).parent / "prompts" / "physical_attribution_cot.txt"
COT_SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8").strip()


class PhysicalAttributor:
    def __init__(
        self,
        store_path: str,
        backend: str = "auto",          # auto | live | mock
        model: str = "gpt-4o",
        seed: int = 42,
        provider: Optional[str] = None,  # force azure | openai | qwen (else auto-detect)
    ):
        self._forced_provider = provider
        self._qwen_model = None
        self._qwen_processor = None
        # All attributions live in ONE json file keyed by asset_id (the shared,
        # git-trackable store). Loaded once, written back whenever a new asset is
        # attributed, so GPT is never called twice for the same asset.
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._store: dict = {}
        if self.store_path.exists():
            try:
                self._store = json.loads(self.store_path.read_text())
            except Exception:
                logger.warning(f"Corrupt attribution store {self.store_path}, starting fresh")
        self.model = model
        self.seed = seed
        self.provider = None  # set to "azure" | "openai" when backend == "live"
        self.backend = self._resolve_backend(backend)
        self._client = None
        logger.info(f"PhysicalAttributor backend={self.backend} provider={self.provider} "
                    f"model={model} store={self.store_path} ({len(self._store)} cached)")

    @staticmethod
    def _detect_provider() -> Optional[str]:
        """Return 'azure' / 'openai' based on which credentials are present.

        Azure only needs the secret key in the env — endpoint/version/deployment
        have defaults (see AZURE_*_DEFAULT above).
        """
        if os.environ.get("AZURE_OPENAI_API_KEY"):
            return "azure"
        if os.environ.get("OPENAI_API_KEY"):
            return "openai"
        return None

    def _resolve_backend(self, backend: str) -> str:
        if backend == "mock":
            return "mock"
        # qwen = local transformers model, no API credentials needed
        if self._forced_provider == "qwen":
            self.provider = "qwen"
            return "live"
        provider = self._forced_provider or self._detect_provider()
        if backend == "live":
            if provider is None:
                raise RuntimeError(
                    "backend='live' but no credentials found. Set AZURE_OPENAI_API_KEY"
                    " + AZURE_OPENAI_ENDPOINT (Azure) or OPENAI_API_KEY (OpenAI), "
                    "or use --provider qwen for a local model.")
            self.provider = provider
            return "live"
        # auto
        if provider is not None:
            self.provider = provider
            return "live"
        return "mock"

    # ----- public API ----------------------------------------------------
    def attribute(
        self,
        asset_id: str,
        image_paths: Optional[list] = None,
        image_path: Optional[str] = None,
        mass_hint: Optional[float] = None,
        category: Optional[str] = None,
    ) -> AttributionResult:
        """Return cached / freshly-computed physical attribution for one asset.

        `image_paths` is a list of multi-view renders fed to the VLM; `image_path`
        is accepted for backwards compatibility (single view). `category` is the
        asset's known object class (from MolmoSpaces) — given to the VLM so it
        grounds the object correctly instead of guessing identity from pixels.
        """
        if image_paths is None:
            image_paths = [image_path] if image_path else []

        if asset_id in self._store:
            try:
                return AttributionResult.from_dict(self._store[asset_id])
            except Exception:
                logger.warning(f"Corrupt entry for {asset_id} in store, recomputing")

        result = self._compute(asset_id, image_paths, mass_hint, category)
        self._store[asset_id] = result.to_dict()
        self._save()
        return result

    def attribute_many(self, items: list, max_workers: int = 8) -> dict:
        """Attribute many assets, calling the VLM concurrently (API is IO-bound).

        `items` is a list of (asset_id, image_paths, mass_hint, category). Returns
        {asset_id: AttributionResult}. Cached assets are returned without a call;
        the store is written once at the end.
        """
        results, todo = {}, []
        for asset_id, paths, hint, cat in items:
            if asset_id in self._store:
                try:
                    results[asset_id] = AttributionResult.from_dict(self._store[asset_id])
                    continue
                except Exception:
                    pass
            todo.append((asset_id, paths, hint, cat))
        if not todo:
            return results

        # Concurrency only helps IO-bound API providers; a local GPU model (qwen)
        # runs serially (threads would contend / OOM).
        if self.backend == "live" and self.provider in ("azure", "openai") and max_workers > 1:
            self._client, self._deployment = self._build_client()  # pre-build (thread-safe reuse)
            from concurrent.futures import ThreadPoolExecutor, as_completed
            logger.info(f"Attributing {len(todo)} assets with {max_workers} concurrent workers")
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = {ex.submit(self._compute, a, p, h, c): a for a, p, h, c in todo}
                done = 0
                for fut in as_completed(futs):
                    results[futs[fut]] = fut.result()
                    done += 1
                    if done % 20 == 0:
                        logger.info(f"  attributed {done}/{len(todo)}")
        else:
            for asset_id, paths, hint, cat in todo:
                results[asset_id] = self._compute(asset_id, paths, hint, cat)

        for asset_id, res in results.items():
            self._store[asset_id] = res.to_dict()
        self._save()
        return results

    def _compute(self, asset_id: str, image_paths: list,
                 mass_hint: Optional[float], category: Optional[str] = None) -> AttributionResult:
        """Compute one attribution (with mock fallback), clamped. No store write."""
        if self.backend == "live":
            try:
                if self.provider == "qwen":
                    result = self._attribute_qwen(asset_id, image_paths, category)
                else:
                    result = self._attribute_live(asset_id, image_paths, category)
            except Exception as e:
                logger.error(f"VLM attribution failed for {asset_id}: {e}; "
                             "falling back to mock")
                result = self._attribute_mock(asset_id, mass_hint)
        else:
            result = self._attribute_mock(asset_id, mass_hint)
        return self._clamp(result)

    def _prompt_for(self, category: Optional[str]) -> str:
        """CoT prompt, optionally grounded with the known object class so the VLM
        doesn't have to guess identity from pixels (e.g. blinds vs a brush)."""
        if not category:
            return COT_SYSTEM_PROMPT
        nice = re.sub(r"(?<!^)(?=[A-Z])", " ", str(category)).replace("_", " ").strip().lower()
        return f"The object shown is a {nice}.\n\n{COT_SYSTEM_PROMPT}"

    def _save(self):
        """Write the whole store back to the single json file (atomic)."""
        tmp = self.store_path.with_name(self.store_path.name + ".tmp")
        tmp.write_text(json.dumps(self._store, indent=2))
        tmp.replace(self.store_path)

    # ----- backends ------------------------------------------------------
    def _ensure_qwen(self):
        """Lazy-load a local Qwen2.5-VL model across available GPUs."""
        if self._qwen_model is not None:
            return
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        # Load on a SINGLE GPU. device_map="auto" splits a model that already fits
        # on one 80GB H100 across both cards as a *sequential* pipeline (only one GPU
        # active at a time + per-layer cross-GPU copies) → ~2 tok/s. One card → 10-20x.
        dev = os.environ.get("QWEN_DEVICE", "cuda:0")
        logger.info(f"Loading local VLM {self.model} on {dev} (bf16, sdpa)…")
        self._qwen_model = AutoModelForImageTextToText.from_pretrained(
            self.model, dtype=torch.bfloat16, device_map=dev,
            attn_implementation="sdpa")
        self._qwen_model.eval()
        self._qwen_processor = AutoProcessor.from_pretrained(self.model)
        logger.info("Local VLM loaded")

    def _attribute_qwen(self, asset_id: str, image_paths: list,
                        category: Optional[str] = None) -> AttributionResult:
        valid = [p for p in (image_paths or []) if p and Path(p).exists()]
        if not valid:
            raise FileNotFoundError(f"no images for qwen attribution: {image_paths}")
        self._ensure_qwen()
        import torch
        from qwen_vl_utils import process_vision_info

        content = [{"type": "image", "image": f"file://{Path(p).resolve()}"} for p in valid]
        content.append({"type": "text", "text": self._prompt_for(category)})
        messages = [{"role": "user", "content": content}]
        proc = self._qwen_processor
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = proc(text=[text], images=image_inputs, videos=video_inputs,
                      padding=True, return_tensors="pt").to(self._qwen_model.device)
        with torch.no_grad():
            # Qwen writes verbose CoT before the JSON — allow enough tokens so the
            # final JSON block isn't truncated (would fall back to mock otherwise).
            gen = self._qwen_model.generate(**inputs, max_new_tokens=1280, do_sample=False)
        trimmed = gen[:, inputs.input_ids.shape[1]:]
        out = proc.batch_decode(trimmed, skip_special_tokens=True,
                                clean_up_tokenization_spaces=False)[0]
        parsed = _extract_json(out)
        return AttributionResult(
            asset_id=asset_id,
            mass_kg=PropertyRange(**parsed["mass_kg"]),
            static_friction=PropertyRange(**parsed["static_friction"]),
            restitution=PropertyRange(**parsed["restitution"]),
            reasoning=parsed.get("reasoning", out[:200]),
            backend=f"qwen:{self.model.split('/')[-1]}",
        )

    def _attribute_live(self, asset_id: str, image_paths: list,
                        category: Optional[str] = None) -> AttributionResult:
        valid = [p for p in (image_paths or []) if p and Path(p).exists()]
        if not valid:
            raise FileNotFoundError(f"no images for live attribution: {image_paths}")
        if self._client is None:
            self._client, self._deployment = self._build_client()

        # Keep everything in a single user turn (no system role) — a system role
        # with strict formatting tends to trip Azure's jailbreak prompt-shield.
        content = [{"type": "text", "text": self._prompt_for(category)}]
        for p in valid:
            b64 = base64.b64encode(Path(p).read_bytes()).decode()
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"}})
        resp = self._client.chat.completions.create(
            model=self._deployment,
            messages=[{"role": "user", "content": content}],
            temperature=0.2,
            max_tokens=600,
        )
        text = resp.choices[0].message.content or ""
        parsed = _extract_json(text)
        return AttributionResult(
            asset_id=asset_id,
            mass_kg=PropertyRange(**parsed["mass_kg"]),
            static_friction=PropertyRange(**parsed["static_friction"]),
            restitution=PropertyRange(**parsed["restitution"]),
            reasoning=parsed.get("reasoning", text[:200]),
            backend=f"live:{self.provider}",
        )

    def _build_client(self):
        """Build the OpenAI/Azure client; return (client, model_or_deployment)."""
        if self.provider == "azure":
            from openai import AzureOpenAI
            client = AzureOpenAI(
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", AZURE_ENDPOINT_DEFAULT),
                api_version=os.environ.get("AZURE_OPENAI_API_VERSION", AZURE_API_VERSION_DEFAULT),
            )
            # On Azure the `model` arg is the *deployment* name. Accept either
            # AZURE_OPENAI_DEPLOYMENT or AZURE_OPENAI_MODEL (your convention).
            deployment = (os.environ.get("AZURE_OPENAI_DEPLOYMENT")
                          or os.environ.get("AZURE_OPENAI_MODEL")
                          or AZURE_DEPLOYMENT_DEFAULT)
            return client, deployment
        from openai import OpenAI
        client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL") or None)
        return client, self.model

    def _attribute_mock(self, asset_id: str, mass_hint: Optional[float]) -> AttributionResult:
        """Deterministic offline attribution seeded by asset id.

        Mass is anchored to `mass_hint` (geometry x density) when provided, else
        drawn from a log-uniform prior; friction/restitution are pseudo-random
        but stable per asset.
        """
        rng = np.random.default_rng(self.seed + _hash_int(asset_id))
        if mass_hint is not None and mass_hint > 0:
            mean_mass = float(mass_hint)
        else:
            mean_mass = float(np.exp(rng.uniform(np.log(0.05), np.log(10.0))))
        fr = float(rng.uniform(0.2, 1.1))
        re_ = float(rng.uniform(0.05, 0.7))
        return AttributionResult(
            asset_id=asset_id,
            mass_kg=PropertyRange(mean_mass, mean_mass * 0.6, mean_mass * 1.6),
            static_friction=PropertyRange(fr, max(0.05, fr - 0.2), fr + 0.2),
            restitution=PropertyRange(re_, max(0.0, re_ - 0.15), re_ + 0.15),
            reasoning="mock (offline) attribution",
            backend="mock",
        )

    def _clamp(self, r: AttributionResult) -> AttributionResult:
        r.mass_kg = r.mass_kg.clamp(*MASS_BOUNDS)
        r.static_friction = r.static_friction.clamp(*FRICTION_BOUNDS)
        r.restitution = r.restitution.clamp(*RESTITUTION_BOUNDS)
        return r


# ==========================================================================
# helpers
# ==========================================================================
def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def _hash_int(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


def _extract_json(text: str) -> dict:
    """Pull the last JSON object out of an LLM response."""
    # Prefer the last {...} block (the schema instructs JSON comes last).
    matches = list(re.finditer(r"\{(?:[^{}]|\{[^{}]*\})*\}", text, re.DOTALL))
    for m in reversed(matches):
        try:
            d = json.loads(m.group(0))
            if "mass_kg" in d:
                return d
        except json.JSONDecodeError:
            continue
    raise ValueError(f"No parseable physics JSON in response: {text[:200]!r}")
