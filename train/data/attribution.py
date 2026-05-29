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
    ):
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
        provider = self._detect_provider()
        if backend == "live":
            if provider is None:
                raise RuntimeError(
                    "backend='live' but no credentials found. Set AZURE_OPENAI_API_KEY"
                    " + AZURE_OPENAI_ENDPOINT (Azure) or OPENAI_API_KEY (OpenAI).")
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
    ) -> AttributionResult:
        """Return cached / freshly-computed physical attribution for one asset.

        `image_paths` is a list of multi-view renders fed to GPT-4o; `image_path`
        is accepted for backwards compatibility (single view).
        """
        if image_paths is None:
            image_paths = [image_path] if image_path else []

        if asset_id in self._store:
            try:
                return AttributionResult.from_dict(self._store[asset_id])
            except Exception:
                logger.warning(f"Corrupt entry for {asset_id} in store, recomputing")

        if self.backend == "live":
            try:
                result = self._attribute_live(asset_id, image_paths)
            except Exception as e:
                logger.error(f"GPT-4o attribution failed for {asset_id}: {e}; "
                             "falling back to mock")
                result = self._attribute_mock(asset_id, mass_hint)
        else:
            result = self._attribute_mock(asset_id, mass_hint)

        result = self._clamp(result)
        self._store[asset_id] = result.to_dict()
        self._save()
        return result

    def _save(self):
        """Write the whole store back to the single json file (atomic)."""
        tmp = self.store_path.with_name(self.store_path.name + ".tmp")
        tmp.write_text(json.dumps(self._store, indent=2))
        tmp.replace(self.store_path)

    # ----- backends ------------------------------------------------------
    def _attribute_live(self, asset_id: str, image_paths: list) -> AttributionResult:
        valid = [p for p in (image_paths or []) if p and Path(p).exists()]
        if not valid:
            raise FileNotFoundError(f"no images for live attribution: {image_paths}")
        if self._client is None:
            self._client, self._deployment = self._build_client()

        # Keep everything in a single user turn (no system role) — a system role
        # with strict formatting tends to trip Azure's jailbreak prompt-shield.
        content = [{"type": "text", "text": COT_SYSTEM_PROMPT}]
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
