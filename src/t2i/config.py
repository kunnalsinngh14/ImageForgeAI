"""
t2i.config — Configuration loading, validation, and resolution.

All pipeline behaviour is driven by YAML config files. This module loads a YAML file,
validates every field with Pydantic, and returns a fully resolved ``T2IConfig`` instance.
Unknown keys raise a ``ValidationError`` so config drift is caught early.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, field_validator, model_validator


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class ModelConfig(BaseModel):
    """Identifies the base model and pins the exact revision."""

    base: str
    revision: str


class ResizeConfig(BaseModel):
    """Parameters for the resize / resolution-optimization step."""

    mode: Literal["none", "lanczos_down", "bicubic_up", "adaptive"] = "none"
    unsharp_radius: float = 1.0
    unsharp_amount: float = 0.5


class ColorConfig(BaseModel):
    """Parameters for CLAHE, white-balance, and vibrance."""

    clahe_clip: float = 2.0
    clahe_tile: list[int] = [8, 8]
    white_balance: Literal["gray_world", "percentile", "none"] = "gray_world"
    vibrance: float = 0.0

    @field_validator("clahe_tile")
    @classmethod
    def _two_ints(cls, v: list[int]) -> list[int]:
        if len(v) != 2:
            raise ValueError("clahe_tile must have exactly 2 elements")
        return v


class EdgeConfig(BaseModel):
    """Parameters for the guided-filter edge-aware post-processing step."""

    guided_radius: int = 8
    guided_eps: float = 0.01
    detail_scale: float = 0.3


class ArtifactsConfig(BaseModel):
    """Parameters for FFT-based checkerboard and over-saturation detection."""

    fft_peak_threshold: float = 6.0
    local_var_threshold: float = 10.0
    correct: bool = False


class CVConfig(BaseModel):
    """Aggregates all classical-CV post-processing sub-configs."""

    resize: ResizeConfig = ResizeConfig()
    color: ColorConfig = ColorConfig()
    edge: EdgeConfig = EdgeConfig()
    artifacts: ArtifactsConfig = ArtifactsConfig()


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


class T2IConfig(BaseModel):
    """
    Complete configuration for one generation or benchmark run.

    Every field is used by the pipeline. Unknown keys are rejected (Pydantic
    ``model_config = {"extra": "forbid"}``).
    """

    model_config = {"extra": "forbid"}

    name: str
    model: ModelConfig
    precision: Literal["fp32", "fp16", "int8_dynamic", "int8_onnx"]
    scheduler: Literal["ddim", "dpm_solver_pp", "lcm"]
    steps: int
    guidance_scale: float
    resolution: list[int]
    variant: Literal["full_unet", "block_pruned", "dreamshaper_v8"]
    lcm_lora: bool
    quantization: Literal["none", "fp16", "int8_dynamic", "int8_onnx"]
    attention: Literal["default", "sdpa", "xformers"]
    vae_tiling: bool
    channels_last: bool = False
    classical_postprocess: bool
    cv: CVConfig = CVConfig()
    seed_base: int

    @field_validator("resolution")
    @classmethod
    def _two_positive_ints(cls, v: list[int]) -> list[int]:
        if len(v) != 2 or any(x <= 0 for x in v):
            raise ValueError("resolution must be [width, height] with positive values")
        return v

    @field_validator("steps")
    @classmethod
    def _positive_steps(cls, v: int) -> int:
        if v < 1:
            raise ValueError("steps must be >= 1")
        return v

    @model_validator(mode="after")
    def _lcm_guidance_check(self) -> "T2IConfig":
        if self.lcm_lora and self.guidance_scale > 2.0:
            # Warn but do not fail — the user may intentionally explore this.
            import warnings

            warnings.warn(
                f"LCM-LoRA is enabled but guidance_scale={self.guidance_scale} is high. "
                "LCM typically uses guidance_scale <= 1.0.",
                UserWarning,
                stacklevel=2,
            )
        return self

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def config_hash(self) -> str:
        """Return a short SHA-256 hash of the serialised config (first 12 hex chars)."""
        blob = json.dumps(self.model_dump(), sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    @property
    def height(self) -> int:
        """Image height in pixels."""
        return self.resolution[1]

    @property
    def width(self) -> int:
        """Image width in pixels."""
        return self.resolution[0]

    @property
    def latent_height(self) -> int:
        """Latent height (height / 8 for SD v1.5 VAE factor 8)."""
        return self.height // 8

    @property
    def latent_width(self) -> int:
        """Latent width (width / 8 for SD v1.5 VAE factor 8)."""
        return self.width // 8


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> T2IConfig:
    """
    Load and validate a YAML config file.

    Parameters
    ----------
    path:
        Path to the YAML config file.

    Returns
    -------
    T2IConfig
        Fully validated configuration object.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    pydantic.ValidationError
        If any field is invalid or an unknown key is present.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)
    return T2IConfig.model_validate(raw)
