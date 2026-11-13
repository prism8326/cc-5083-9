"""End-to-end low-light restoration pipeline and image I/O.

Stages (all in linear-light space after decoding):

    input image
      -> decode to linear RGB
      -> noise calibration (provided parameters, or robust estimation)
      -> GAT variance stabilization
      -> edge-preserving denoise (unit-variance domain, size-invariant)
      -> exact unbiased inverse GAT
      -> constrained shadow lift (bounded gain, per-channel colour preserved)
      -> reliability map + residual statistics
      -> sRGB encode / export

The clean reference is never part of this path; it is only used by the
offline test harness in ``tests/``.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .calibration import estimate_noise
from .color import srgb_to_linear, encode_display, linear_luma
from .denoise import denoise
from .lifting import LiftParams, lift_shadows
from .noise import NoiseParams
from .reliability import build_reliability, flags_to_rgb

try:
    import cv2
    _HAVE_CV2 = True
except Exception:  # pragma: no cover
    _HAVE_CV2 = False

try:
    from PIL import Image
    _HAVE_PIL = True
except Exception:  # pragma: no cover
    _HAVE_PIL = False


@dataclass
class PipelineConfig:
    strength: float = 1.0            # denoising in stabilized-sigma units
    method: str = "nlm"
    alpha: Optional[float] = None    # provided calibration, linear units
    sigma_r: Optional[float] = None
    input_is_linear: bool = False
    lift: Optional[LiftParams] = None
    sat_threshold: float = 0.995
    max_side: Optional[int] = None   # optional display/processing cap


@dataclass
class PipelineResult:
    rgb_linear_obs: np.ndarray
    rgb_linear_denoised: np.ndarray
    rgb_linear_restored: np.ndarray
    residual_linear: np.ndarray
    flags: np.ndarray
    params: NoiseParams
    config: PipelineConfig
    stats: dict = field(default_factory=dict)

    # Convenience display arrays (8-bit sRGB).
    def display_original(self) -> np.ndarray:
        return encode_display(self.rgb_linear_obs)

    def display_restored(self) -> np.ndarray:
        return encode_display(self.rgb_linear_restored)

    def display_residual(self, gain: float | None = None) -> np.ndarray:
        """Signed residual centred on mid-grey.

        With ``gain=None`` the scale is chosen adaptively from the residual's
        own standard deviation (about +/-3 sigma spans the byte range), so the
        view stays informative on both quiet and noisy frames.
        """
        r = self.residual_linear
        if gain is None:
            sigma = float(np.std(r))
            gain = 0.5 / max(sigma * 3.0, 1.0e-6)
        vis = np.clip(0.5 + r * gain, 0.0, 1.0)
        return (vis * 255.0 + 0.5).astype(np.uint8)

    def display_reliability(self) -> np.ndarray:
        return flags_to_rgb(self.flags)


def read_image(path: str) -> tuple[np.ndarray, dict]:
    """Read an image as float RGB in [0, 1] (sRGB-encoded bytes assumed)."""
    if _HAVE_CV2:
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(path)
        if raw.ndim == 2:
            raw = np.stack([raw] * 3, axis=-1)
        if raw.shape[2] == 4:
            raw = raw[..., :3]
        raw = raw[..., ::-1]  # BGR -> RGB
    elif _HAVE_PIL:
        with Image.open(path) as im:
            raw = np.asarray(im.convert("RGB"))
    else:
        raise RuntimeError("neither OpenCV nor Pillow is available for image I/O")

    max_val = float(np.iinfo(raw.dtype).max) if np.issubdtype(raw.dtype, np.integer) else 1.0
    rgb = raw.astype(np.float64) / max_val
    meta = {"path": os.path.abspath(path), "dtype": str(raw.dtype),
            "max_code_value": max_val, "shape": list(rgb.shape)}
    return rgb, meta


def write_image(path: str, rgb_u8: np.ndarray) -> None:
    if _HAVE_CV2:
        cv2.imwrite(path, rgb_u8[..., ::-1])
    elif _HAVE_PIL:
        Image.fromarray(rgb_u8).save(path)
    else:  # pragma: no cover
        raise RuntimeError("no image writer available")


def run_pipeline(rgb01: np.ndarray, config: PipelineConfig | None = None
                 ) -> PipelineResult:
    """Run the full restoration on an sRGB-range float RGB array (HxWx3)."""
    cfg = config or PipelineConfig()
    rgb01 = np.asarray(rgb01, dtype=np.float64)
    if rgb01.ndim != 3 or rgb01.shape[2] != 3:
        raise ValueError("expected HxWx3 RGB input")
    rgb01 = np.clip(rgb01, 0.0, 1.0)

    obs_linear = rgb01 if cfg.input_is_linear else srgb_to_linear(rgb01)

    warnings: list[str] = []
    if cfg.alpha is not None and cfg.sigma_r is not None:
        if cfg.alpha < 0.0 or cfg.sigma_r < 0.0:
            raise ValueError("calibration parameters must be non-negative")
        params = NoiseParams(float(cfg.alpha), float(cfg.sigma_r),
                             source="calibration")
        estimation = None
    else:
        params, est = estimate_noise(obs_linear)
        estimation = est.as_dict()
        warnings.extend(est.warnings)

    den_linear, den_stats = denoise(obs_linear, params,
                                    strength=cfg.strength, method=cfg.method)
    # Keep slight negatives visible to the reliability stage, then lift.
    flags, rel_summary = build_reliability(obs_linear, den_linear, params,
                                           sat_threshold=cfg.sat_threshold)

    lift_params = cfg.lift or LiftParams()
    restored_linear, lift_stats = lift_shadows(
        np.maximum(den_linear, 0.0), lift_params)

    residual = den_linear - obs_linear
    stats = {
        "noise_params": params.as_dict(),
        "estimation": estimation,
        "denoise": den_stats,
        "lift": lift_stats,
        "reliability": rel_summary,
        "warnings": warnings,
        "color": {
            "mean_rgb_observed_linear": [float(x) for x in obs_linear.reshape(-1, 3).mean(axis=0)],
            "mean_rgb_restored_linear": [float(x) for x in restored_linear.reshape(-1, 3).mean(axis=0)],
        },
        "luma_mean_observed": float(linear_luma(obs_linear).mean()),
        "luma_mean_restored": float(linear_luma(restored_linear).mean()),
    }
    return PipelineResult(obs_linear, den_linear, restored_linear, residual,
                          flags, params, cfg, stats)


def export_results(result: PipelineResult, out_dir: str,
                   base_name: str = "restored",
                   residual_gain: float | None = None) -> dict:
    """Write original/restored/residual/reliability views and a JSON record.

    ``residual_gain=None`` selects adaptive scaling from the residual sigma.
    """
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    files = {
        "original": result.display_original(),
        base_name: result.display_restored(),
        "residual": result.display_residual(gain=residual_gain),
        "reliability": result.display_reliability(),
    }
    for name, img in files.items():
        p = os.path.join(out_dir, f"{name}.png")
        write_image(p, img)
        paths[name] = os.path.abspath(p)

    record = {
        "config": {
            "strength": result.config.strength,
            "method": result.config.method,
            "alpha": result.config.alpha,
            "sigma_r": result.config.sigma_r,
            "input_is_linear": result.config.input_is_linear,
            "sat_threshold": result.config.sat_threshold,
            "lift": result.config.lift.as_dict() if result.config.lift else LiftParams().as_dict(),
        },
        "stats": result.stats,
        "outputs": paths,
    }
    json_path = os.path.join(out_dir, "restoration_record.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    paths["record"] = os.path.abspath(json_path)
    return paths
