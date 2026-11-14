"""Reliability hints for regions where the model does not apply or data are lost.

Pixel-level flags (bit field) plus an easy-to-use mask name:

* ``SATURATED``   : observed pixels at the sensor ceiling - information lost;
* ``BLACK_FLOOR`` : signal below the read-noise floor (under the inverse GAT
                    bound) - denoise output is a prior-driven estimate;
* ``MODEL_BOUND`` : parameters outside their physically valid range, e.g. a
                    negative estimated shot coefficient;
* ``OUT_OF_RANGE``: non-finite or negative inputs that had to be clamped.
"""
from __future__ import annotations

import numpy as np

from .noise import NoiseParams

_LUMA = np.array([0.2126, 0.7152, 0.0722])

SATURATED = 1
BLACK_FLOOR = 2
MODEL_BOUND = 4
OUT_OF_RANGE = 8

FLAG_NAMES = {SATURATED: "saturated", BLACK_FLOOR: "black_floor",
              MODEL_BOUND: "model_bound", OUT_OF_RANGE: "out_of_range"}


def build_reliability(rgb_linear_in: np.ndarray, rgb_linear_den: np.ndarray,
                      params: NoiseParams, sat_threshold: float = 0.995
                      ) -> tuple[np.ndarray, dict]:
    """Return a per-pixel flag image and region summaries."""
    obs = np.asarray(rgb_linear_in, dtype=np.float64)
    den = np.asarray(rgb_linear_den, dtype=np.float64)
    flags = np.zeros(obs.shape[:2], dtype=np.uint8)

    oor = ~np.isfinite(obs) | (obs < 0.0)
    flags[np.any(oor, axis=-1)] |= OUT_OF_RANGE

    sat = np.any(obs >= sat_threshold, axis=-1)
    flags[sat] |= SATURATED

    # Read-noise floor in linear units: any channel whose observed value is
    # below roughly one read-noise standard deviation carries no recoverable
    # signal, so the denoised output there is prior-driven rather than data.
    floor = max(float(params.sigma_r), 1.0e-9)
    below_floor = np.all(obs < floor, axis=-1)
    flags[below_floor] |= BLACK_FLOOR

    if params.alpha < 0.0 or params.sigma_r < 0.0 or not np.isfinite(params.alpha):
        flags |= MODEL_BOUND

    summary = {}
    for bit, name in FLAG_NAMES.items():
        summary[name] = float(np.mean((flags & bit) > 0))
    summary["reliable_fraction"] = float(np.mean(flags == 0))
    # Residual diagnostics in linear units (before lifting, so gain does not
    # hide the actual denoising error).
    residual = den - np.nan_to_num(obs, nan=0.0)
    summary["residual_rms_linear"] = float(np.sqrt(np.mean(residual ** 2)))
    summary["residual_mean_linear"] = float(residual.mean())
    return flags, summary


def flags_to_rgb(flags: np.ndarray) -> np.ndarray:
    """Render the flag field as an 8-bit colour overlay for the UI/exports."""
    rgb = np.full((*flags.shape, 3), 255, dtype=np.uint8)  # reliable = white
    rgb[(flags & BLACK_FLOOR) > 0] = (90, 90, 90)
    rgb[(flags & MODEL_BOUND) > 0] = (0, 160, 255)
    rgb[(flags & SATURATED) > 0] = (255, 0, 0)
    rgb[(flags & OUT_OF_RANGE) > 0] = (255, 0, 255)
    return rgb


def restored_anomalies(restored: np.ndarray, observed: np.ndarray,
                       dark_luma: float = 0.25) -> dict:
    """Describe artifacts *in the restored image*, which the input flags miss.

    The existing flags inspect the observation (is a pixel saturated? below the
    read floor?), so a badly mis-declared calibration can produce a visibly
    corrupted restoration while ``reliable_fraction`` stays high.  This looks at
    the output instead and returns both the numbers and a list of plain-language
    warnings, without altering a single pixel.

    Metrics:
    * ``speckle_fraction`` - dark pixels with a large R-G-B spread (colour
      speckling, the classic sign of an over-aggressive gain on noise);
    * ``clip_fraction`` - output pixels pinned at 0 or 1 after lifting;
    * ``dark_residual_rms`` - RMS(restored - observed) over dark pixels, i.e.
      how far the lift moved the noisy shadow away from the measurement.
    """
    out = np.asarray(restored, dtype=np.float64)
    obs = np.asarray(observed, dtype=np.float64)
    luma = np.einsum("...c,c->...", out, _LUMA)
    dark = luma < dark_luma
    spread = out.max(axis=-1) - out.min(axis=-1)
    speckle = float(np.mean((spread > 0.15) & dark)) if dark.any() else 0.0
    clip = float(np.mean((out >= 0.999) | (out <= 0.0)))
    if dark.any():
        dark_res = float(np.sqrt(np.mean((out[dark] - obs[dark]) ** 2)))
    else:
        dark_res = 0.0

    warnings = []
    if speckle > 0.02:
        warnings.append(
            f"恢复图像暗部彩色斑点 {speckle:.1%}：噪声被放大，"
            "可能是标定 α/σ_r 偏高或强度过大")
    if clip > 0.05:
        warnings.append(f"恢复图像 {clip:.1%} 像素触顶/触底：提升增益可能过高")
    if dark_res > 0.10:
        warnings.append(
            f"暗部残差 RMS {dark_res:.3f} 偏大：恢复结果与观测偏离，"
            "请核对噪声标定或降低强度")
    return {"speckle_fraction": speckle, "clip_fraction": clip,
            "dark_residual_rms": dark_res, "warnings": warnings}
