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
