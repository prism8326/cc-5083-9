"""Estimation of Poisson-Gaussian noise parameters when no calibration is given.

The estimator works on flat image patches: for a patch whose underlying signal
is constant, the sample mean m and unbiased sample variance v satisfy

        E[v] = alpha * m + sigma_r^2.

A robust Theil-Sen regression of v against m therefore yields the
signal-dependent coefficient alpha (slope) and the fixed-bias variance
sigma_r^2 (intercept).  Patches crossing edges or containing clipped pixels
are rejected so textures are not mistaken for noise.

Scale invariance
----------------
Patches are always measured in *physical pixels* with a fixed window size and
the parameters are expressed in physical linear-light units (0..1), never as
fractions of image dimensions.  Re-sizing an image does not change the noise
level, and the estimator returns the same parameters within sampling error.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from .noise import NoiseParams

BLOCK_SIZE = 8          # physical pixels, fixed regardless of image size
BLOCK_STRIDE = 4
MIN_SAMPLES = 24        # below this the estimate is flagged unreliable


@dataclass
class EstimationInfo:
    alpha: float
    sigma_r: float
    n_blocks: int
    n_used: int
    per_channel: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    method: str = "theilsen"

    def as_dict(self) -> dict:
        return {"alpha": float(self.alpha), "sigma_r": float(self.sigma_r),
                "n_blocks": int(self.n_blocks), "n_used": int(self.n_used),
                "per_channel": self.per_channel, "warnings": self.warnings,
                "method": self.method,
                "block_size": BLOCK_SIZE, "block_stride": BLOCK_STRIDE}


def _patch_stats(plane: np.ndarray, block: int, stride: int,
                 sat_hi: float, sat_lo: float):
    """Means / variances / gradient activity of non-overlapping-ish patches."""
    h, w = plane.shape
    ny = max(0, (h - block) // stride + 1)
    nx = max(0, (w - block) // stride + 1)
    if ny < 2 or nx < 2:
        return None
    # Build patch views with sliding windows.
    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(plane, (block, block))[::stride, ::stride]
    n = block * block
    means = windows.mean(axis=(-2, -1))
    # Unbiased sample variance.
    var = windows.var(axis=(-2, -1), ddof=1)
    # Mean absolute gradient inside each patch: edges give large values.
    gy = np.abs(windows[..., 1:, :] - windows[..., :-1, :]).mean(axis=(-2, -1))
    gx = np.abs(windows[..., :, 1:] - windows[..., :, :-1]).mean(axis=(-2, -1))
    grad = gy + gx
    sat = ((windows >= sat_hi).any(axis=(-2, -1))
           | (windows <= sat_lo).all(axis=(-2, -1)))
    return means, var, grad, sat, n


def _estimate_channel(plane: np.ndarray) -> tuple[float, float, int, int, list]:
    warnings_: list[str] = []
    block = BLOCK_SIZE
    stride = BLOCK_STRIDE
    sat_hi = float(np.nextafter(1.0, 0.0))
    sat_lo = 0.0
    stats_ = _patch_stats(plane, block, stride, sat_hi, sat_lo)
    if stats_ is None:
        # Very small image: shrink the patch window once.
        block, stride = 4, 2
        stats_ = _patch_stats(plane, block, stride, sat_hi, sat_lo)
        if stats_ is None:
            return np.nan, np.nan, 0, 0, ["image too small for patch statistics"]
    means, var, grad, sat, n = stats_
    n_blocks = means.size

    flat = (~sat) & (grad <= max(1.5 * np.median(grad), 1.0e-6))
    m = means[flat]
    v = var[flat]
    if m.size < MIN_SAMPLES:
        warnings_.append(f"only {m.size} flat patches (need >= {MIN_SAMPLES})")
        if m.size < 4:
            return np.nan, np.nan, n_blocks, int(m.size), warnings_

    # Theil-Sen is insensitive to residual texture outliers and to a minority
    # of misclassified patches.
    if m.size > 1:
        res = stats.theilslopes(v, m)
        alpha = float(res.slope)
        s2 = float(res.intercept)
    else:
        alpha, s2 = np.nan, np.nan

    # Physical constraints: both terms must be non-negative.  A negative slope
    # means no detectable signal dependence; fall back to a constant model.
    if not np.isfinite(alpha) or alpha < 0.0:
        warnings_.append("non-positive shot coefficient; using constant-noise model")
        alpha = 0.0
        s2 = float(np.median(v))
    if not np.isfinite(s2) or s2 < 0.0:
        s2 = 0.0
    return alpha, float(np.sqrt(max(s2, 0.0))), n_blocks, int(m.size), warnings_


def estimate_noise(rgb_linear: np.ndarray) -> tuple[NoiseParams, EstimationInfo]:
    """Estimate (alpha, sigma_r) from a linear-light sRGB-range float image.

    Returns the parameters plus diagnostics.  When the data contain too little
    information (e.g. a pure black frame) conservative defaults are returned
    and the warning list says so explicitly.
    """
    img = np.asarray(rgb_linear, dtype=np.float64)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("estimate_noise expects an HxWx3 linear RGB image")

    per_channel = []
    alphas, sigmas = [], []
    total_blocks = 0
    warnings: list[str] = []
    for c, name in enumerate(("R", "G", "B")):
        a, s, nb, used, w = _estimate_channel(img[..., c])
        per_channel.append({"channel": name, "alpha": None if not np.isfinite(a) else a,
                            "sigma_r": None if not np.isfinite(s) else s,
                            "n_blocks": nb, "n_used": used, "warnings": w})
        total_blocks = max(total_blocks, nb)
        warnings.extend(f"{name}: {msg}" for msg in w)
        if np.isfinite(a):
            alphas.append(a)
        if np.isfinite(s) and s > 0.0:
            sigmas.append(s)

    if alphas:
        alpha = float(np.median(alphas))
    else:
        alpha = 0.0
        warnings.append("no shot-noise dependence detectable; alpha set to 0")
    if sigmas:
        sigma_r = float(np.median(sigmas))
    else:
        # Conservative read-noise floor: 1/2 LSB of the 8-bit input in linear
        # units of a mid-tone signal.  The range is documented in the output.
        sigma_r = 1.0 / 510.0
        warnings.append(
            "read noise not measurable; default sigma_r=1/510 (~1/2 code value)")

    info = EstimationInfo(alpha=alpha, sigma_r=sigma_r,
                          n_blocks=total_blocks,
                          n_used=int(np.median([pc["n_used"] for pc in per_channel])),
                          per_channel=per_channel, warnings=warnings)
    return NoiseParams(alpha=alpha, sigma_r=sigma_r, source="estimate"), info
