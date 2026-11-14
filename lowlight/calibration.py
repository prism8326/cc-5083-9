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

Large images
------------
Patch statistics are computed once into a reusable
:class:`lowlight.stats.PatchStatistics` object (fixed-pixel windows, widened
stride only when the block count would exceed a budget, never a coarser patch
window).  The Theil-Sen slope is located by counting pairwise slopes rather
than materialising the ``O(n^2)`` slope array, so a 4000-patch fit takes
milliseconds instead of gigabytes.  The dense image is never up-cast to a
window-per-pixel 8-D array, which is what made the previous implementation
die on large frames.

``estimate_noise_legacy`` keeps the previous implementation intact for
pixel-error comparison; it is not used by the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats as _scipy_stats

from .noise import NoiseParams
from .stats import (BLOCK_SIZE, BLOCK_STRIDE, MAX_BLOCKS, MIN_SAMPLES,
                    THEIL_BUDGET, ChannelFit, PatchStatistics,
                    compute_patch_statistics, fit_channel, uniform_subsample,
                    theilslopes_fast)


@dataclass
class EstimationInfo:
    alpha: float
    sigma_r: float
    n_blocks: int
    n_used: int
    per_channel: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    method: str = "theilsen-fast"
    approximate: bool = False
    n_regression: int = 0
    patch_stats: dict = field(default_factory=dict)
    patch_compute_s: float = 0.0
    estimate_compute_s: float = 0.0
    cache_hit: bool = False

    def as_dict(self) -> dict:
        return {"alpha": float(self.alpha), "sigma_r": float(self.sigma_r),
                "n_blocks": int(self.n_blocks), "n_used": int(self.n_used),
                "per_channel": self.per_channel, "warnings": self.warnings,
                "method": self.method,
                "block_size": self.patch_stats.get("block_size", BLOCK_SIZE),
                "block_stride": self.patch_stats.get("stride", BLOCK_STRIDE),
                "theil_budget": THEIL_BUDGET,
                "n_regression": int(self.n_regression),
                "approximate": bool(self.approximate),
                "patch_stats": self.patch_stats,
                "patch_compute_s": float(self.patch_compute_s),
                "estimate_compute_s": float(self.estimate_compute_s),
                "cache_hit": bool(self.cache_hit)}


def _aggregate(fits: list[ChannelFit], patch: PatchStatistics,
               approximate: bool) -> tuple[NoiseParams, EstimationInfo]:
    per_channel = []
    alphas, sigmas = [], []
    total_blocks = 0
    warnings: list[str] = []
    n_regression = 0
    for name, fit in zip(("R", "G", "B"), fits):
        per_channel.append({
            "channel": name,
            "alpha": None if not np.isfinite(fit.alpha) else fit.alpha,
            "sigma_r": None if not np.isfinite(fit.sigma_r) else fit.sigma_r,
            "n_blocks": fit.n_blocks, "n_used": fit.n_used,
            "n_regression": fit.n_regression, "warnings": fit.warnings})
        total_blocks = max(total_blocks, fit.n_blocks)
        n_regression = max(n_regression, fit.n_regression)
        warnings.extend(f"{name}: {msg}" for msg in fit.warnings)
        if np.isfinite(fit.alpha):
            alphas.append(fit.alpha)
        if np.isfinite(fit.sigma_r) and fit.sigma_r > 0.0:
            sigmas.append(fit.sigma_r)

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
    n_used = int(np.median([pc["n_used"] for pc in per_channel]))
    info = EstimationInfo(
        alpha=alpha, sigma_r=sigma_r, n_blocks=total_blocks, n_used=n_used,
        per_channel=per_channel, warnings=warnings, approximate=approximate,
        n_regression=n_regression, patch_stats=patch.as_dict())
    return NoiseParams(alpha=alpha, sigma_r=sigma_r, source="estimate"), info


def estimate_noise_from_stats(patch: PatchStatistics,
                              min_samples: int = MIN_SAMPLES,
                              theil_budget: int = THEIL_BUDGET,
                              ) -> tuple[NoiseParams, EstimationInfo]:
    """Fit noise parameters from precomputed patch statistics."""
    fits = [fit_channel(patch.means[c], patch.variances[c],
                        patch.gradients[c], patch.saturated[c],
                        min_samples=min_samples, theil_budget=theil_budget)
            for c in range(3)]
    return _aggregate(fits, patch, patch.approximate)


def estimate_noise(rgb_linear: np.ndarray,
                   return_stats: bool = False,
                   min_samples: int = MIN_SAMPLES,
                   theil_budget: int = THEIL_BUDGET,
                   patch_stats: PatchStatistics | None = None,
                   ) -> tuple[NoiseParams, EstimationInfo]:
    """Estimate (alpha, sigma_r) from a linear-light sRGB-range float image.

    Returns the parameters plus diagnostics.  When the data contain too little
    information (e.g. a pure black frame) conservative defaults are returned
    and the warning list says so explicitly.

    ``patch_stats`` lets a caller (or the pipeline cache) supply statistics
    already computed for this image content, avoiding recomputation.
    """
    from time import perf_counter
    img = np.asarray(rgb_linear, dtype=np.float64)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("estimate_noise expects an HxWx3 linear RGB image")
    if patch_stats is None:
        t0 = perf_counter()
        patch_stats = compute_patch_statistics(img, BLOCK_SIZE, BLOCK_STRIDE)
        patch_s = perf_counter() - t0
    else:
        patch_s = patch_stats.elapsed_s
    t1 = perf_counter()
    params, info = estimate_noise_from_stats(patch_stats, min_samples,
                                             theil_budget)
    info.patch_compute_s = float(patch_s)
    info.estimate_compute_s = float(perf_counter() - t1)
    return params, info


# --------------------------------------------------------------------------
# Legacy implementation (kept only for exact pixel-error A/B comparison).
# --------------------------------------------------------------------------
def _patch_stats_legacy(plane: np.ndarray, block: int, stride: int,
                        sat_hi: float, sat_lo: float):
    h, w = plane.shape
    ny = max(0, (h - block) // stride + 1)
    nx = max(0, (w - block) // stride + 1)
    if ny < 2 or nx < 2:
        return None
    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(plane, (block, block))[::stride, ::stride]
    n = block * block
    means = windows.mean(axis=(-2, -1))
    var = windows.var(axis=(-2, -1), ddof=1)
    gy = np.abs(windows[..., 1:, :] - windows[..., :-1, :]).mean(axis=(-2, -1))
    gx = np.abs(windows[..., :, 1:] - windows[..., :, :-1]).mean(axis=(-2, -1))
    grad = gy + gx
    sat = ((windows >= sat_hi).any(axis=(-2, -1))
           | (windows <= sat_lo).all(axis=(-2, -1)))
    return means, var, grad, sat, n


def _estimate_channel_legacy(plane: np.ndarray):
    warnings_: list[str] = []
    block = BLOCK_SIZE
    stride = BLOCK_STRIDE
    sat_hi = float(np.nextafter(1.0, 0.0))
    sat_lo = 0.0
    stats_ = _patch_stats_legacy(plane, block, stride, sat_hi, sat_lo)
    if stats_ is None:
        block, stride = 4, 2
        stats_ = _patch_stats_legacy(plane, block, stride, sat_hi, sat_lo)
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
    if m.size > 1:
        res = _scipy_stats.theilslopes(v, m)
        alpha = float(res.slope)
        s2 = float(res.intercept)
    else:
        alpha, s2 = np.nan, np.nan
    if not np.isfinite(alpha) or alpha < 0.0:
        warnings_.append("non-positive shot coefficient; using constant-noise model")
        alpha = 0.0
        s2 = float(np.median(v))
    if not np.isfinite(s2) or s2 < 0.0:
        s2 = 0.0
    return alpha, float(np.sqrt(max(s2, 0.0))), n_blocks, int(m.size), warnings_


def estimate_noise_legacy(rgb_linear: np.ndarray) -> tuple[NoiseParams, EstimationInfo]:
    """Previous (uncached, O(n^2) Theil-Sen) estimator, for comparison only."""
    img = np.asarray(rgb_linear, dtype=np.float64)
    per_channel = []
    alphas, sigmas = [], []
    total_blocks = 0
    warnings: list[str] = []
    for c, name in enumerate(("R", "G", "B")):
        a, s, nb, used, w = _estimate_channel_legacy(img[..., c])
        per_channel.append({"channel": name,
                            "alpha": None if not np.isfinite(a) else a,
                            "sigma_r": None if not np.isfinite(s) else s,
                            "n_blocks": nb, "n_used": used, "warnings": w})
        total_blocks = max(total_blocks, nb)
        warnings.extend(f"{name}: {msg}" for msg in w)
        if np.isfinite(a):
            alphas.append(a)
        if np.isfinite(s) and s > 0.0:
            sigmas.append(s)
    alpha = float(np.median(alphas)) if alphas else 0.0
    if not alphas:
        warnings.append("no shot-noise dependence detectable; alpha set to 0")
    if sigmas:
        sigma_r = float(np.median(sigmas))
    else:
        sigma_r = 1.0 / 510.0
        warnings.append(
            "read noise not measurable; default sigma_r=1/510 (~1/2 code value)")
    info = EstimationInfo(alpha=alpha, sigma_r=sigma_r,
                          n_blocks=total_blocks,
                          n_used=int(np.median([pc["n_used"] for pc in per_channel])),
                          per_channel=per_channel, warnings=warnings,
                          method="theilsen-legacy")
    return NoiseParams(alpha=alpha, sigma_r=sigma_r, source="estimate"), info
