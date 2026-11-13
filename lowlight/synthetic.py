"""Synthetic observation generation for offline evaluation.

Everything here is test-only: the runtime pipeline never imports this module
and never sees a clean reference.  A known clean linear-light scene is exposed
at different exposure factors and corrupted with the same affine
Poisson-Gaussian model the pipeline claims to invert:

    y = alpha * Poisson(x / alpha) + N(0, sigma_r^2),  then clipped to [0, 1].
"""
from __future__ import annotations

import numpy as np

from .color import srgb_to_linear, linear_to_srgb


def make_clean_scene(size: int = 256, seed: int = 0) -> np.ndarray:
    """Build a clean linear-light scene with thin lines, dark tones and colour.

    Components: smooth colour gradient, a dark grey panel with fine bright
    lines, pure-black and single-colour patches (edge cases).
    """
    rng = np.random.default_rng(seed)
    h = w = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    # Smooth low-frequency colour gradient (subtle hue differences).
    r = 0.10 + 0.06 * (xx / w)
    g = 0.08 + 0.07 * (yy / h)
    b = 0.12 + 0.05 * (1.0 - xx / w)
    scene = np.stack([r, g, b], axis=-1)

    # Dark panel with thin 1-px lines at several orientations.
    panel = (slice(h // 4, 3 * h // 4), slice(w // 4, 3 * w // 4))
    scene[panel] *= 0.35
    ph = panel[0].stop - panel[0].start
    pw = panel[1].stop - panel[1].start
    for k in range(0, pw, 14):
        scene[panel[0].start, panel[1].start + k] = [0.5, 0.55, 0.45]
        scene[panel[0].start + 1, panel[1].start + k] = [0.45, 0.5, 0.4]
    for k in range(0, ph, 11):
        scene[panel[0].start + k, panel[1].stop - 1] = [0.4, 0.5, 0.6]
    # A thin diagonal line.
    for k in range(0, min(ph, pw) - 2, 3):
        scene[panel[0].start + k, panel[1].start + k] = [0.6, 0.5, 0.3]

    # Pure black patch (zero signal) and single-colour patches.
    scene[0:size // 10, 0:size // 10] = 0.0
    scene[0:size // 10, size // 10:size // 5] = [0.35, 0.02, 0.02]
    scene[0:size // 10, size // 5:3 * size // 10] = [0.02, 0.30, 0.02]
    scene[0:size // 10, 3 * size // 10:2 * size // 5] = [0.02, 0.02, 0.32]
    # Near-black but non-zero patch to probe shadow recovery.
    scene[size - size // 12:, :size // 8] = 0.004
    # Uniform flat grey calibration area (no lines/gradient) for variance
    # tests, placed outside every patterned panel.
    scene[size // 2 + 16:size // 2 + 48, 3 * size // 4 + 8:3 * size // 4 + 40] = 0.06
    return np.clip(scene, 0.0, 1.0)


def expose(clean_linear: np.ndarray, exposure: float) -> np.ndarray:
    """Apply an exposure factor (fraction of photons captured)."""
    return np.clip(clean_linear * float(exposure), 0.0, 1.0)


def add_noise(clean_linear: np.ndarray, alpha: float, sigma_r: float,
              seed: int = 1, clip: bool = True) -> np.ndarray:
    """Sample an affine Poisson-Gaussian observation of a linear scene."""
    rng = np.random.default_rng(seed)
    x = np.asarray(clean_linear, dtype=np.float64)
    if alpha > 0.0:
        lam = np.maximum(x / alpha, 0.0)
        shot = alpha * rng.poisson(lam)
    else:
        shot = x
    read = rng.normal(0.0, sigma_r, size=x.shape)
    y = shot + read
    if clip:
        y = np.clip(y, 0.0, 1.0)
    return y


def make_observation(size: int = 256, exposure: float = 0.25,
                     alpha: float = 0.004, sigma_r: float = 0.006,
                     seed: int = 1) -> dict:
    """Full pipeline test fixture: clean scene + noisy sRGB observation."""
    clean = make_clean_scene(size=size, seed=seed)
    exposed = expose(clean, exposure)
    noisy_linear = add_noise(exposed, alpha=alpha, sigma_r=sigma_r, seed=seed + 100)
    return {"clean_linear": clean,
            "exposed_linear": exposed,
            "observed_linear": noisy_linear,
            "observed_srgb": np.clip(linear_to_srgb(noisy_linear), 0, 1),
            "true_alpha": alpha, "true_sigma_r": sigma_r,
            "exposure": exposure}


def make_pure_black(size: int = 128, sigma_r: float = 0.006,
                    seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(0.0, sigma_r, (size, size, 3)), 0.0, 1.0)


def make_flat_colour(size: int = 128, rgb_linear=(0.08, 0.03, 0.02),
                     alpha: float = 0.004, sigma_r: float = 0.006,
                     seed: int = 9) -> np.ndarray:
    clean = np.zeros((size, size, 3))
    clean[..., 0], clean[..., 1], clean[..., 2] = rgb_linear
    return add_noise(clean, alpha, sigma_r, seed=seed)
