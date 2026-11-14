"""Workload generator for the large-photo optimization benchmarks.

Three scenarios are generated separately because they stress different parts
of the pipeline and can fail independently:

``uniform_dark``
    Nearly flat, very dim scene (a smooth low-level gradient plus faint low
    frequency structure).  This is the easy case for denoising; a fast result
    here alone must not be read as an overall speedup.

``dense_texture``
    High-frequency foliage/fabric-like texture over the full frame with a few
    thin bright strokes.  NLM has to evaluate many candidate patches, so this
    is the worst case for the denoise stage and for any tiling overhead.

``noise_step``
    A smooth scene whose noise level changes abruptly along a diagonal and at
    a vertical seam (alpha and sigma_r jump).  Calibration blocks straddling
    the change are disproportional, and a naive single global threshold either
    leaves one half noisy or smears the other; this scenario checks that the
    optimization does not hide a semantic change at the discontinuity.

Every generator is deterministic given ``seed``; the returned dict carries the
clean linear scene, the exposed reference, the noisy observation (linear and
sRGB) and the ground-truth noise parameters per region, so the measurement
harness can compute pixel error and thin-line residuals without the runtime
pipeline ever seeing a clean reference.
"""
from __future__ import annotations

import numpy as np

from .color import linear_to_srgb


def _thin_strokes(scene: np.ndarray, rng: np.random.Generator,
                  n: int, level: float) -> np.ndarray:
    """Draw ``n`` 1-px bright strokes at mixed orientations."""
    h, w = scene.shape[:2]
    for _ in range(n):
        y = int(rng.integers(2, h - 2))
        x = int(rng.integers(2, w - 2))
        length = int(rng.integers(12, max(13, min(h, w) // 4)))
        theta = float(rng.uniform(0.0, np.pi))
        for k in range(length):
            yy = int(round(y + k * np.sin(theta)))
            xx = int(round(x + k * np.cos(theta)))
            if 0 <= yy < h and 0 <= xx < w:
                scene[yy, xx] = level
    return scene


def make_uniform_dark(size: int, seed: int = 0) -> np.ndarray:
    """Smooth near-flat dim scene: no sharp edges, little high frequency."""
    rng = np.random.default_rng(seed)
    h = w = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    base = 0.010 + 0.006 * (xx / w) + 0.004 * (yy / h)
    # Very faint low-frequency veil; deliberately no texture.
    veil = 0.002 * np.sin(2.0 * np.pi * xx / (w * 0.9)) * np.cos(2.0 * np.pi * yy / (h * 0.8))
    luma = base + veil
    scene = np.stack([luma * 0.95, luma, luma * 1.05], axis=-1)
    # A single faint calibration square, still flat.
    s = size // 8
    scene[size // 2:size // 2 + s, size // 2:size // 2 + s] *= 2.0
    return np.clip(scene, 0.0, 1.0)


def make_dense_texture(size: int, seed: int = 0) -> np.ndarray:
    """Full-frame dense high-frequency texture plus thin bright strokes."""
    rng = np.random.default_rng(seed)
    h = w = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    # Multi-scale deterministic texture (sum of incommensurate gratings) so
    # results are reproducible and free of spectral holes.
    luma = np.full((h, w), 0.05)
    for k, (fx, fy, amp) in enumerate([(0.37, 0.11, 0.030), (0.53, -0.29, 0.024),
                                       (0.91, 0.71, 0.018), (0.17, 0.63, 0.020)]):
        luma += amp * np.sin(2.0 * np.pi * (fx * xx / 16.0 + fy * yy / 16.0) + k)
    # Hash-like micro texture (deterministic, uncorrelated at 1 px).
    micro = rng.standard_normal((h, w))
    luma = luma + 0.012 * np.tanh(micro * 1.7)
    luma = np.clip(luma, 0.005, 0.95)
    scene = np.stack([luma * 0.9 + 0.004, luma, luma * 1.1], axis=-1)
    scene = _thin_strokes(np.clip(scene, 0.0, 1.0), rng, n=size // 16, level=0.72)
    return np.clip(scene, 0.0, 1.0)


def make_noise_step(size: int, seed: int = 0) -> np.ndarray:
    """Smooth scene with two abrupt noise regions (no shot/read step in scene)."""
    rng = np.random.default_rng(seed)
    h = w = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    luma = 0.04 + 0.03 * (xx / w) + 0.02 * np.sin(2.0 * np.pi * yy / (h * 0.7)) * 0.5
    # Flat calibration patches on both sides of the seam.
    luma[h // 8:h // 8 + size // 10, w // 8:w // 8 + size // 10] = 0.05
    luma[h // 8:h // 8 + size // 10, 5 * w // 8:5 * w // 8 + size // 10] = 0.05
    scene = np.stack([luma, luma * 0.97, luma * 1.03], axis=-1)
    return np.clip(scene, 0.0, 1.0)


_NOISE_PROFILES = {
    "uniform_dark": (0.010, 0.004),     # low signal, low-ish shot noise
    "dense_texture": (0.006, 0.008),
    "noise_step": (0.006, 0.010),       # mid region; step applied below
}


def make_workload(kind: str, size: int, seed: int = 0,
                  exposure: float = 0.6) -> dict:
    """Build one benchmark workload with ground-truth parameters.

    For ``noise_step`` the frame is split: left/upper region uses the base
    parameters, right/lower uses a stronger pair; the masks are returned so the
    harness can score each region separately.
    """
    if kind not in _NOISE_PROFILES:
        raise ValueError(f"unknown workload {kind!r}")
    if kind == "uniform_dark":
        clean = make_uniform_dark(size, seed)
    elif kind == "dense_texture":
        clean = make_dense_texture(size, seed)
    else:
        clean = make_noise_step(size, seed)

    rng = np.random.default_rng(seed + 1000)
    exposed = np.clip(clean * exposure, 0.0, 1.0)
    alpha0, sigma0 = _NOISE_PROFILES[kind]

    if kind == "noise_step":
        h, w = size, size
        yy, xx = np.mgrid[0:h, 0:w]
        strong = (xx + yy) >= int(1.05 * size)
        alpha = np.where(strong, alpha0 * 4.0, alpha0)
        sigma = np.where(strong, sigma0 * 2.5, sigma0)
        lam = np.maximum(exposed / alpha[..., None], 0.0)
        shot = alpha[..., None] * rng.poisson(lam)
        read = rng.normal(0.0, 1.0, size=exposed.shape) * sigma[..., None]
        observed = np.clip(shot + read, 0.0, 1.0)
        params = {"alpha_weak": alpha0, "sigma_r_weak": sigma0,
                  "alpha_strong": alpha0 * 4.0, "sigma_r_strong": sigma0 * 2.5,
                  "strong_mask_fraction": float(strong.mean())}
        masks = {"strong": strong}
        primary = (alpha0, sigma0)
    else:
        lam = np.maximum(exposed / alpha0, 0.0)
        observed = np.clip(alpha0 * rng.poisson(lam)
                           + rng.normal(0.0, sigma0, size=exposed.shape), 0.0, 1.0)
        params = {"alpha": alpha0, "sigma_r": sigma0}
        masks = {}
        primary = (alpha0, sigma0)

    return {"kind": kind, "size": int(size), "seed": int(seed),
            "clean_linear": clean, "exposed_linear": exposed,
            "observed_linear": observed,
            "observed_srgb": np.clip(linear_to_srgb(observed), 0.0, 1.0),
            "alpha": primary[0], "sigma_r": primary[1],
            "params": params, "masks": masks}


def make_gui_dark_photo(size: int = 720, seed: int = 3) -> dict:
    """A GUI-friendly dark photo: textured midtones, deep shadow, a window.

    Unlike the benchmark workloads this is meant to be opened in the恢复窗口:
    a dark interior with fine detail plus a bright window that must not blow
    out when the shadow lift is applied.
    """
    rng = np.random.default_rng(seed + 77)
    h = w = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    # Dark interior base with gentle vignetting.
    r = (xx - w / 2) / (w / 2)
    c = (yy - h / 2) / (h / 2)
    vign = 1.0 - 0.35 * np.clip(r * r + c * c, 0.0, 1.0)
    base = (0.035 + 0.02 * (xx / w)) * vign
    texture = (0.010 * np.sin(2.0 * np.pi * xx / 5.0)
               + 0.008 * np.sin(2.0 * np.pi * yy / 7.0)
               + 0.006 * np.tanh(rng.standard_normal((h, w)) * 1.5))
    luma = np.clip(base + texture, 0.004, 1.0)
    scene = np.stack([luma * 0.98, luma, luma * 1.06], axis=-1)
    # Bright window (clipped highlight) in the upper right.
    wy0, wy1 = size // 10, size // 10 + size // 4
    wx0, wx1 = 3 * size // 5, 3 * size // 5 + size // 4
    scene[wy0:wy1, wx0:wx1] = [0.95, 0.92, 0.80]
    # A few fine structural lines in the shadow for detail inspection.
    for k in range(0, size // 2, 24):
        scene[size // 2 + k, size // 6: 2 * size // 3] = [0.16, 0.15, 0.13]
    exposed = np.clip(scene * 0.5, 0.0, 1.0)
    observed = np.clip(0.008 * rng.poisson(exposed / 0.008)
                       + rng.normal(0.0, 0.005, size=exposed.shape), 0.0, 1.0)
    return {"clean_linear": clean_clip(scene), "exposed_linear": exposed,
            "observed_linear": observed,
            "observed_srgb": np.clip(linear_to_srgb(observed), 0.0, 1.0),
            "alpha": 0.008, "sigma_r": 0.005}


def clean_clip(a: np.ndarray) -> np.ndarray:
    return np.clip(a, 0.0, 1.0)
