"""Quantitative metrics for the offline evaluation harness.

Metrics (all computed in linear-light space unless stated):

* thin-line contrast retention on the synthetic line panel;
* shadow-region mean error (dark mean bias);
* per-channel colour bias on flat coloured patches;
* black-patch behaviour (mean must stay at ~0, no invented signal);
* PSNR / RMSE against the clean reference.
"""
from __future__ import annotations

import numpy as np

from .color import LUMA_W


def rmse(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    if mask is not None:
        a, b = a[mask], b[mask]
    return float(np.sqrt(np.mean((a - b) ** 2)))


def psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    e = rmse(a, b, mask)
    if e <= 1.0e-12:
        return float("inf")
    return float(20.0 * np.log10(1.0 / e))


def line_contrast(reference_linear: np.ndarray, restored_linear: np.ndarray,
                  size: int) -> dict:
    """Contrast (bright line minus dark neighbourhood) on the top horizontal
    1-px line set inside the dark panel.

    The reference is normally the *exposed* clean scene: restoration cannot
    recreate photons that were never captured, so the honest retention measure
    is how much of the contrast actually present in the observation survives.
    """
    h = w = size
    y = h // 4
    xs = np.arange(w // 4, 3 * w // 4)
    line_x = xs[(xs - w // 4) % 14 == 0]
    bg_x = xs[(xs - w // 4) % 14 != 0]
    luma_ref = np.einsum("...c,c->...", reference_linear, LUMA_W)
    luma_rest = np.einsum("...c,c->...", restored_linear, LUMA_W)
    c_ref = float(luma_ref[y, line_x].mean() - luma_ref[y + 2:y + 6, :][:, bg_x].mean())
    c_rest = float(luma_rest[y, line_x].mean() - luma_rest[y + 2:y + 6, :][:, bg_x].mean())
    return {"contrast_reference": c_ref, "contrast_restored": c_rest,
            "retention": float(c_rest / c_ref) if c_ref > 0 else float("nan")}


def shadow_mean_error(reference_linear: np.ndarray, restored_linear: np.ndarray,
                      threshold: float = 0.05) -> dict:
    """Bias of the mean in genuinely dark reference regions.

    The reference is the exposed clean scene (the recoverable signal), so the
    bias measures restoration error rather than the un-recoverable photon
    shortfall of the low exposure itself.
    """
    luma = np.einsum("...c,c->...", reference_linear, LUMA_W)
    mask = (luma > 0.0) & (luma < threshold)
    if mask.sum() < 8:
        return {"n_pixels": int(mask.sum()), "mean_reference": None,
                "mean_restored": None, "bias": None}
    mc = float(reference_linear[mask].mean())
    mr = float(restored_linear[mask].mean())
    return {"n_pixels": int(mask.sum()), "mean_reference": mc,
            "mean_restored": mr, "bias": mr - mc,
            "relative_bias": (mr - mc) / max(mc, 1.0e-9)}


def colour_bias(clean_linear: np.ndarray, restored_linear: np.ndarray,
                mask: np.ndarray) -> dict:
    """Per-channel mean bias inside a masked flat-colour region."""
    diff = restored_linear[mask].mean(axis=0) - clean_linear[mask].mean(axis=0)
    return {"per_channel_bias": [float(x) for x in diff],
            "max_abs_bias": float(np.max(np.abs(diff))),
            "chroma_angle_change_rad": _chroma_angle(clean_linear, restored_linear, mask)}


def _chroma_angle(clean, rest, mask) -> float:
    mc = clean[mask].mean(axis=0)
    mr = rest[mask].mean(axis=0)
    ac = np.arctan2(mc[1] - mc[2], mc[0] - mc[2])
    ar = np.arctan2(mr[1] - mr[2], mr[0] - mr[2])
    return float(ar - ac)


def black_patch_stats(restored_linear: np.ndarray, size: int) -> dict:
    patch = restored_linear[0:size // 10, 0:size // 10]
    return {"mean": float(patch.mean()), "max": float(patch.max()),
            "std": float(patch.std())}


def all_metrics(fixture: dict, result,
                size: int) -> dict:
    """Two-level evaluation.

    Denoise level compares the GAT-denoised image against the exposed clean
    scene at the *same* brightness scale, isolating noise/texture performance.
    Lift level checks the constrained gain behaviour of the final image.
    The fully-exposed clean reference never enters the runtime pipeline.
    """
    reference = fixture["exposed_linear"]
    denoised = np.maximum(result.rgb_linear_denoised, 0.0)
    restored = result.rgb_linear_restored
    h, w = size, size
    red_mask = np.zeros((h, w), dtype=bool)
    red_mask[0:size // 10, size // 10:size // 5] = True
    lift = result.stats["lift"]
    # Invert the applied gain on the restored image to bring it back to the
    # observation scale, so the lift-level error excludes the intended boost.
    gain = lift["applied_shadow_gain"]
    restored_unlifted = restored / gain
    return {
        "denoise": {
            "rmse_linear": rmse(denoised, reference),
            "psnr_linear": psnr(denoised, reference),
            "line_contrast": line_contrast(reference, denoised, size),
            "shadow_mean": shadow_mean_error(reference, denoised),
            "colour_bias_red_patch": colour_bias(reference, denoised, red_mask),
        },
        "lift": {
            "applied_gain": gain,
            "gain_capped": lift["gain_capped"],
            "clip_fraction": lift["clip_fraction_after_lift"],
            "rmse_after_gain_normalization": rmse(restored_unlifted, reference),
            "luma_boost": (result.stats["luma_mean_restored"]
                           / max(result.stats["luma_mean_observed"], 1.0e-12)),
        },
        "black_patch": black_patch_stats(restored, size),
        "rmse_vs_observation": rmse(restored, fixture["observed_linear"]),
    }
