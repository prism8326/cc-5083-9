"""Constrained lifting of dark regions.

The lift is *gain applied to the denoised signal* (never a brightening step
followed by aggressive smoothing).  Two constraints keep the operation safe:

1. the gain is capped (``max_gain``) so shadow noise is never amplified without
   bound and the output cannot clip silently;
2. the gain rolls off smoothly toward 1.0 above the shadow range, so mid-tones
   and highlights - and therefore colour balance - are left untouched.

Gain is computed from luma and applied per channel proportionally, preserving
the inter-channel ratios that carry subtle dark-tone colour differences.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .color import LUMA_W


@dataclass(frozen=True)
class LiftParams:
    shadow_level: float = 0.12   # target linear level for the shadow reference
    shadow_quantile: float = 0.10  # which input quantile defines "shadow"
    max_gain: float = 8.0
    rolloff_quantile: float = 0.60  # gain is back to 1 above this quantile
    black_fraction: float = 0.001  # fraction treated as the optical black floor

    def as_dict(self) -> dict:
        return {"shadow_level": self.shadow_level,
                "shadow_quantile": self.shadow_quantile,
                "max_gain": self.max_gain,
                "rolloff_quantile": self.rolloff_quantile,
                "black_fraction": self.black_fraction}


def _smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / max(e1 - e0, 1.0e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def lift_shadows(rgb_linear: np.ndarray, params: LiftParams | None = None
                 ) -> tuple[np.ndarray, dict]:
    """Apply the constrained shadow lift to a denoised linear-light image."""
    p = params or LiftParams()
    img = np.asarray(rgb_linear, dtype=np.float64)
    luma = np.einsum("...c,c->...", img, LUMA_W)

    valid = luma[np.isfinite(luma)]
    if valid.size == 0:
        raise ValueError("lift_shadows received an empty image")
    q_shadow = float(np.quantile(valid, np.clip(p.shadow_quantile, 0.0, 1.0)))
    q_rolloff = float(np.quantile(valid, np.clip(p.rolloff_quantile, 0.0, 1.0)))
    black = float(np.quantile(valid, p.black_fraction))

    # Gain needed to bring the shadow reference up to the requested level.
    base_gain = p.shadow_level / max(q_shadow - black, 1.0e-6)
    shadow_gain = float(np.clip(base_gain, 1.0, p.max_gain))

    # Smooth gain mask: shadow_gain in the dark, 1.0 past the roll-off point.
    weight = 1.0 - _smoothstep(q_shadow, max(q_rolloff, q_shadow + 1.0e-6), luma)
    gain = 1.0 + (shadow_gain - 1.0) * weight

    # Optical-black subtraction before scaling, only above the black floor.
    lifted = (np.maximum(img - black, 0.0)) * gain[..., None]

    before_clip = lifted
    clipped_hi = np.mean(lifted > 1.0)
    lifted = np.clip(lifted, 0.0, 1.0)

    stats = {"params": p.as_dict(),
             "luma_quantiles": {"shadow_q": q_shadow, "rolloff_q": q_rolloff,
                                "black_floor": black},
             "applied_shadow_gain": shadow_gain,
             "requested_gain_uncapped": float(base_gain),
             "gain_capped": bool(base_gain > p.max_gain),
             "clip_fraction_after_lift": float(clipped_hi),
             "luma_mean_before": float(luma.mean()),
             "luma_mean_after": float(np.einsum("...c,c->...", lifted, LUMA_W).mean()),
             "max_output_before_clip": float(np.nanmax(before_clip))}
    return lifted, stats
