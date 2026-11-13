"""Color space helpers.

All noise modelling and lifting happens in *linear-light* space so that the
signal-dependent (Poisson-like) noise term has a physical meaning.  The common
case is an sRGB-encoded input image; an already-linear input can be declared
explicitly.
"""
from __future__ import annotations

import numpy as np

# sRGB EOTF breakpoint (linear segment).
_SRGB_K0 = 0.04045
_SRGB_PHI = 12.92
_SRGB_GAMMA = 2.4
_SRGB_A = 1.055
_SRGB_B = 0.055


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """Decode sRGB-encoded values (float in [0, 1]) to linear light."""
    x = np.asarray(x, dtype=np.float64)
    return np.where(x <= _SRGB_K0, x / _SRGB_PHI,
                    np.power(np.maximum(x + _SRGB_B, 0.0) / _SRGB_A, _SRGB_GAMMA))


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    """Encode linear-light values (float) back to sRGB in [0, 1]."""
    x = np.asarray(x, dtype=np.float64)
    out = np.where(x <= _SRGB_K0 / _SRGB_PHI,
                   _SRGB_PHI * x,
                   _SRGB_A * np.power(np.maximum(x, 0.0), 1.0 / _SRGB_GAMMA) - _SRGB_B)
    return np.clip(out, 0.0, 1.0)


# Rec.709 luma coefficients applied to *linear* channels.
LUMA_W = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)


def linear_luma(rgb_linear: np.ndarray) -> np.ndarray:
    """Luminance of an HxWx3 linear-light image."""
    return np.einsum("...c,c->...", rgb_linear, LUMA_W)


def encode_display(rgb_linear: np.ndarray) -> np.ndarray:
    """Convert a linear-light RGB image to an 8-bit sRGB display array."""
    return (np.clip(linear_to_srgb(rgb_linear), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
