"""Edge-preserving denoising in the variance-stabilized GAT domain.

After the GAT every pixel has unit noise variance regardless of brightness, so
a single strength value has a fixed physical meaning (denoising in units of
noise standard deviations) and does not change when the image is resized.

Two filters are provided:

* ``nlm``  : OpenCV fast non-local means (default), good thin-line retention;
* ``wavelet``: dependency-free multi-scale Haar shrinkage fallback.

Channels are filtered independently: no cross-channel mixing means subtle
colour differences in dark tones are preserved rather than averaged grey.
"""
from __future__ import annotations

import numpy as np

from .noise import NoiseParams, gat, inverse_gat

try:
    import cv2
    _HAVE_CV2 = True
except Exception:  # pragma: no cover - exercised only without OpenCV
    _HAVE_CV2 = False


# OpenCV's NLM only accepts 8-bit images.  The GAT domain has unit variance
# with no image-dependent scaling, so one fixed quantization factor preserves
# the size-invariant meaning of ``strength`` across all inputs.  The factor
# maps the useful GAT range (0..~14 on real dark frames) across the byte range.
_NLM_SCALE = 16.0
_NLM_MAX = 255


def _nlm_channel(z: np.ndarray, strength: float, template: int, search: int) -> np.ndarray:
    h = float(np.clip(strength, 0.0, 5.0))
    scaled = np.clip(np.clip(z, 0.0, None) * _NLM_SCALE, 0, _NLM_MAX).astype(np.uint8)
    # h is expressed in GAT-sigma units and converted through the same fixed
    # quantization factor used for the data.
    den = cv2.fastNlMeansDenoising(scaled, None, h=h * _NLM_SCALE,
                                   templateWindowSize=int(template),
                                   searchWindowSize=int(search))
    return den.astype(np.float64) / _NLM_SCALE


def _haar_forward(a: np.ndarray):
    """One level of 2-D Haar DWT; returns LL and the three detail bands."""
    pad_r, pad_c = a.shape[0] % 2, a.shape[1] % 2
    if pad_r or pad_c:
        a = np.pad(a, ((0, pad_r), (0, pad_c)), mode="edge")
    s00, s01 = a[0::2, 0::2], a[0::2, 1::2]
    s10, s11 = a[1::2, 0::2], a[1::2, 1::2]
    ll = (s00 + s01 + s10 + s11) / 2.0
    hl = ((s00 + s10) - (s01 + s11)) / 2.0   # horizontal detail
    lh = ((s00 + s01) - (s10 + s11)) / 2.0   # vertical detail
    hh = (s00 - s01 - s10 + s11) / 2.0       # diagonal detail
    return ll, (hl, lh, hh), a.shape


def _haar_inverse(ll: np.ndarray, details, shape) -> np.ndarray:
    hl, lh, hh = details
    s00 = (ll + hl + lh + hh) / 2.0
    s01 = (ll - hl + lh - hh) / 2.0
    s10 = (ll + hl - lh - hh) / 2.0
    s11 = (ll - hl - lh + hh) / 2.0
    out = np.empty((2 * ll.shape[0], 2 * ll.shape[1]), dtype=np.float64)
    out[0::2, 0::2], out[0::2, 1::2] = s00, s01
    out[1::2, 0::2], out[1::2, 1::2] = s10, s11
    return out[:shape[0], :shape[1]]


def _wavelet_channel(z: np.ndarray, strength: float, levels: int = 4) -> np.ndarray:
    """Multi-scale Haar shrinkage; soft thresholds grow with the level."""
    threshold = 0.5 * float(np.clip(strength, 0.0, 5.0))
    bands: list[tuple] = []
    ll = z.astype(np.float64)
    for level in range(levels):
        if min(ll.shape) < 2:
            break
        ll, details, shape = _haar_forward(ll)
        t = threshold * (1.25 ** level)
        details = tuple(np.sign(d) * np.maximum(np.abs(d) - t, 0.0)
                        for d in details)
        bands.append((details, shape))
    for details, shape in reversed(bands):
        ll = _haar_inverse(ll, details, shape)
    return ll


def denoise(rgb_linear: np.ndarray, params: NoiseParams, strength: float = 1.0,
            method: str = "nlm", template_size: int = 7,
            search_size: int = 21) -> tuple[np.ndarray, dict]:
    """Denoise a linear-light image.

    strength : multiplier on the unit-noise denoising threshold; 1.0 matches
               the measured noise, <1 preserves more texture, >2 smooths hard.
    Returns the denoised linear-light image and run statistics.
    """
    if strength < 0.0:
        raise ValueError("strength must be non-negative")
    img = np.asarray(rgb_linear, dtype=np.float64)
    z = gat(img, params)

    if method == "nlm" and not _HAVE_CV2:
        method = "wavelet"
    if method not in ("nlm", "wavelet"):
        raise ValueError(f"unknown denoise method {method!r}")

    z_den = np.empty_like(z)
    per_channel = []
    for c in range(3):
        before = float(z[..., c].var())
        if method == "nlm":
            z_den[..., c] = _nlm_channel(z[..., c], strength,
                                         int(template_size), int(search_size))
        else:
            z_den[..., c] = _wavelet_channel(z[..., c], strength)
        per_channel.append({"channel": "RGB"[c],
                            "gat_var_before": before,
                            "gat_var_after": float(z_den[..., c].var())})

    out = inverse_gat(z_den, params)
    # The unbiased inverse can produce tiny negatives in deep black; the
    # pipeline's reliability map flags those pixels separately.
    stats = {"method": method, "strength": float(strength),
             "template_size": int(template_size) if method == "nlm" else None,
             "search_size": int(search_size) if method == "nlm" else None,
             "per_channel": per_channel,
             "gat_clip_fraction": float(np.mean(z <= 0.0))}
    return out, stats
