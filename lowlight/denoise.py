"""Edge-preserving denoising in the variance-stabilized GAT domain.

After the GAT every pixel has unit noise variance regardless of brightness, so
a single strength value has a fixed physical meaning (denoising in units of
noise standard deviations) and does not change when the image is resized.

Two filters are provided:

* ``nlm``  : OpenCV fast non-local means (default), good thin-line retention;
* ``wavelet``: dependency-free multi-scale Haar shrinkage fallback.

Channels are filtered independently: no cross-channel mixing means subtle
colour differences in dark tones are preserved rather than averaged grey.

Large images / tiling
---------------------
``fastNlMeansDenoising`` needs the whole image at once.  For frames bigger
than ``tile_pixels`` the channel is processed in overlapping tiles.  A tile's
*padding ring* (halo) is computed from both window sizes:

    halo = search_size // 2 + template_size // 2

so every output pixel in the tile's core sees the same candidate patches it
would see in the full frame.  Cores are the non-overlapping tile rectangles
themselves, so each output pixel is written exactly once; halos only provide
input and are never written, hence no seam and no double-processed border.
Tiling is only applied when the tile rectangle is smaller than the frame,
which keeps small-image results bit-identical to the untiled call.
"""
from __future__ import annotations

import time

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

# Only tile when the channel exceeds this many pixels; below it the untiled
# call is already fast and is bit-identical to the reference implementation.
DEFAULT_TILE_PIXELS = 1024 * 1024
# Target core side length; actual grid is derived from the image shape.
DEFAULT_TILE_SIDE = 512


def nlm_halo(template_size: int, search_size: int) -> int:
    """Padding ring that makes a tiled NLM call equal to the untiled call."""
    return max(1, int(search_size) // 2 + int(template_size) // 2)


def _nlm_channel(z: np.ndarray, strength: float, template: int, search: int,
                 out: np.ndarray | None = None) -> np.ndarray:
    h = float(np.clip(strength, 0.0, 5.0))
    scaled = np.clip(np.clip(z, 0.0, None) * _NLM_SCALE, 0, _NLM_MAX).astype(np.uint8)
    # h is expressed in GAT-sigma units and converted through the same fixed
    # quantization factor used for the data.
    den = cv2.fastNlMeansDenoising(scaled, None, h=h * _NLM_SCALE,
                                   templateWindowSize=int(template),
                                   searchWindowSize=int(search))
    result = den.astype(np.float64) / _NLM_SCALE
    if out is not None:
        out[...] = result
        return out
    return result


def _tile_spans(length: int, side: int) -> list[tuple[int, int]]:
    if length <= side:
        return [(0, length)]
    n = (length + side - 1) // side
    edges = np.linspace(0, length, n + 1).round().astype(int)
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def _tile_rects(h: int, w: int, tile_side: int, halo: int):
    """Yield ``(core, padded)`` rectangles; cores are disjoint, padding is halo."""
    ys = _tile_spans(h, tile_side)
    xs = _tile_spans(w, tile_side)
    for y0, y1 in ys:
        for x0, x1 in xs:
            ey0, ey1 = max(0, y0 - halo), min(h, y1 + halo)
            ex0, ex1 = max(0, x0 - halo), min(w, x1 + halo)
            yield (y0, y1, x0, x1), (ey0, ey1, ex0, ex1)


def _nlm_channel_tiled(z: np.ndarray, strength: float, template: int,
                       search: int, tile_side: int,
                       halo: int | None = None) -> tuple[np.ndarray, dict]:
    """Halo-padded tiled NLM; cores are disjoint so every pixel is written once."""
    h, w = z.shape
    if halo is None:
        halo = nlm_halo(template, search)
    out = np.empty_like(z)
    n_tiles = 0
    for (y0, y1, x0, x1), (ey0, ey1, ex0, ex1) in _tile_rects(h, w, tile_side,
                                                              halo):
        den = _nlm_channel(z[ey0:ey1, ex0:ex1], strength, template, search)
        out[y0:y1, x0:x1] = den[y0 - ey0:y1 - ey0, x0 - ex0:x1 - ex0]
        n_tiles += 1
    stats = {"tiles": n_tiles, "tile_side": int(tile_side), "halo": int(halo),
             "grid": [len(_tile_spans(h, tile_side)),
                      len(_tile_spans(w, tile_side))]}
    return out, stats


def _haar_forward(a: np.ndarray):
    """One level of 2-D Haar DWT; returns LL and the three detail bands."""
    orig_shape = a.shape
    pad_r, pad_c = a.shape[0] % 2, a.shape[1] % 2
    if pad_r or pad_c:
        a = np.pad(a, ((0, pad_r), (0, pad_c)), mode="edge")
    s00, s01 = a[0::2, 0::2], a[0::2, 1::2]
    s10, s11 = a[1::2, 0::2], a[1::2, 1::2]
    ll = (s00 + s01 + s10 + s11) / 2.0
    hl = ((s00 + s10) - (s01 + s11)) / 2.0   # horizontal detail
    lh = ((s00 + s01) - (s10 + s11)) / 2.0   # vertical detail
    hh = (s00 - s01 - s10 + s11) / 2.0       # diagonal detail
    return ll, (hl, lh, hh), orig_shape


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
            search_size: int = 21,
            tile_pixels: int = DEFAULT_TILE_PIXELS,
            tile_side: int = DEFAULT_TILE_SIDE,
            z_gat: np.ndarray | None = None,
            ) -> tuple[np.ndarray, dict]:
    """Denoise a linear-light image.

    strength : multiplier on the unit-noise denoising threshold; 1.0 matches
               the measured noise, <1 preserves more texture, >2 smooths hard.
    z_gat    : optional precomputed GAT image (reused across strength changes).
    Returns the denoised linear-light image and run statistics.
    """
    if strength < 0.0:
        raise ValueError("strength must be non-negative")
    img = np.asarray(rgb_linear, dtype=np.float64)
    t_gat = time.perf_counter()
    z = z_gat if z_gat is not None else gat(img, params)
    gat_s = time.perf_counter() - t_gat

    if method == "nlm" and not _HAVE_CV2:
        method = "wavelet"
    if method not in ("nlm", "wavelet"):
        raise ValueError(f"unknown denoise method {method!r}")

    t_filter = time.perf_counter()
    z_den = np.empty_like(z)
    per_channel = []
    tile_stats = None
    h, w = z.shape[:2]
    use_tiles = method == "nlm" and h * w > int(tile_pixels)
    for c in range(3):
        before = float(z[..., c].var())
        if method == "nlm":
            if use_tiles:
                z_den[..., c], tile_stats = _nlm_channel_tiled(
                    z[..., c], strength, int(template_size), int(search_size),
                    int(tile_side))
            else:
                z_den[..., c] = _nlm_channel(z[..., c], strength,
                                             int(template_size), int(search_size))
        else:
            z_den[..., c] = _wavelet_channel(z[..., c], strength)
        per_channel.append({"channel": "RGB"[c],
                            "gat_var_before": before,
                            "gat_var_after": float(z_den[..., c].var())})
    filter_s = time.perf_counter() - t_filter

    t_inv = time.perf_counter()
    out = inverse_gat(z_den, params)
    inv_s = time.perf_counter() - t_inv
    # The unbiased inverse can produce tiny negatives in deep black; the
    # pipeline's reliability map flags those pixels separately.
    stats = {"method": method, "strength": float(strength),
             "template_size": int(template_size) if method == "nlm" else None,
             "search_size": int(search_size) if method == "nlm" else None,
             "per_channel": per_channel,
             "gat_clip_fraction": float(np.mean(z <= 0.0)),
             "timing_s": {"gat": float(gat_s), "filter": float(filter_s),
                          "inverse_gat": float(inv_s)},
             "tiling": tile_stats,
             "z_den": z_den}
    return out, stats


def denoise_channels_only(rgb_linear: np.ndarray, params: NoiseParams,
                          z_gat: np.ndarray, strength: float = 1.0,
                          method: str = "nlm", template_size: int = 7,
                          search_size: int = 21,
                          tile_pixels: int = DEFAULT_TILE_PIXELS,
                          tile_side: int = DEFAULT_TILE_SIDE) -> np.ndarray:
    """Return the denoised *GAT-domain* image (cacheable before inverse GAT)."""
    return denoise(rgb_linear, params, strength=strength, method=method,
                   template_size=template_size, search_size=search_size,
                   tile_pixels=tile_pixels, tile_side=tile_side,
                   z_gat=z_gat)[0]
