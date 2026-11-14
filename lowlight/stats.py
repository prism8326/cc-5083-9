"""Reusable block-statistics representation and a fast exact Theil-Sen fit.

Why this module exists
----------------------
The original calibration recomputed ``(mean, variance, gradient, saturation)``
for every fixed-pixel patch on every run and then handed *all* flat patches to
``scipy.stats.theilslopes``.  Theil-Sen forms every pairwise slope, i.e. an
``O(n^2)`` slope array; at 512 px that is already ~4.2e9 float64 values
(tens of GB) and the process is killed.  This module keeps the *same* estimator
semantics but:

* computes patch statistics once into a compact, reusable
  :class:`PatchStatistics` object (chunked, deterministic reduction order),
* replaces the pairwise-slope array with a count-based selection of the
  median of all pairwise slopes (``O(n log n)``, no ``n^2`` allocation),
* bounds the number of blocks fed to the regression with an explicitly
  recorded, deterministic uniform subsample (an approximation, never passed
  off as an equivalent rewrite).

The statistics object is content-addressed so parameter changes invalidate
only the caches that actually depend on them (see :mod:`lowlight.cache`).
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

import numpy as np

BLOCK_SIZE = 8
BLOCK_STRIDE = 4
MIN_SAMPLES = 24

# Block count above which the stride is widened.  Patch *size* never changes
# (physical pixels), only how densely the image is sampled, so each block keeps
# the same noise statistics; this is a sampling approximation, reported as
# ``approximate=True`` in the diagnostics.
MAX_BLOCKS = 262_144
# Number of flat patches handed to the regression.  Theil-Sen is exact for
# that sample set; the subsample itself is a deterministic, reported policy.
THEIL_BUDGET = 4096
# Leaf size for the merge-sort inversion count used by the exact median
# pairwise slope.  Tuned for numpy's sort/searchsorted cost curve.
_INVERSION_BASE = 128
# Rows of blocks processed per chunk while accumulating statistics.  Chunks
# are concatenated in a fixed order, so the result does not depend on worker
# count (there is no floating-point reduction across chunks beyond a final
# fixed-order concatenation of independent per-block values).
BLOCK_CHUNK_ROWS = 96


def content_digest(rgb_linear: np.ndarray) -> str:
    """Content digest of the (linear) input array.

    Small arrays are hashed byte-for-byte.  For very large frames (where
    hashing the whole buffer would itself become a measured preprocessing
    cost) a strided sample is hashed together with shape and dtype; this is
    recorded as ``digest_mode`` and is only used as a cache key, never as an
    algorithmic input.
    """
    arr = np.ascontiguousarray(rgb_linear)
    h = hashlib.blake2b(digest_size=20)
    h.update(str(arr.shape).encode())
    h.update(str(arr.dtype).encode())
    if arr.nbytes <= 64 * 1024 * 1024:
        h.update(memoryview(arr).cast("B"))
        mode = "full"
    else:
        step_y = max(1, arr.shape[0] // 512)
        step_x = max(1, arr.shape[1] // 512)
        sample = np.ascontiguousarray(arr[::step_y, ::step_x])
        h.update(memoryview(sample).cast("B"))
        mode = f"sampled(1/{step_y},{1}/{step_x})"
    return f"{mode}:{h.hexdigest()}"


@dataclass
class PatchStatistics:
    """Reusable per-patch statistics for all three colour channels.

    Arrays are kept at their block-grid shape ``(ny, nx)`` per channel.  They
    depend only on image content, block geometry and saturation thresholds -
    never on denoise strength, filter choice or lift parameters - so they can
    be reused across those changes.
    """

    means: np.ndarray          # (3, ny, nx)
    variances: np.ndarray      # (3, ny, nx), unbiased (ddof=1)
    gradients: np.ndarray      # (3, ny, nx)
    saturated: np.ndarray      # (3, ny, nx) bool
    block_size: int
    stride: int
    requested_stride: int
    sat_hi: float
    sat_lo: float
    digest: str
    digest_mode: str
    approximate: bool = False
    elapsed_s: float = 0.0
    n_blocks: int = 0

    def as_dict(self) -> dict:
        return {
            "block_size": self.block_size,
            "stride": self.stride,
            "requested_stride": self.requested_stride,
            "grid": [int(self.means.shape[1]), int(self.means.shape[2])],
            "n_blocks": int(self.n_blocks),
            "approximate_stride": bool(self.approximate),
            "digest": self.digest,
            "digest_mode": self.digest_mode,
            "sat_hi": self.sat_hi,
            "sat_lo": self.sat_lo,
            "elapsed_s": float(self.elapsed_s),
            "bytes": int(self.means.nbytes + self.variances.nbytes
                         + self.gradients.nbytes + self.saturated.nbytes),
        }


def _block_grid(height: int, width: int, block: int, stride: int) -> tuple[int, int]:
    ny = max(0, (height - block) // stride + 1)
    nx = max(0, (width - block) // stride + 1)
    return ny, nx


def choose_stride(height: int, width: int, block: int, requested: int,
                  max_blocks: int = MAX_BLOCKS) -> tuple[int, bool]:
    """Widen the block stride until the block grid fits ``max_blocks``."""
    ny, nx = _block_grid(height, width, block, requested)
    if ny * nx <= max_blocks:
        return requested, False
    stride = requested
    while True:
        stride += 1
        ny, nx = _block_grid(height, width, block, stride)
        if ny == 0 or nx == 0:
            return max(requested, min(height, width) // 2), True
        if ny * nx <= max_blocks:
            return stride, True


def compute_patch_statistics(rgb_linear: np.ndarray, block: int = BLOCK_SIZE,
                             stride: int = BLOCK_STRIDE,
                             sat_hi: float | None = None,
                             sat_lo: float = 0.0,
                             max_blocks: int = MAX_BLOCKS,
                             digest: str | None = None,
                             digest_mode: str = "provided") -> PatchStatistics:
    """Compute (and pack) patch statistics once per image content.

    The reduction is chunked over block rows; each block's mean/variance/
    gradient is reduced independently with numpy's fixed pairwise summation,
    and chunks are concatenated in index order, so the arrays are bit-identical
    regardless of how many worker threads produced the chunks.
    """
    t0 = time.perf_counter()
    img = np.asarray(rgb_linear, dtype=np.float64)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("compute_patch_statistics expects HxWx3 linear RGB")
    h, w = img.shape[:2]
    if sat_hi is None:
        sat_hi = float(np.nextafter(1.0, 0.0))
    eff_stride, approximate = choose_stride(h, w, block, stride, max_blocks)
    ny, nx = _block_grid(h, w, block, eff_stride)
    if ny == 0 or nx == 0:
        raise ValueError(f"image {h}x{w} too small for {block}px blocks")

    means = np.empty((3, ny, nx), dtype=np.float64)
    variances = np.empty((3, ny, nx), dtype=np.float64)
    gradients = np.empty((3, ny, nx), dtype=np.float64)
    saturated = np.empty((3, ny, nx), dtype=bool)

    # Number of pixel rows spanned by chunk_rows block rows.
    chunk_rows = max(1, min(BLOCK_CHUNK_ROWS, ny))
    y0_px, y1_px = 0, (chunk_rows - 1) * eff_stride + block
    for c in range(3):
        plane = img[..., c]
        row = 0
        while row < ny:
            rows = min(chunk_rows, ny - row)
            span_y = (rows - 1) * eff_stride + block
            sub = plane[row * eff_stride: row * eff_stride + span_y, :]
            # windows: (rows, nx, block, block)
            windows = np.lib.stride_tricks.sliding_window_view(
                sub, (block, block))[::eff_stride, ::eff_stride][:rows, :nx]
            n = block * block
            means[c, row:row + rows] = windows.mean(axis=(-2, -1))
            variances[c, row:row + rows] = windows.var(axis=(-2, -1), ddof=1)
            gy = np.abs(windows[..., 1:, :] - windows[..., :-1, :]).mean(axis=(-2, -1))
            gx = np.abs(windows[..., :, 1:] - windows[..., :, :-1]).mean(axis=(-2, -1))
            gradients[c, row:row + rows] = gy + gx
            saturated[c, row:row + rows] = (
                (windows >= sat_hi).any(axis=(-2, -1))
                | (windows <= sat_lo).all(axis=(-2, -1)))
            row += rows

    elapsed = time.perf_counter() - t0
    if digest is None:
        digest = content_digest(img)
        digest_mode = digest.split(":", 1)[0]
    return PatchStatistics(
        means=means, variances=variances, gradients=gradients,
        saturated=saturated, block_size=block, stride=eff_stride,
        requested_stride=stride, sat_hi=float(sat_hi), sat_lo=float(sat_lo),
        digest=digest, digest_mode=digest_mode, approximate=approximate,
        elapsed_s=elapsed, n_blocks=int(ny * nx))


# --------------------------------------------------------------------------
# Exact median of all pairwise slopes, without materializing the n^2 array.
# --------------------------------------------------------------------------
_UPPER_IDX: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _upper_indices(n: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        return _UPPER_IDX[n]
    except KeyError:
        iu = np.triu_indices(n, 1)
        _UPPER_IDX[n] = iu
        return iu


def _base_inversions(a: np.ndarray, base: int = _INVERSION_BASE) -> tuple[np.ndarray, int]:
    """Sorted values plus strict-inversion count for a small block.

    Only the upper triangle of the comparison matrix is inspected, using
    index pairs cached per block size, so no ``n^2`` matrix is materialized
    beyond the cached index pair list.
    """
    n = a.size
    if n < 2:
        return np.sort(a), 0
    iu, ju = _upper_indices(n)
    inv = int(np.count_nonzero(a[iu] > a[ju]))
    return np.sort(a), inv


def _merge_sorted_inversions(left: np.ndarray, right: np.ndarray,
                             base: int = _INVERSION_BASE) -> tuple[np.ndarray, int]:
    """Count pairs ``l in left, r in right`` with ``l > r`` and merge them."""
    # #left > r  ==  len(left) - #left <= r
    le = np.searchsorted(left, right, side="right")
    inv = int(np.sum(left.size - le))
    merged = np.sort(np.concatenate([left, right]))
    return merged, inv


def _sorted_runs_inversions(a: np.ndarray, base: int = _INVERSION_BASE) -> tuple[np.ndarray, int]:
    """Bottom-up merge sort returning sorted array and strict inversion count.

    The array is split into ``base``-sized leaves (counted with an exact
    broadcast comparison), then merged level by level.  Every level uses
    vectorized ``searchsorted`` so total work is O(n log n) with no Python
    per-element loop.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    n = a.size
    if n == 0:
        return a.copy(), 0
    runs: list[np.ndarray] = []
    count = 0
    for start in range(0, n, base):
        s, inv = _base_inversions(a[start:start + base])
        runs.append(s)
        count += inv
    while len(runs) > 1:
        nxt: list[np.ndarray] = []
        it = iter(runs)
        for left in it:
            right = next(it, None)
            if right is None:
                nxt.append(left)
                continue
            merged, inv = _merge_sorted_inversions(left, right)
            nxt.append(merged)
            count += inv
        runs = nxt
    return runs[0], count


def _equal_pairs_sorted(s: np.ndarray) -> int:
    """Number of unordered pairs with equal values, from an already-sorted array."""
    if s.size < 2:
        return 0
    d = np.r_[True, s[1:] != s[:-1]]
    starts = np.flatnonzero(d)
    ends = np.r_[starts[1:], s.size]
    sizes = ends - starts
    return int(np.sum(sizes * (sizes - 1) // 2))


def _x_equal_pairs_count(x_sorted: np.ndarray) -> int:
    """Number of unordered pairs sharing an x value (adjacent runs)."""
    return _equal_pairs_sorted(x_sorted)


def _count_pairs_le_slope(x_sorted: np.ndarray, cost: np.ndarray,
                          x_tie_le: int) -> int:
    """#ordered pairs (i < j, x_i != x_j) with (y_j-y_i)/(x_j-x_i) <= s.

    ``cost[k] = y[k] - s*x[k]``.  For x-sorted rows with x_j > x_i (i < j):
    ``cost_j <= cost_i  <=>  slope_ij <= s``.  The count of ordered pairs with
    ``cost_j <= cost_i`` is ``strict_inversions(cost) + equal_pairs(cost)``.
    Pairs sharing an x value are excluded by subtracting the (s-independent)
    within-tie-group count.
    """
    srt, inv = _sorted_runs_inversions(cost)
    return inv + _equal_pairs_sorted(srt) - x_tie_le


def _x_tie_within_counts(x_sorted: np.ndarray, y_sorted: np.ndarray) -> int:
    """Number of within-x-tie ordered pairs with y_j <= y_i (s independent)."""
    if x_sorted.size < 2:
        return 0
    d = np.r_[True, x_sorted[1:] != x_sorted[:-1]]
    starts = np.flatnonzero(d)
    ends = np.r_[starts[1:], x_sorted.size]
    total = 0
    for a, b in zip(starts, ends):
        if b - a < 2:
            continue
        srt, inv = _sorted_runs_inversions(y_sorted[a:b])
        total += inv + _equal_pairs_sorted(srt)
    return total


def median_pairwise_slope(x: np.ndarray, y: np.ndarray,
                          rel_tol: float = 1.0e-11,
                          max_iter: int = 96) -> float:
    """Median of all pairwise slopes (Theil-Sen slope) for x-sorted inputs.

    Equivalent to ``np.median`` over ``(y_j-y_i)/(x_j-x_i)`` for every pair
    with ``x_i != x_j``.  Duplicate x values are excluded exactly as scipy's
    ``theilslopes`` excludes ``deltax == 0``.

    The median value is located by counting pairs below a candidate slope,
    which requires no ``n^2`` array.  The returned value is accurate to
    ``rel_tol`` relative to the exact median value.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    order = np.argsort(x, kind="stable")
    xs = np.ascontiguousarray(x[order])
    ys = np.ascontiguousarray(y[order])
    has_ties = bool(np.any(xs[1:] == xs[:-1] )) if xs.size > 1 else False
    x_tie_le = _x_tie_within_counts(xs, ys) if has_ties else 0

    # Total number of slopes = #pairs with distinct x.
    n_pairs = xs.size * (xs.size - 1) // 2
    if has_ties:
        n_pairs -= _x_equal_pairs_count(xs)
    if n_pairs <= 0:
        return float("nan")
    if n_pairs == 1:
        for i in range(xs.size - 1):
            if xs[i + 1] != xs[i]:
                return float((ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]))
        return float("nan")

    # Bracketing bound: |slope| <= (max y - min y) / min positive dx.
    pos_dx = np.diff(xs)
    pos_dx = pos_dx[pos_dx > 0.0]
    span_y = float(np.max(ys) - np.min(ys))
    if span_y <= 0.0:
        return 0.0
    bound = span_y / float(np.min(pos_dx))
    lo = -bound
    hi = bound

    # np.median of an even-length set averages the two central values.
    k_lo = (n_pairs - 1) // 2
    k_hi = n_pairs // 2

    def value_at_rank(k: int) -> float:
        lo_, hi_ = lo, hi
        while _count_pairs_le_slope(xs, ys - lo_ * xs, x_tie_le) > k:
            if lo_ == 0.0:
                return 0.0
            lo_ = lo_ * 2.0 if lo_ < 0 else lo_ / 2.0
        while _count_pairs_le_slope(xs, ys - hi_ * xs, x_tie_le) <= k:
            hi_ = hi_ * 2.0 if hi_ > 0 else hi_ / 2.0
        for _ in range(max_iter):
            mid = 0.5 * (lo_ + hi_)
            c = _count_pairs_le_slope(xs, ys - mid * xs, x_tie_le)
            if c <= k:
                lo_ = mid
            else:
                hi_ = mid
            if hi_ - lo_ <= rel_tol * max(1.0, abs(mid)):
                break
        return 0.5 * (lo_ + hi_)

    s_lo = value_at_rank(k_lo)
    s_hi = value_at_rank(k_hi)
    return 0.5 * (s_lo + s_hi)


def theilslopes_fast(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """Exact Theil-Sen slope and intercept with scipy-compatible semantics.

    Intercept is ``median(y) - slope * median(x)`` (scipy's default method).
    """
    slope = median_pairwise_slope(x, y)
    if not np.isfinite(slope):
        return float("nan"), float("nan")
    intercept = float(np.median(y) - slope * np.median(x))
    return slope, intercept


def uniform_subsample(values: np.ndarray, budget: int) -> np.ndarray:
    """Deterministic evenly spaced subsample (no RNG, no order dependence)."""
    n = values.size
    if n <= budget:
        return values
    idx = (np.arange(budget, dtype=np.int64) * n) // budget
    return values[idx]


@dataclass
class ChannelFit:
    alpha: float
    sigma_r: float
    n_blocks: int
    n_used: int
    n_regression: int
    warnings: list[str] = field(default_factory=list)


def fit_channel(means: np.ndarray, variances: np.ndarray, gradients: np.ndarray,
                saturated: np.ndarray, min_samples: int = MIN_SAMPLES,
                theil_budget: int = THEIL_BUDGET) -> ChannelFit:
    """Flat-patch selection and Theil-Sen fit for one channel's block grid."""
    warnings_: list[str] = []
    flat = (~saturated) & (gradients <= max(1.5 * np.median(gradients), 1.0e-6))
    m = means[flat]
    v = variances[flat]
    n_blocks = int(means.size)
    n_used = int(m.size)
    if m.size < min_samples:
        warnings_.append(f"only {m.size} flat patches (need >= {min_samples})")
        if m.size < 4:
            return ChannelFit(np.nan, np.nan, n_blocks, n_used, 0, warnings_)
    ms = uniform_subsample(m, theil_budget)
    vs = uniform_subsample(v, theil_budget)
    if ms.size > 1 and bool(np.all(np.isfinite(ms))) and bool(np.all(np.isfinite(vs))):
        slope, intercept = theilslopes_fast(vs, ms)
    else:
        slope, intercept = np.nan, np.nan
    if not np.isfinite(slope) or slope < 0.0:
        warnings_.append("non-positive shot coefficient; using constant-noise model")
        slope = 0.0
        intercept = float(np.median(v))
    if not np.isfinite(intercept) or intercept < 0.0:
        intercept = 0.0
    return ChannelFit(float(slope), float(np.sqrt(max(intercept, 0.0))),
                      n_blocks, n_used, int(ms.size), warnings_)
