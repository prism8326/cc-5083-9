"""End-to-end low-light restoration pipeline and image I/O.

Stages (all in linear-light space after decoding):

    input image
      -> decode to linear RGB
      -> noise calibration (provided parameters, or robust estimation)
      -> GAT variance stabilization
      -> edge-preserving denoise (unit-variance domain, size-invariant)
      -> exact unbiased inverse GAT
      -> constrained shadow lift (bounded gain, per-channel colour preserved)
      -> reliability map + residual statistics
      -> sRGB encode / export

The clean reference is never part of this path; it is only used by the
offline test harness in ``tests/``.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .calibration import (estimate_noise, estimate_noise_from_stats,
                          estimate_noise_legacy)
from .cache import CacheConfig, PipelineCache
from .color import srgb_to_linear, encode_display, linear_luma
from .denoise import (DEFAULT_TILE_PIXELS, DEFAULT_TILE_SIDE, denoise,
                      nlm_halo)
from .lifting import LiftParams, lift_shadows
from .noise import NoiseParams, gat
from .reliability import (build_reliability, flags_to_rgb,
                          restored_anomalies)
from .stats import (BLOCK_SIZE, BLOCK_STRIDE, MAX_BLOCKS, MIN_SAMPLES,
                    THEIL_BUDGET, PatchStatistics, compute_patch_statistics,
                    content_digest, fit_channel)

try:
    import cv2
    _HAVE_CV2 = True
except Exception:  # pragma: no cover
    _HAVE_CV2 = False

try:
    from PIL import Image
    _HAVE_PIL = True
except Exception:  # pragma: no cover
    _HAVE_PIL = False

_SESSION_CACHE: Optional[PipelineCache] = None


@dataclass
class PipelineConfig:
    strength: float = 1.0            # denoising in stabilized-sigma units
    method: str = "nlm"
    alpha: Optional[float] = None    # provided calibration, linear units
    sigma_r: Optional[float] = None
    input_is_linear: bool = False
    lift: Optional[LiftParams] = None
    sat_threshold: float = 0.995
    max_side: Optional[int] = None   # optional display/processing cap
    # Large-frame strategy: tile NLM above this pixel count; the halo is
    # derived from both window sizes (see lowlight.denoise.nlm_halo).
    tile_pixels: int = DEFAULT_TILE_PIXELS
    tile_side: int = DEFAULT_TILE_SIDE
    template_size: int = 7
    search_size: int = 21
    # Reuse of stage results across parameter changes.
    cache: Optional[CacheConfig] = None
    use_cache: bool = True


@dataclass
class PipelineResult:
    rgb_linear_obs: np.ndarray
    rgb_linear_denoised: np.ndarray
    rgb_linear_restored: np.ndarray
    residual_linear: np.ndarray
    flags: np.ndarray
    params: NoiseParams
    config: PipelineConfig
    stats: dict = field(default_factory=dict)
    cache: Optional[PipelineCache] = field(default=None, repr=False)

    # Convenience display arrays (8-bit sRGB).
    def display_original(self) -> np.ndarray:
        return encode_display(self.rgb_linear_obs)

    def display_restored(self) -> np.ndarray:
        return encode_display(self.rgb_linear_restored)

    def display_residual(self, gain: float | None = None) -> np.ndarray:
        """Signed residual centred on mid-grey.

        With ``gain=None`` the scale is chosen adaptively from the residual's
        own standard deviation (about +/-3 sigma spans the byte range), so the
        view stays informative on both quiet and noisy frames.
        """
        r = self.residual_linear
        if gain is None:
            sigma = float(np.std(r))
            gain = 0.5 / max(sigma * 3.0, 1.0e-6)
        vis = np.clip(0.5 + r * gain, 0.0, 1.0)
        return (vis * 255.0 + 0.5).astype(np.uint8)

    def display_reliability(self) -> np.ndarray:
        return flags_to_rgb(self.flags)


def read_image(path: str) -> tuple[np.ndarray, dict]:
    """Read an image as float RGB in [0, 1] (sRGB-encoded bytes assumed)."""
    if _HAVE_CV2:
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(path)
        if raw.ndim == 2:
            raw = np.stack([raw] * 3, axis=-1)
        if raw.shape[2] == 4:
            raw = raw[..., :3]
        raw = raw[..., ::-1]  # BGR -> RGB
    elif _HAVE_PIL:
        with Image.open(path) as im:
            raw = np.asarray(im.convert("RGB"))
    else:
        raise RuntimeError("neither OpenCV nor Pillow is available for image I/O")

    max_val = float(np.iinfo(raw.dtype).max) if np.issubdtype(raw.dtype, np.integer) else 1.0
    rgb = raw.astype(np.float64) / max_val
    meta = {"path": os.path.abspath(path), "dtype": str(raw.dtype),
            "max_code_value": max_val, "shape": list(rgb.shape)}
    return rgb, meta


def write_image(path: str, rgb_u8: np.ndarray) -> None:
    if _HAVE_CV2:
        cv2.imwrite(path, rgb_u8[..., ::-1])
    elif _HAVE_PIL:
        Image.fromarray(rgb_u8).save(path)
    else:  # pragma: no cover
        raise RuntimeError("no image writer available")


def _split_stages(rgb01: np.ndarray, cfg: PipelineConfig,
                  cache: PipelineCache | None, timings: dict) -> dict:
    """Run the preprocessing stages, using the cache where keys match.

    ``timings`` accumulates wall-clock seconds *including* every cache build
    cost, so the report cannot hide setup work outside the measured interval.
    """
    t0 = time.perf_counter()
    rgb01 = np.clip(np.asarray(rgb01, dtype=np.float64), 0.0, 1.0)
    obs_linear = rgb01 if cfg.input_is_linear else srgb_to_linear(rgb01)
    timings["decode_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    digest = content_digest(obs_linear)
    timings["digest_s"] = time.perf_counter() - t0

    warnings: list[str] = []
    estimation = None
    if cfg.alpha is not None and cfg.sigma_r is not None:
        if cfg.alpha < 0.0 or cfg.sigma_r < 0.0:
            raise ValueError("calibration parameters must be non-negative")
        params = NoiseParams(float(cfg.alpha), float(cfg.sigma_r),
                             source="calibration")
    else:
        # Reusable patch statistics depend only on image content, block
        # geometry and saturation thresholds - not on filter/strength/lift.
        pkey = PipelineCache.patch_key(
            digest, BLOCK_SIZE, BLOCK_STRIDE,
            float(np.nextafter(1.0, 0.0)), 0.0, MAX_BLOCKS)
        t0 = time.perf_counter()
        patch_stats_obj, hit = _cache_lookup(cache, "patch", pkey)
        timings["patch_cache_hit"] = bool(hit)
        if patch_stats_obj is None:
            patch_stats_obj = compute_patch_statistics(obs_linear)
            _cache_store(cache, "patch", pkey, patch_stats_obj,
                         patch_stats_obj.as_dict()["bytes"])
        timings["patch_stats_s"] = time.perf_counter() - t0

        ekey = PipelineCache.estimate_key(
            patch_stats_obj.digest, patch_stats_obj.block_size,
            patch_stats_obj.stride, patch_stats_obj.sat_hi,
            patch_stats_obj.sat_lo, MAX_BLOCKS, MIN_SAMPLES, THEIL_BUDGET)
        t0 = time.perf_counter()
        est, hit = _cache_lookup(cache, "estimate", ekey)
        timings["estimate_cache_hit"] = bool(hit)
        if est is None:
            params, info = estimate_noise_from_stats(patch_stats_obj)
            info.patch_compute_s = float(patch_stats_obj.elapsed_s)
            _cache_store(cache, "estimate", ekey, (params, info), 64 * 1024)
        else:
            params, info = est
        info = dataclasses.replace(info)
        info.patch_compute_s = (0.0 if timings["patch_cache_hit"]
                                else float(patch_stats_obj.elapsed_s))
        info.cache_hit = bool(hit)
        estimation = info.as_dict()
        warnings.extend(info.warnings)
        timings["estimate_s"] = time.perf_counter() - t0

    # GAT depends on the noise parameters only.  A larger tile halo would only
    # widen the candidate-patch ring, so the transformed frame is one array
    # reused across strength changes without altering any pixel.
    gkey = PipelineCache.gat_key(digest, float(params.alpha),
                                 float(params.sigma_r),
                                 bool(cfg.input_is_linear))
    t0 = time.perf_counter()
    z_gat, hit = _cache_lookup(cache, "gat", gkey)
    timings["gat_cache_hit"] = bool(hit)
    if z_gat is None:
        z_gat = gat(obs_linear, params)
        _cache_store(cache, "gat", gkey, z_gat, z_gat.nbytes)
    timings["gat_s"] = time.perf_counter() - t0

    # Denoised GAT image depends on z, the filter and both window sizes.
    dkey = PipelineCache.denoise_key(
        gkey, cfg.method, float(cfg.strength),
        int(cfg.template_size) if cfg.method == "nlm" else None,
        int(cfg.search_size) if cfg.method == "nlm" else None)
    t0 = time.perf_counter()
    cached, hit = _cache_lookup(cache, "denoise", dkey)
    timings["denoise_cache_hit"] = bool(hit)
    if cached is not None:
        den_linear, den_stats = cached
        den_stats = copy.deepcopy(den_stats)
    else:
        den_linear, den_stats = denoise(
            obs_linear, params, strength=cfg.strength, method=cfg.method,
            template_size=cfg.template_size, search_size=cfg.search_size,
            tile_pixels=cfg.tile_pixels, tile_side=cfg.tile_side,
            z_gat=z_gat)
        den_stats.pop("z_den", None)
        _cache_store(cache, "denoise", dkey, (den_linear, den_stats),
                     den_linear.nbytes)
        den_stats = copy.deepcopy(den_stats)
    timings["denoise_s"] = time.perf_counter() - t0
    return {"obs_linear": obs_linear, "params": params,
            "estimation": estimation, "warnings": warnings,
            "den_linear": den_linear, "den_stats": den_stats}


def _cache_lookup(cache: PipelineCache | None, name: str, key: str):
    if cache is None or not cache.enabled:
        return None, False
    return cache.stage(name).get(key)


def _cache_store(cache: PipelineCache | None, name: str, key: str,
                 value, nbytes: int) -> None:
    if cache is not None and cache.enabled:
        cache.stage(name).put(key, value, nbytes)


def get_session_cache(config: CacheConfig | None = None) -> PipelineCache:
    """Process-wide cache shared by successive GUI runs with the same layout.

    Sliders re-run the pipeline repeatedly on the same image; sharing one
    cache lets patch statistics, the GAT image and the denoised output survive
    those calls.  :func:`reset_session_cache` drops everything.
    """
    global _SESSION_CACHE
    if _SESSION_CACHE is None:
        _SESSION_CACHE = PipelineCache(config)
    return _SESSION_CACHE


def reset_session_cache() -> None:
    global _SESSION_CACHE
    if _SESSION_CACHE is not None:
        _SESSION_CACHE.clear()
    _SESSION_CACHE = None


def run_pipeline(rgb01: np.ndarray, config: PipelineConfig | None = None,
                 cache: PipelineCache | None = None,
                 ) -> PipelineResult:
    """Run the full restoration on an sRGB-range float RGB array (HxWx3).

    ``cache`` (or ``config.cache``) enables reuse of patch statistics, the
    noise estimate, the GAT image and the denoised GAT image across calls
    with different denoise/lift parameters.  Every timing in ``stats`` is
    wall-clock inclusive of cache construction, and cache hit/miss counts are
    reported so warm reuse cannot be mistaken for algorithmic speedup.
    """
    cfg = config or PipelineConfig()
    rgb01 = np.asarray(rgb01, dtype=np.float64)
    if rgb01.ndim != 3 or rgb01.shape[2] != 3:
        raise ValueError("expected HxWx3 RGB input")
    rgb01 = np.clip(rgb01, 0.0, 1.0)

    if cache is None and cfg.use_cache:
        cache = get_session_cache(cfg.cache)
    elif not cfg.use_cache:
        cache = None
    elif not isinstance(cache, PipelineCache):
        raise TypeError("cache= must be a PipelineCache instance (or None)")

    timings: dict = {}
    stages = _split_stages(rgb01, cfg, cache, timings)
    obs_linear = stages["obs_linear"]
    params = stages["params"]
    estimation = stages["estimation"]
    warnings = stages["warnings"]
    den_linear = stages["den_linear"]
    den_stats = stages["den_stats"]

    t0 = time.perf_counter()
    # Keep slight negatives visible to the reliability stage, then lift.
    flags, rel_summary = build_reliability(obs_linear, den_linear, params,
                                           sat_threshold=cfg.sat_threshold)

    lift_params = cfg.lift or LiftParams()
    restored_linear, lift_stats = lift_shadows(
        np.maximum(den_linear, 0.0), lift_params)

    residual = den_linear - obs_linear
    timings["reliability_lift_s"] = time.perf_counter() - t0
    # Anomalies are described on the *restored* image: input-only flags keep
    # calling a corrupt restoration "reliable" when calibration is wrong.
    anomalies = restored_anomalies(restored_linear, obs_linear)

    stats = {
        "noise_params": params.as_dict(),
        "estimation": estimation,
        "denoise": den_stats,
        "lift": lift_stats,
        "reliability": rel_summary,
        "restored_anomalies": anomalies,
        "warnings": warnings + anomalies["warnings"],
        "timing_s": timings,
        "tiling": {"enabled": den_stats.get("tiling") is not None,
                   "halo_rule": "search_size//2 + template_size//2",
                   "nlm_halo": nlm_halo(int(cfg.template_size),
                                        int(cfg.search_size))},
        "cache": cache.summary() if cache is not None else {"enabled": False},
        "color": {
            "mean_rgb_observed_linear": [float(x) for x in obs_linear.reshape(-1, 3).mean(axis=0)],
            "mean_rgb_restored_linear": [float(x) for x in restored_linear.reshape(-1, 3).mean(axis=0)],
        },
        "luma_mean_observed": float(linear_luma(obs_linear).mean()),
        "luma_mean_restored": float(linear_luma(restored_linear).mean()),
    }
    return PipelineResult(obs_linear, den_linear, restored_linear, residual,
                          flags, params, cfg, stats, cache)


def run_pipeline_legacy(rgb01: np.ndarray, config: PipelineConfig | None = None
                        ) -> PipelineResult:
    """Original uncached path (legacy estimator, plain single-call NLM).

    Kept for pixel-error comparison against the optimized pipeline; the
    graphical and command-line entry points use :func:`run_pipeline`.
    """
    cfg = config or PipelineConfig()
    rgb01 = np.asarray(rgb01, dtype=np.float64)
    if rgb01.ndim != 3 or rgb01.shape[2] != 3:
        raise ValueError("expected HxWx3 RGB input")
    rgb01 = np.clip(rgb01, 0.0, 1.0)
    timings: dict = {}
    t0 = time.perf_counter()
    obs_linear = rgb01 if cfg.input_is_linear else srgb_to_linear(rgb01)
    timings["decode_s"] = time.perf_counter() - t0

    warnings: list[str] = []
    if cfg.alpha is not None and cfg.sigma_r is not None:
        if cfg.alpha < 0.0 or cfg.sigma_r < 0.0:
            raise ValueError("calibration parameters must be non-negative")
        params = NoiseParams(float(cfg.alpha), float(cfg.sigma_r),
                             source="calibration")
        estimation = None
    else:
        t0 = time.perf_counter()
        params, est = estimate_noise_legacy(obs_linear)
        estimation = est.as_dict()
        warnings.extend(est.warnings)
        timings["patch_stats_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    den_linear, den_stats = denoise(obs_linear, params,
                                    strength=cfg.strength, method=cfg.method,
                                    template_size=cfg.template_size,
                                    search_size=cfg.search_size,
                                    tile_pixels=0, tile_side=cfg.tile_side)
    den_stats.pop("z_den", None)
    timings["denoise_s"] = time.perf_counter() - t0
    flags, rel_summary = build_reliability(obs_linear, den_linear, params,
                                           sat_threshold=cfg.sat_threshold)
    lift_params = cfg.lift or LiftParams()
    restored_linear, lift_stats = lift_shadows(
        np.maximum(den_linear, 0.0), lift_params)
    residual = den_linear - obs_linear
    anomalies = restored_anomalies(restored_linear, obs_linear)
    stats = {
        "noise_params": params.as_dict(),
        "estimation": estimation,
        "denoise": den_stats,
        "lift": lift_stats,
        "reliability": rel_summary,
        "restored_anomalies": anomalies,
        "warnings": warnings + anomalies["warnings"],
        "timing_s": timings,
        "cache": {"enabled": False},
        "color": {
            "mean_rgb_observed_linear": [float(x) for x in obs_linear.reshape(-1, 3).mean(axis=0)],
            "mean_rgb_restored_linear": [float(x) for x in restored_linear.reshape(-1, 3).mean(axis=0)],
        },
        "luma_mean_observed": float(linear_luma(obs_linear).mean()),
        "luma_mean_restored": float(linear_luma(restored_linear).mean()),
    }
    return PipelineResult(obs_linear, den_linear, restored_linear, residual,
                          flags, params, cfg, stats, None)


def export_results(result: PipelineResult, out_dir: str,
                   base_name: str = "restored",
                   residual_gain: float | None = None) -> dict:
    """Write original/restored/residual/reliability views and a JSON record.

    ``residual_gain=None`` selects adaptive scaling from the residual sigma.
    """
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    files = {
        "original": result.display_original(),
        base_name: result.display_restored(),
        "residual": result.display_residual(gain=residual_gain),
        "reliability": result.display_reliability(),
    }
    for name, img in files.items():
        p = os.path.join(out_dir, f"{name}.png")
        write_image(p, img)
        paths[name] = os.path.abspath(p)

    record = {
        "config": {
            "strength": result.config.strength,
            "method": result.config.method,
            "alpha": result.config.alpha,
            "sigma_r": result.config.sigma_r,
            "input_is_linear": result.config.input_is_linear,
            "sat_threshold": result.config.sat_threshold,
            "lift": result.config.lift.as_dict() if result.config.lift else LiftParams().as_dict(),
        },
        "stats": result.stats,
        "outputs": paths,
    }
    json_path = os.path.join(out_dir, "restoration_record.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    paths["record"] = os.path.abspath(json_path)
    return paths
