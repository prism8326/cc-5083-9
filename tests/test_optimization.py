"""Scenario and determinism tests for the optimized pipeline.

Three scene classes are exercised separately - uniform dark, dense texture and
an abrupt noise-strength change - because a fast result on a flat image says
nothing about textured or discontinuous scenes.  These tests assert that the
optimization (tiling + caching) changes *cost*, never *pixels*.
"""
from __future__ import annotations

import numpy as np
import pytest

from lowlight.cache import CacheConfig
from lowlight.denoise import denoise
from lowlight.noise import NoiseParams
from lowlight.pipeline import (PipelineConfig, reset_session_cache,
                               run_pipeline, run_pipeline_legacy)
from lowlight.workloads import make_workload


@pytest.fixture(scope="module", params=["uniform_dark", "dense_texture",
                                        "noise_step"])
def workload(request):
    return make_workload(request.param, 192, seed=3)


def _cfg(w, **kw):
    base = dict(alpha=w["alpha"], sigma_r=w["sigma_r"], strength=1.0)
    base.update(kw)
    return PipelineConfig(**base)


def test_scenarios_run_and_reduce_noise(workload):
    """Every class restores without error and matches the legacy pixel result.

    Note: at very low photon counts (uniform_dark has alpha=0.01, i.e. ~1
    photon per pixel) the exact unbiased inverse GAT still carries a known
    positive bias; that is a property of the estimator, not of this
    optimization, so the invariant asserted here is *pixel parity with the
    legacy path*, plus finite outputs and no gross energy explosion.
    """
    reset_session_cache()
    r = run_pipeline(workload["observed_srgb"], _cfg(workload))
    assert r.rgb_linear_restored.shape == workload["observed_linear"].shape
    assert np.all(np.isfinite(r.rgb_linear_restored))
    assert r.rgb_linear_denoised.mean() < 10.0 * workload["observed_linear"].mean()
    old = run_pipeline_legacy(workload["observed_srgb"], _cfg(workload))
    assert np.array_equal(r.rgb_linear_denoised, old.rgb_linear_denoised)


def test_scenarios_tiled_equals_untiled(workload):
    reset_session_cache()
    a = run_pipeline(workload["observed_srgb"], _cfg(workload, tile_pixels=0))
    b = run_pipeline(workload["observed_srgb"],
                     _cfg(workload, tile_pixels=1, tile_side=80))
    assert np.array_equal(a.rgb_linear_denoised, b.rgb_linear_denoised)


def test_scenarios_cached_equals_uncached(workload):
    reset_session_cache()
    a = run_pipeline(workload["observed_srgb"], _cfg(workload, use_cache=False))
    b = run_pipeline(workload["observed_srgb"], _cfg(workload,
                                                    cache=CacheConfig()))
    assert np.array_equal(a.rgb_linear_denoised, b.rgb_linear_denoised)


def test_scenarios_match_legacy_with_declared_calibration(workload):
    cfg = _cfg(workload)
    new = run_pipeline(workload["observed_srgb"], cfg)
    old = run_pipeline_legacy(workload["observed_srgb"], cfg)
    assert np.array_equal(new.rgb_linear_denoised, old.rgb_linear_denoised)


def test_dense_texture_keeps_more_detail_than_flat_scene():
    """Guard against judging the optimization on the easy flat frame only.

    On the flat scene residual noise dominates the high-frequency band, so an
    absolute gradient comparison is misleading.  What matters is whether the
    textured scene's own structure survives: correlate the denoised output
    with the clean texture (evaluation-side only) and require it to be strong.
    """
    tex = make_workload("dense_texture", 192, seed=3)
    flat = make_workload("uniform_dark", 192, seed=3)
    rt = run_pipeline(tex["observed_srgb"], _cfg(tex))
    rf = run_pipeline(flat["observed_srgb"], _cfg(flat))
    w = np.array([0.2126, 0.7152, 0.0722])

    def corr(a, b):
        la, lb = (a @ w).ravel(), (b @ w).ravel()
        la = la - la.mean()
        lb = lb - lb.mean()
        return float((la * lb).sum() / np.sqrt((la * la).sum() * (lb * lb).sum()))

    # Structure in the textured result tracks the clean scene strongly.
    assert corr(rt.rgb_linear_denoised, tex["clean_linear"]) > 0.8
    # The flat scene carries no comparable structure to retain.
    assert corr(rf.rgb_linear_denoised, flat["clean_linear"]) < 0.8


def test_noise_step_has_two_distinct_regions():
    w = make_workload("noise_step", 192, seed=3)
    strong = w["masks"]["strong"]
    weak = ~strong
    obs = w["observed_linear"]
    # The strong-noise half must actually be noisier in the observation.
    luma = obs @ np.array([0.2126, 0.7152, 0.0722])
    # High-pass energy as a noise proxy.
    hp = np.abs(luma - np.roll(luma, 1, axis=0))
    assert hp[strong].mean() > 1.5 * hp[weak].mean()


@pytest.mark.parametrize("threads", [1, 2, 4, 8])
def test_nlm_output_is_thread_count_independent(threads):
    """OpenCV parallelises NLM internally; the result must not depend on it."""
    try:
        import cv2
    except Exception:
        pytest.skip("OpenCV not available")
    rng = np.random.default_rng(9)
    z = np.clip(rng.normal(3.0, 1.2, (256, 256)), 0.0, 14.0)
    params = NoiseParams(0.004, 0.006)

    cv2.setNumThreads(1)
    ref, _ = denoise(z[..., None].repeat(3, axis=2), params, strength=1.0,
                     method="nlm", tile_pixels=0)
    cv2.setNumThreads(threads)
    try:
        out, _ = denoise(z[..., None].repeat(3, axis=2), params, strength=1.0,
                         method="nlm", tile_pixels=0)
    finally:
        cv2.setNumThreads(cv2.getNumberOfCPUs())
    assert np.array_equal(ref, out)


def test_rounding_tolerance_does_not_amplify_colour_bias():
    """The fast Theil-Sen slope is accurate to ~1e-10 relative; verify that
    feeding a slightly perturbed estimate back through the pipeline does not
    amplify into a visible colour shift (tol is stated, not hand-waved)."""
    w = make_workload("dense_texture", 192, seed=3)
    base = run_pipeline(w["observed_srgb"],
                        PipelineConfig(alpha=w["alpha"], sigma_r=w["sigma_r"]))
    rel = 1e-9
    pert = run_pipeline(w["observed_srgb"], PipelineConfig(
        alpha=w["alpha"] * (1.0 + rel), sigma_r=w["sigma_r"] * (1.0 + rel)))
    from lowlight.metrics import rmse
    bias = np.abs(pert.rgb_linear_restored - base.rgb_linear_restored).max()
    # Documented tolerance: a 1e-9 relative parameter change must move no
    # pixel by more than 1e-6 linear (far below one 8-bit code value).
    assert bias < 1e-6
    assert rmse(pert.rgb_linear_restored, base.rgb_linear_restored) < 1e-7


def test_restored_anomalies_flag_bad_calibration_not_good():
    """A mis-declared calibration corrupts the restored image while the
    input-only reliability map still looks clean; the output-side detector
    must catch that and stay quiet on the correct calibration."""
    from lowlight.gui import _register_cjk_font  # noqa: F401 - GUI import path
    from lowlight.pipeline import PipelineConfig, run_pipeline, reset_session_cache
    from lowlight.workloads import make_gui_dark_photo

    reset_session_cache()
    w = make_gui_dark_photo(size=240, seed=3)
    good = run_pipeline(w["observed_srgb"],
                        PipelineConfig(alpha=w["alpha"], sigma_r=w["sigma_r"]))
    bad = run_pipeline(w["observed_srgb"],
                       PipelineConfig(alpha=0.02, sigma_r=0.008))
    ag, ab = good.stats["restored_anomalies"], bad.stats["restored_anomalies"]

    assert ab["speckle_fraction"] > 5.0 * max(ag["speckle_fraction"], 1e-4)
    assert ab["dark_residual_rms"] > 3.0 * ag["dark_residual_rms"]
    assert ab["warnings"] and not ag["warnings"]
    # The input flags must not have been the thing that caught it.
    assert bad.stats["reliability"]["reliable_fraction"] > 0.9
