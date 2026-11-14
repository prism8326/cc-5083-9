"""Core algorithm tests using synthetic observations (no clean reference is
ever passed to the runtime pipeline)."""
from __future__ import annotations

import numpy as np
import pytest

from lowlight.calibration import estimate_noise
from lowlight.color import srgb_to_linear, linear_to_srgb
from lowlight.denoise import denoise
from lowlight.lifting import LiftParams, lift_shadows
from lowlight.metrics import all_metrics
from lowlight.noise import NoiseParams, gat, inverse_gat
from lowlight.pipeline import PipelineConfig, run_pipeline
from lowlight.synthetic import (make_flat_colour, make_observation,
                                make_pure_black)

ALPHA, SIGMA = 0.004, 0.006


@pytest.fixture(scope="module")
def fixture256():
    return make_observation(size=256, exposure=0.5,
                            alpha=ALPHA, sigma_r=SIGMA, seed=3)


def test_gat_stabilizes_variance_to_unit():
    # At reasonable photon counts the normalized GAT yields ~unit variance;
    # in very dark regions Poisson discreteness inflates it, which is an
    # intrinsic property of the transform rather than an implementation bug.
    rng = np.random.default_rng(0)
    params = NoiseParams(ALPHA, SIGMA)
    for x in (0.1, 0.4):
        y = ALPHA * rng.poisson(np.full((4096,), x / ALPHA)) \
            + rng.normal(0, SIGMA, 4096)
        z = gat(y, params)
        assert 0.8 < z.var() < 1.25, f"stabilized variance {z.var()} at x={x}"


def test_inverse_gat_unbiased_above_few_photons():
    rng = np.random.default_rng(1)
    params = NoiseParams(ALPHA, SIGMA)
    for x in (0.05, 0.2, 0.6):
        y = ALPHA * rng.poisson(np.full((20000,), x / ALPHA)) \
            + rng.normal(0, SIGMA, 20000)
        hat = inverse_gat(gat(y, params), params)
        assert abs(hat.mean() - x) < 0.2 * x
        assert np.all(np.isfinite(hat))


def test_inverse_gat_non_negative_even_in_black():
    rng = np.random.default_rng(2)
    params = NoiseParams(ALPHA, SIGMA)
    y = rng.normal(0, SIGMA, 10000)  # pure read noise around black
    hat = inverse_gat(gat(np.clip(y, 0, 1), params), params)
    assert np.all(hat >= 0.0) and np.all(np.isfinite(hat))


def test_estimate_recovers_calibration_order(fixture256):
    params, info = estimate_noise(fixture256["observed_linear"])
    # Theil-Sen on 256px data is noisy; require correct order of magnitude and
    # correct separation of the two noise terms.
    assert 0.2 * ALPHA < params.alpha < 5 * ALPHA
    assert 0.2 * SIGMA < params.sigma_r < 5 * SIGMA
    assert info.n_used >= 24
    assert params.source == "estimate"


def test_estimation_is_size_invariant():
    """Same noise level on 192px and 384px frames must give similar params."""
    small = make_observation(size=192, exposure=0.3, alpha=ALPHA,
                             sigma_r=SIGMA, seed=11)
    large = make_observation(size=384, exposure=0.3, alpha=ALPHA,
                             sigma_r=SIGMA, seed=11)
    p1, _ = estimate_noise(small["observed_linear"])
    p2, _ = estimate_noise(large["observed_linear"])
    assert abs(p1.alpha - p2.alpha) < 2.0 * ALPHA
    assert abs(p1.sigma_r - p2.sigma_r) < 1.5 * SIGMA


def test_denoise_reduces_noise_and_keeps_means(fixture256):
    obs = fixture256["observed_linear"]
    params = NoiseParams(ALPHA, SIGMA, source="calibration")
    den, stats = denoise(obs, params, strength=1.0)
    # The dedicated uniform calibration area of the synthetic scene.
    flat = (slice(144, 176), slice(200, 232))
    assert den[flat].var() < 0.5 * obs[flat].var()
    # Global mean must be nearly preserved (no brightness bias from denoise).
    assert abs(den.mean() - obs.mean()) < 0.015
    assert stats["per_channel"][0]["gat_var_after"] < stats["per_channel"][0]["gat_var_before"]


def test_strength_zero_is_near_identity(fixture256):
    params = NoiseParams(ALPHA, SIGMA)
    den, _ = denoise(fixture256["observed_linear"], params, strength=0.0)
    # With no thresholding only the fixed GAT round-trip/quantization remains.
    assert np.sqrt(np.mean((den - fixture256["observed_linear"]) ** 2)) < 0.03


def test_pipeline_thin_line_retention(fixture256):
    cfg = PipelineConfig(alpha=ALPHA, sigma_r=SIGMA, strength=1.0)
    result = run_pipeline(fixture256["observed_srgb"], cfg)
    m = all_metrics(fixture256, result, size=256)
    # At least 60% of the contrast present in the exposed scene survives.
    assert m["denoise"]["line_contrast"]["retention"] > 0.6


def test_pipeline_shadow_mean_and_black_patch(fixture256):
    cfg = PipelineConfig(alpha=ALPHA, sigma_r=SIGMA, strength=1.0)
    result = run_pipeline(fixture256["observed_srgb"], cfg)
    m = all_metrics(fixture256, result, size=256)
    # Denoised shadow mean must track the exposed reference within 35%.
    assert abs(m["denoise"]["shadow_mean"]["relative_bias"]) < 0.35
    # Pure black must not be turned into an invented bright texture: the mean
    # stays near zero even though individual noisy pixels may be lifted.
    assert m["black_patch"]["mean"] < 0.02
    # The lift must respect its gain cap and must not clip highlights.
    assert m["lift"]["applied_gain"] <= 8.0
    assert m["lift"]["clip_fraction"] < 0.01


def test_pipeline_colour_bias_on_flat_patch(fixture256):
    cfg = PipelineConfig(alpha=ALPHA, sigma_r=SIGMA, strength=1.0)
    result = run_pipeline(fixture256["observed_srgb"], cfg)
    m = all_metrics(fixture256, result, size=256)
    cb = m["denoise"]["colour_bias_red_patch"]
    assert cb["max_abs_bias"] < 0.05
    assert abs(cb["chroma_angle_change_rad"]) < 0.25


def test_pure_black_frame_is_flagged_not_invented():
    black = make_pure_black(size=128, sigma_r=SIGMA)
    cfg = PipelineConfig(alpha=ALPHA, sigma_r=SIGMA, strength=1.0)
    result = run_pipeline(black, PipelineConfig(
        strength=1.0, alpha=ALPHA, sigma_r=SIGMA, input_is_linear=True))
    rel = result.stats["reliability"]
    # Most of the frame sits below the read-noise floor and is flagged.
    assert rel["black_floor"] > 0.5
    assert result.rgb_linear_restored.mean() < 0.03


def test_flat_single_colour_keeps_hue():
    img = make_flat_colour(size=128, rgb_linear=(0.08, 0.03, 0.02),
                           alpha=ALPHA, sigma_r=SIGMA)
    result = run_pipeline(img, PipelineConfig(
        strength=1.0, alpha=ALPHA, sigma_r=SIGMA, input_is_linear=True))
    rest = result.rgb_linear_restored.reshape(-1, 3).mean(axis=0)
    # Red channel must remain clearly dominant (hue ordering preserved).
    assert rest[0] > rest[1] and rest[1] >= rest[2] * 0.5
    assert rest.std() > 0  # not collapsed to grey


def test_lift_gain_is_capped_and_highlights_untouched():
    rng = np.random.default_rng(2)
    img = np.clip(rng.normal(0.02, 0.005, (64, 64, 3)), 0, 1)
    img[0:8, 0:8] = 0.9  # highlight patch
    lifted, stats = lift_shadows(img, LiftParams(max_gain=4.0,
                                                 shadow_quantile=0.5,
                                                 rolloff_quantile=0.95))
    assert stats["applied_shadow_gain"] <= 4.0
    # The bright patch keeps essentially its original level.
    assert abs(lifted[0:8, 0:8].mean() - 0.9) < 0.06
    assert lifted.max() <= 1.0


def test_saturation_is_flagged():
    img = np.full((64, 64, 3), 0.2)
    img[10:20, 10:20] = 1.0
    result = run_pipeline(img, PipelineConfig(
        strength=1.0, alpha=ALPHA, sigma_r=SIGMA, input_is_linear=True))
    assert result.stats["reliability"]["saturated"] > 0.0


def test_wavelet_fallback_runs_and_reduces_variance(monkeypatch):
    import importlib
    denoise_mod = importlib.import_module("lowlight.denoise")
    monkeypatch.setattr(denoise_mod, "_HAVE_CV2", False)
    img = make_observation(size=128, alpha=ALPHA, sigma_r=SIGMA, seed=5)
    params = NoiseParams(ALPHA, SIGMA)
    # An explicit nlm request must transparently fall back to wavelets.
    den, stats = denoise_mod.denoise(img["observed_linear"], params,
                                     strength=1.0, method="nlm")
    assert stats["method"] == "wavelet"
    assert den.shape == img["observed_linear"].shape
    assert den.var() < img["observed_linear"].var()


def test_invalid_calibration_rejected():
    with pytest.raises(ValueError):
        run_pipeline(np.zeros((16, 16, 3)),
                     PipelineConfig(alpha=-1.0, sigma_r=0.01))


def test_estimation_warns_on_uninformative_black_frame():
    black = make_pure_black(size=96, sigma_r=SIGMA)
    params, info = estimate_noise(black)
    assert info.warnings  # lack of flat patches / undetectable terms reported
    # A conservative non-negative read-noise floor is returned, never NaN.
    assert np.isfinite(params.alpha) and params.alpha >= 0.0
    assert np.isfinite(params.sigma_r) and params.sigma_r > 0.0
    assert params.source == "estimate"


def test_zero_image_pipeline_does_not_invent_signal():
    r = run_pipeline(np.zeros((64, 64, 3)), PipelineConfig(strength=1.0))
    assert r.rgb_linear_restored.max() < 1e-6
    assert r.stats["reliability"]["reliable_fraction"] == 0.0


def test_srgb_roundtrip():
    x = np.linspace(0, 1, 50)
    np.testing.assert_allclose(linear_to_srgb(srgb_to_linear(x)), x, atol=1e-10)


# ---------------------------------------------------------------------------
# Optimization invariants: tiling, caching, exact Theil-Sen, determinism.
# ---------------------------------------------------------------------------
from lowlight.cache import CacheConfig
from lowlight.calibration import estimate_noise_legacy
from lowlight.denoise import nlm_halo, _nlm_channel, _nlm_channel_tiled
from lowlight.pipeline import reset_session_cache, run_pipeline_legacy
from lowlight.stats import (compute_patch_statistics, content_digest,
                            median_pairwise_slope, theilslopes_fast,
                            uniform_subsample)
from lowlight.workloads import make_workload


def _tiled_vs_untiled(img, side, **kw):
    a = run_pipeline(img, PipelineConfig(alpha=ALPHA, sigma_r=SIGMA,
                                         tile_pixels=0, **kw))
    b = run_pipeline(img, PipelineConfig(alpha=ALPHA, sigma_r=SIGMA,
                                         tile_pixels=1, tile_side=side, **kw))
    return a, b


@pytest.mark.parametrize("side", [96, 128, 181, 256])
def test_tiled_nlm_is_bit_exact(fixture256, side):
    """Cores are disjoint and halo >= search//2 + template//2, so tiles must
    reproduce the untiled result exactly (no seam, no double write)."""
    a, b = _tiled_vs_untiled(fixture256["observed_srgb"], side)
    assert np.array_equal(a.rgb_linear_denoised, b.rgb_linear_denoised)


def test_tiled_nlm_odd_shape_is_bit_exact():
    rng = np.random.default_rng(4)
    img = np.clip(rng.normal(0.08, 0.03, (333, 517, 3)), 0, 1)
    a, b = _tiled_vs_untiled(img, 128)
    assert np.array_equal(a.rgb_linear_denoised, b.rgb_linear_denoised)


def test_halo_formula_covers_min_window():
    # For the smallest supported windows the halo must be a positive ring
    # (zero halo would drop the search neighbours that shape each output).
    assert nlm_halo(3, 5) == max(1, 5 // 2 + 3 // 2)
    assert nlm_halo(7, 21) == 13


def test_fast_theilsen_matches_scipy_on_random_data():
    from scipy import stats
    rng = np.random.default_rng(5)
    for n in (50, 200, 1000, 2500):
        x = rng.uniform(0.0, 0.3, n)
        y = 0.004 * x + 0.000036 + rng.normal(0.0, 5e-6, n)
        slope, intercept = theilslopes_fast(y, x)
        ref = stats.theilslopes(y, x)
        assert abs(slope - ref.slope) <= 1e-8 * max(abs(ref.slope), 1e-12)
        assert abs(intercept - ref.intercept) <= 1e-10


def test_fast_theilsen_handles_duplicate_x():
    from scipy import stats
    rng = np.random.default_rng(6)
    x = np.round(rng.uniform(0.0, 0.3, 400), 2)  # many equal x
    y = 0.005 * x + 0.00002 + rng.normal(0.0, 3e-6, 400)
    slope, intercept = theilslopes_fast(y, x)
    ref = stats.theilslopes(y, x)
    assert abs(slope - ref.slope) <= 1e-8 * max(abs(ref.slope), 1e-12)


def test_estimator_matches_legacy_where_legacy_is_feasible(fixture256):
    """At 256px the legacy O(n^2) fit is still runnable; parameters must agree."""
    new_p, new_i = None, None
    from lowlight.calibration import estimate_noise
    new_p, new_i = estimate_noise(fixture256["observed_linear"])
    old_p, old_i = estimate_noise_legacy(fixture256["observed_linear"])
    assert abs(new_p.alpha - old_p.alpha) < 5e-6
    assert abs(new_p.sigma_r - old_p.sigma_r) < 5e-6
    assert new_i.n_used == old_i.n_used


def test_patch_statistics_chunking_is_bit_identical():
    import lowlight.stats as S
    w = make_workload("dense_texture", 200, seed=2)
    base = compute_patch_statistics(w["observed_linear"])
    orig = S.BLOCK_CHUNK_ROWS
    try:
        for rows in (1, 9, 500):
            S.BLOCK_CHUNK_ROWS = rows
            p = compute_patch_statistics(w["observed_linear"])
            assert np.array_equal(p.means, base.means)
            assert np.array_equal(p.variances, base.variances)
    finally:
        S.BLOCK_CHUNK_ROWS = orig


def test_uniform_subsample_is_deterministic_and_bounded():
    v = np.arange(10000)
    s1 = uniform_subsample(v, 100)
    s2 = uniform_subsample(v.copy(), 100)
    assert s1.size == 100 and np.array_equal(s1, s2)
    assert s1[0] == 0 and s1[-1] == 9900


def test_cache_reuses_and_invalidates_on_parameter_change(fixture256):
    reset_session_cache()
    img = fixture256["observed_srgb"]
    cfgc = CacheConfig()
    r1 = run_pipeline(img, PipelineConfig(alpha=ALPHA, sigma_r=SIGMA,
                                          cache=cfgc))
    assert r1.stats["timing_s"]["denoise_cache_hit"] is False
    r2 = run_pipeline(img, PipelineConfig(alpha=ALPHA, sigma_r=SIGMA,
                                          cache=cfgc))
    assert r2.stats["timing_s"]["denoise_cache_hit"] is True
    assert np.array_equal(r2.rgb_linear_denoised, r1.rgb_linear_denoised)

    # A different strength must invalidate the denoise entry but keep the GAT.
    r3 = run_pipeline(img, PipelineConfig(alpha=ALPHA, sigma_r=SIGMA,
                                          strength=1.5, cache=cfgc))
    assert r3.stats["timing_s"]["denoise_cache_hit"] is False
    assert r3.stats["timing_s"]["gat_cache_hit"] is True
    assert not np.array_equal(r3.rgb_linear_denoised, r1.rgb_linear_denoised)

    # A different template size must also invalidate the denoise entry.
    r4 = run_pipeline(img, PipelineConfig(alpha=ALPHA, sigma_r=SIGMA,
                                          strength=1.5, template_size=5,
                                          cache=cfgc))
    assert r4.stats["timing_s"]["denoise_cache_hit"] is False

    # A different alpha invalidates GAT and hence the denoise entry.
    r5 = run_pipeline(img, PipelineConfig(alpha=0.005, sigma_r=SIGMA,
                                          cache=cfgc))
    assert r5.stats["timing_s"]["gat_cache_hit"] is False


def test_cached_and_uncached_results_are_identical(fixture256):
    reset_session_cache()
    img = fixture256["observed_srgb"]
    a = run_pipeline(img, PipelineConfig(alpha=ALPHA, sigma_r=SIGMA,
                                         use_cache=False))
    b = run_pipeline(img, PipelineConfig(alpha=ALPHA, sigma_r=SIGMA,
                                         cache=CacheConfig()))
    assert np.array_equal(a.rgb_linear_denoised, b.rgb_linear_denoised)
    assert np.array_equal(a.rgb_linear_restored, b.rgb_linear_restored)
    assert np.array_equal(a.flags, b.flags)


def test_new_pipeline_matches_legacy_pixels_with_declared_calibration(fixture256):
    cfg = PipelineConfig(alpha=ALPHA, sigma_r=SIGMA, strength=1.0)
    new = run_pipeline(fixture256["observed_srgb"], cfg)
    old = run_pipeline_legacy(fixture256["observed_srgb"], cfg)
    assert np.array_equal(new.rgb_linear_denoised, old.rgb_linear_denoised)
    assert np.array_equal(new.rgb_linear_restored, old.rgb_linear_restored)
