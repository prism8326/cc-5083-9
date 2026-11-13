#!/usr/bin/env python3
"""Offline evaluation harness.

Generates known-clean scenes, exposes them at several exposure/noise settings,
runs the real restoration pipeline (which never receives the clean reference),
and reports thin-line retention, dark-region bias, colour bias, black-patch and
flat-colour behaviour, plus size-invariance of the noise interpretation.

Usage:  python scripts/evaluate.py [--out-dir runs/eval]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lowlight.calibration import estimate_noise
from lowlight.metrics import all_metrics
from lowlight.pipeline import PipelineConfig, export_results, run_pipeline
from lowlight.synthetic import (make_flat_colour, make_observation,
                                make_pure_black)

ALPHA, SIGMA = 0.004, 0.006


def evaluate_grid() -> list:
    rows = []
    for exposure in (0.25, 0.5, 1.0):
        for alpha, sigma in ((0.004, 0.006), (0.002, 0.012)):
            fx = make_observation(size=256, exposure=exposure,
                                  alpha=alpha, sigma_r=sigma, seed=4)
            for strength in (0.6, 1.0):
                cfg = PipelineConfig(alpha=alpha, sigma_r=sigma,
                                     strength=strength)
                result = run_pipeline(fx["observed_srgb"], cfg)
                m = all_metrics(fx, result, size=256)
                rows.append({
                    "exposure": exposure, "true_alpha": alpha,
                    "true_sigma_r": sigma, "strength": strength,
                    "line_retention": round(m["denoise"]["line_contrast"]["retention"], 3),
                    "shadow_relative_bias": (
                        None if m["denoise"]["shadow_mean"]["relative_bias"] is None
                        else round(m["denoise"]["shadow_mean"]["relative_bias"], 3)),
                    "colour_max_abs_bias": round(
                        m["denoise"]["colour_bias_red_patch"]["max_abs_bias"], 4),
                    "black_patch_mean": round(m["black_patch"]["mean"], 5),
                    "lift_gain": round(m["lift"]["applied_gain"], 2),
                    "lift_clip_fraction": round(m["lift"]["clip_fraction"], 5),
                    "denoise_rmse": round(m["denoise"]["rmse_linear"], 4),
                    "reliable_fraction": round(
                        result.stats["reliability"]["reliable_fraction"], 3),
                })
    return rows


def evaluate_edge_cases() -> dict:
    out = {}
    # Pure black frame.
    black = make_pure_black(size=128, sigma_r=SIGMA)
    rb = run_pipeline(black, PipelineConfig(
        strength=1.0, alpha=ALPHA, sigma_r=SIGMA, input_is_linear=True))
    out["pure_black"] = {
        "black_floor_flag_fraction": rb.stats["reliability"]["black_floor"],
        "restored_mean": float(rb.rgb_linear_restored.mean()),
        "restored_max": float(rb.rgb_linear_restored.max()),
    }
    # Single flat colour (dark warm tone).
    flat = make_flat_colour(size=128, rgb_linear=(0.08, 0.03, 0.02))
    rf = run_pipeline(flat, PipelineConfig(
        strength=1.0, alpha=ALPHA, sigma_r=SIGMA, input_is_linear=True))
    mean_rgb = rf.rgb_linear_restored.reshape(-1, 3).mean(axis=0)
    out["flat_single_colour"] = {
        "restored_mean_rgb": [float(x) for x in mean_rgb],
        "hue_order_preserved": bool(mean_rgb[0] > mean_rgb[1] > mean_rgb[2]),
        "not_collapsed_to_grey": bool(np.ptp(mean_rgb) > 1e-4),
    }
    # Fully saturated frame.
    sat = np.ones((96, 96, 3))
    rs = run_pipeline(sat, PipelineConfig(
        strength=1.0, alpha=ALPHA, sigma_r=SIGMA, input_is_linear=True))
    out["saturated_frame"] = {
        "saturated_flag_fraction": rs.stats["reliability"]["saturated"]}
    return out


def evaluate_size_invariance() -> dict:
    estimates = {}
    for size in (128, 192, 256, 384):
        fx = make_observation(size=size, exposure=0.5,
                              alpha=ALPHA, sigma_r=SIGMA, seed=11)
        params, info = estimate_noise(fx["observed_linear"])
        estimates[size] = {"alpha": params.alpha, "sigma_r": params.sigma_r,
                           "n_used": info.n_used}
    alphas = [v["alpha"] for v in estimates.values()]
    sigmas = [v["sigma_r"] for v in estimates.values()]
    return {"per_size": estimates,
            "alpha_spread": float(max(alphas) - min(alphas)),
            "sigma_spread": float(max(sigmas) - min(sigmas)),
            "interpretation": "parameters are physical linear-light values from "
                              "fixed-pixel patch statistics, not image fractions"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="runs/eval")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    report = {
        "grid": evaluate_grid(),
        "edge_cases": evaluate_edge_cases(),
        "size_invariance": evaluate_size_invariance(),
    }
    path = os.path.join(args.out_dir, "evaluation_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Also export one visual example with all four views.
    fx = make_observation(size=256, exposure=0.5, alpha=ALPHA,
                          sigma_r=SIGMA, seed=4)
    result = run_pipeline(fx["observed_srgb"],
                          PipelineConfig(alpha=ALPHA, sigma_r=SIGMA, strength=1.0))
    export_results(result, os.path.join(args.out_dir, "example_views"))

    print(f"wrote {path}")
    print("grid summary (line_retention / colour_bias / black_mean):")
    for r in report["grid"][:6]:
        print(f"  exp={r['exposure']:<4} st={r['strength']:<3} "
              f"line={r['line_retention']:<5} color={r['colour_max_abs_bias']:<7} "
              f"black={r['black_patch_mean']:<7} reliable={r['reliable_fraction']}")
    print("edge cases:", json.dumps(report["edge_cases"], indent=2))
    print("size invariance spreads:",
          report["size_invariance"]["alpha_spread"],
          report["size_invariance"]["sigma_spread"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
