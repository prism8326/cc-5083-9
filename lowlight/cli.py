"""Command-line helper for low-light restoration.

The graphical window (``python -m lowlight.gui`` or ``lowlight-gui``) is the
primary interface; this CLI exists for scripting and batch verification.

Examples
--------
# Estimate noise from the image itself, restore with default strength:
python -m lowlight.cli restore dark.jpg --out-dir runs/01

# Declare a noise calibration (alpha = shot coefficient, sigma_r = read std,
# both in normalized linear-light units):
python -m lowlight.cli restore dark.jpg --alpha 0.004 --sigma-r 0.006 \\
    --strength 1.2 --out-dir runs/02
"""
from __future__ import annotations

import argparse
import json
import sys

from .lifting import LiftParams
from .pipeline import PipelineConfig, export_results, read_image, run_pipeline


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lowlight",
                                description="Low-light static photo restoration (CLI helper)")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("restore", help="restore a single image")
    r.add_argument("image", help="path to the input photo")
    r.add_argument("--out-dir", default="lowlight_out", help="output directory")
    r.add_argument("--alpha", type=float, default=None,
                   help="signal-dependent (shot) noise coefficient, linear units")
    r.add_argument("--sigma-r", type=float, default=None,
                   help="fixed read-noise standard deviation, linear units")
    r.add_argument("--strength", type=float, default=1.0,
                   help="denoising strength in stabilized-noise sigma units (default 1.0)")
    r.add_argument("--method", choices=["nlm", "wavelet"], default="nlm")
    r.add_argument("--shadow-level", type=float, default=0.12,
                   help="target linear level for the shadow reference (default 0.12)")
    r.add_argument("--max-gain", type=float, default=8.0,
                   help="hard cap on shadow gain (default 8)")
    r.add_argument("--input-linear", action="store_true",
                   help="input is already linear-light rather than sRGB")
    r.add_argument("--residual-gain", type=float, default=None,
                   help="residual display gain (default: adaptive from sigma)")

    e = sub.add_parser("estimate", help="only estimate and print noise parameters")
    e.add_argument("image")
    e.add_argument("--input-linear", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    rgb01, meta = read_image(args.image)

    if args.command == "estimate":
        from .color import srgb_to_linear
        from .calibration import estimate_noise
        linear = rgb01 if args.input_linear else srgb_to_linear(rgb01)
        params, info = estimate_noise(linear)
        payload = {"image": meta, "noise_params": params.as_dict(),
                   "estimation": info.as_dict(),
                   "valid_range_note": ("alpha in [0,1), sigma_r in normalized "
                                        "linear-light units of an 8-bit-equivalent input")}
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    lift = LiftParams(shadow_level=args.shadow_level, max_gain=args.max_gain)
    cfg = PipelineConfig(strength=args.strength, method=args.method,
                         alpha=args.alpha, sigma_r=args.sigma_r,
                         input_is_linear=args.input_linear, lift=lift)
    result = run_pipeline(rgb01, cfg)
    paths = export_results(result, args.out_dir, residual_gain=args.residual_gain)

    print(f"noise source : {result.params.source} "
          f"(alpha={result.params.alpha:.6g}, sigma_r={result.params.sigma_r:.6g})")
    if result.stats["warnings"]:
        print("warnings     : " + "; ".join(result.stats["warnings"]))
    print(f"reliable px  : {result.stats['reliability']['reliable_fraction']:.3%}")
    print(f"shadow gain  : {result.stats['lift']['applied_shadow_gain']:.2f} "
          f"(capped={result.stats['lift']['gain_capped']})")
    for name, path in paths.items():
        print(f"{name:12s}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
