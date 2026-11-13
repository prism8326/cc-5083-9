"""Affine Poisson-Gaussian noise model and variance stabilization.

The observation model in *linear-light* units is

        y = alpha * p + n,   p ~ Poisson(x / alpha),   n ~ N(0, sigma_r^2)

so that

        E[y] = x,   Var(y) = alpha * x + sigma_r^2.

``alpha`` is the **signal-dependent** (shot/Poisson) noise coefficient while
``sigma_r`` is the standard deviation of the **fixed bias** (read) noise.
Keeping the two terms separate is what lets the pipeline distinguish photon
granularity from sensor read noise instead of smoothing everything alike.

The Generalized Anscombe transform (GAT, Mäkitalo & Foi 2011) maps the data to
a domain where the stabilized variance is exactly 1; denoising there followed
by the *exact unbiased inverse GAT* avoids the bias of a plain algebraic
inverse in deep shadows.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EPS = 1.0e-10


@dataclass(frozen=True)
class NoiseParams:
    """Parameters of the affine Poisson-Gaussian model.

    alpha   : signal-dependent coefficient, Var = alpha*x + sigma_r^2
    sigma_r : fixed (read) noise standard deviation, in linear-light units
    source  : where the values came from ("calibration", "estimate", "default")
    """

    alpha: float
    sigma_r: float
    source: str = "default"

    def variance(self, signal: np.ndarray) -> np.ndarray:
        return self.alpha * np.maximum(signal, 0.0) + self.sigma_r ** 2

    def as_dict(self) -> dict:
        return {"alpha": float(self.alpha),
                "sigma_r": float(self.sigma_r),
                "source": self.source}


def gat(y: np.ndarray, params: NoiseParams) -> np.ndarray:
    """Forward *normalized* Generalized Anscombe transform.

    The normalization factor 2/sqrt(alpha) makes the stabilized variance
    exactly 1 regardless of the shot coefficient, so denoising strengths are
    expressed in noise-sigma units and are independent of image scale.
    """
    a = max(float(params.alpha), _EPS)
    s2 = float(params.sigma_r) ** 2
    inner = (3.0 / 8.0) * a + (np.asarray(y, dtype=np.float64) - s2 / a)
    return (2.0 / np.sqrt(a)) * np.sqrt(np.maximum(inner, 0.0))


# Coefficients of the rational polynomial approximation of the exact
# unbiased inverse GAT (Mäkitalo & Foi, IEEE TIP 2011, Table I), valid for
# 0 <= z < 2.
_P1 = 1.0 / 8.0
_P2 = np.sqrt(3.0 / 8.0) * (2.0 / np.sqrt(np.pi) - 1.0)
_P3 = (1.0 / 8.0) * np.sqrt(3.0 / np.pi)
_P4 = (-1.0 / 8.0) * np.sqrt(3.0 / np.pi)
_P5 = np.sqrt(3.0 / 8.0) * (6.0 / np.pi - 4.0 * np.sqrt(6.0 / (9.0 * np.pi)) + 0.5)
_P6 = (1.0 / 8.0) * np.sqrt(3.0 / np.pi) * (12.0 / np.sqrt(np.pi) - 22.0 / np.pi + 3.0 * np.sqrt(6.0 / np.pi) - 2.0)


def inverse_gat(z: np.ndarray, params: NoiseParams) -> np.ndarray:
    """Exact unbiased inverse of the normalized GAT, in linear-light units.

    The normalized GAT maps data to unit variance; its inverse approximation
    (rational form below 2, asymptotic form above) returns the underlying
    value in *count units* x/alpha, so the result is scaled by alpha.  The
    forward transform already subtracts the read-noise offset sigma_r^2/alpha,
    hence no offset is added back here.
    """
    zn = np.asarray(z, dtype=np.float64)
    a = max(float(params.alpha), _EPS)

    asymptotic = np.zeros_like(zn)
    hi = zn >= 2.0
    zh = zn[hi]
    asymptotic[hi] = (0.25 * zh * zh
                      + 0.25 * np.sqrt(1.5) / zh
                      - 1.125 / zh ** 2
                      + 0.125 * np.sqrt(1.5) / zh ** 3
                      - 0.125 / zh ** 4)
    # The rational approximation is numerically safest on [0, 2); clip only
    # the tiny sub-zero tail that can arise from finite-precision inputs.
    zc = np.clip(zn, 0.0, 2.0)
    rational = (zc * (zc * (zc * (zc * (zc * _P6 + _P5) + _P4) + _P3) + _P2) + _P1) \
        / (zc * (zc * (zc * _P4 + _P3) + _P2) + 1.0)
    return a * np.where(hi, asymptotic, rational)
