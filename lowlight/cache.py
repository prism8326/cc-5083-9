"""Content-addressed caches for the expensive, reusable computation stages.

The restoration pipeline repeats several expensive computations when the user
changes one control:

* **patch statistics** depend on the image content, the physical block
  geometry and the saturation thresholds - *not* on denoise strength or the
  shadow lift;
* **the noise estimate** additionally depends on the flat-patch rejection
  rule and the Theil-Sen sample budget;
* **the GAT-domain image** ``z = GAT(linear)`` depends on the noise parameters
  only, so changing ``strength`` while keeping a declared calibration reuses
  it;
* **the denoised GAT image** depends on ``z``, the filter method, strength,
  template and search sizes.

Each cache key is built from the *complete* set of inputs that can change the
result; changing any of them (including calibration parameters, filter choice
or window sizes) yields a different key, so stale entries are never served.
Entries are bounded LRU with an explicit byte budget, and every hit/miss is
counted so the timing report can separate cold cache setup from warm reuse.

The cache never mutates returned arrays: callers treat results as read-only.
"""
from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np


def _hashable_payload(*parts) -> str:
    h = hashlib.blake2b(digest_size=20)
    for p in parts:
        h.update(repr(p).encode())
        h.update(b"\x00")
    return h.hexdigest()


class LRUCache:
    """Byte-bounded LRU cache counting hits, misses and evictions."""

    def __init__(self, max_bytes: int):
        self.max_bytes = int(max_bytes)
        self._data: OrderedDict[str, tuple[object, int]] = OrderedDict()
        self._bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self._lock = threading.Lock()

    def get(self, key: str):
        with self._lock:
            if key in self._data:
                value, nbytes = self._data.pop(key)
                self._data[key] = (value, nbytes)
                self.hits += 1
                return value, True
            self.misses += 1
            return None, False

    def put(self, key: str, value, nbytes: int) -> None:
        with self._lock:
            if key in self._data:
                _, old = self._data.pop(key)
                self._bytes -= old
            self._data[key] = (value, int(nbytes))
            self._bytes += int(nbytes)
            while self._bytes > self.max_bytes and self._data:
                _, (_, old) = self._data.popitem(last=False)
                self._bytes -= old
                self.evictions += 1

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._bytes = 0

    @property
    def nbytes(self) -> int:
        return self._bytes

    def __len__(self) -> int:
        return len(self._data)

    def stats(self) -> dict:
        return {"entries": len(self._data), "bytes": self._bytes,
                "hits": self.hits, "misses": self.misses,
                "evictions": self.evictions,
                "max_bytes": self.max_bytes}


@dataclass
class CacheConfig:
    """Cache sizing and switch.  ``enabled=False`` forces the original,
    uncached computation path for A/B measurement."""
    enabled: bool = True
    stats_bytes: int = 96 * 1024 * 1024
    estimate_bytes: int = 8 * 1024 * 1024
    gat_bytes: int = 256 * 1024 * 1024
    denoise_bytes: int = 256 * 1024 * 1024


class PipelineCache:
    """Holds the four stage caches for one pipeline session."""

    def __init__(self, config: CacheConfig | None = None):
        self.config = config or CacheConfig()
        if self.config.enabled:
            self.patch = LRUCache(self.config.stats_bytes)
            self.estimate = LRUCache(self.config.estimate_bytes)
            self.gat = LRUCache(self.config.gat_bytes)
            self.denoise = LRUCache(self.config.denoise_bytes)
        else:
            self.patch = self.estimate = self.gat = self.denoise = None

    @property
    def enabled(self) -> bool:
        return self.patch is not None

    def stage(self, name: str) -> LRUCache:
        if not self.enabled:
            raise RuntimeError("cache is disabled")
        try:
            return {"patch": self.patch, "estimate": self.estimate,
                    "gat": self.gat, "denoise": self.denoise}[name]
        except KeyError:
            raise KeyError(f"unknown cache stage {name!r}") from None

    @staticmethod
    def patch_key(digest: str, block: int, stride: int, sat_hi: float,
                  sat_lo: float, max_blocks: int) -> str:
        return _hashable_payload("patch", digest, block, stride, sat_hi,
                                 sat_lo, max_blocks)

    @staticmethod
    def estimate_key(patch_digest: str, block: int, stride: int,
                     sat_hi: float, sat_lo: float, max_blocks: int,
                     min_samples: int, theil_budget: int) -> str:
        return _hashable_payload("estimate", patch_digest, block, stride,
                                 sat_hi, sat_lo, max_blocks, min_samples,
                                 theil_budget)

    @staticmethod
    def gat_key(digest: str, alpha: float, sigma_r: float,
                input_is_linear: bool) -> str:
        return _hashable_payload("gat", digest, alpha, sigma_r, input_is_linear)

    @staticmethod
    def denoise_key(gat_digest: str, method: str, strength: float,
                    template: int | None, search: int | None) -> str:
        return _hashable_payload("denoise", gat_digest, method, strength,
                                 template, search)

    def summary(self) -> dict:
        if not self.enabled:
            return {"enabled": False}
        return {"enabled": True,
                "patch": self.patch.stats(),
                "estimate": self.estimate.stats(),
                "gat": self.gat.stats(),
                "denoise": self.denoise.stats()}

    def clear(self) -> None:
        if self.enabled:
            for c in (self.patch, self.estimate, self.gat, self.denoise):
                c.clear()
