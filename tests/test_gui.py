"""Headless smoke tests for the Tk restoration window.

These require a running X display; under CI/container use Xvfb (the project
README documents the invocation).  They exercise the real pipeline through the
UI callbacks and assert that errors are surfaced in-window rather than raised.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pytest

tk = pytest.importorskip("tkinter")

from lowlight.gui import LowlightWindow
from lowlight.pipeline import export_results, read_image, write_image
from lowlight.color import linear_to_srgb
from lowlight.synthetic import make_observation


@pytest.fixture(scope="module")
def image_path(tmp_path_factory):
    fx = make_observation(size=192, exposure=0.4, alpha=0.004,
                          sigma_r=0.006, seed=13)
    p = tmp_path_factory.mktemp("gui") / "dark.png"
    write_image(str(p), (np.clip(linear_to_srgb(fx["observed_linear"]), 0, 1)
                         * 255 + 0.5).astype(np.uint8))
    return str(p)


@pytest.fixture()
def window():
    root = tk.Tk()
    win = LowlightWindow(root)
    win.synchronous = True
    yield win
    root.destroy()


def _pump(root, seconds=0.4):
    end = time.time() + seconds
    while time.time() < end:
        root.update_idletasks()
        root.update()
        time.sleep(0.03)


def test_window_constructs(window):
    assert window.result is None
    assert window.rgb01 is None


def test_open_and_run_with_estimated_noise(window, image_path):
    window.rgb01, _ = read_image(image_path)
    window._draw_original()
    _pump(window.root)
    window.on_run()
    _pump(window.root)
    assert window.result is not None
    p = window.result.params
    assert p.source == "estimate"
    assert p.alpha >= 0 and p.sigma_r >= 0
    assert all(len(ax.images) for ax in window.axes)


def test_run_with_declared_calibration(window, image_path):
    window.rgb01, _ = read_image(image_path)
    window.use_cal_var.set(True)
    window.alpha_var.set("0.004")
    window.sigma_var.set("0.006")
    window.strength_var.set(1.2)
    window.on_run()
    _pump(window.root)
    assert window.result is not None
    assert window.result.params.source == "calibration"
    assert window.result.params.alpha == pytest.approx(0.004)


def test_invalid_calibration_reported_in_window(window, image_path):
    window.rgb01, _ = read_image(image_path)
    window.use_cal_var.set(True)
    window.alpha_var.set("not-a-number")
    window.on_run()
    _pump(window.root)
    text = window.status.get("1.0", tk.END)
    assert "数字" in text
    assert window.result is None


def test_missing_image_asks_user(window):
    window.on_run()
    text = window.status.get("1.0", tk.END)
    assert "先打开图像" in text


def test_export_writes_views_and_record(window, image_path, tmp_path):
    window.rgb01, _ = read_image(image_path)
    window.on_run()
    _pump(window.root)
    out = tmp_path / "exports"
    out.mkdir()
    paths = export_results(window.result, str(out))
    for key in ("original", "restored", "residual", "reliability", "record"):
        assert os.path.exists(paths[key])
