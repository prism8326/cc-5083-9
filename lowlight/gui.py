"""Graphical low-light restoration window (primary interface).

Layout
------
Left control bar: file open, denoising strength / method, optional declared
noise calibration (shot coefficient alpha and read noise sigma_r), shadow-lift
controls, run and export buttons, and an estimation/error message panel.
Result area: 2x2 grid showing the original, the restored image, the signed
residual and the reliability map; a dark-region zoom strip underneath renders
the actual computed arrays so shadow detail can be inspected directly.

All result imagery is produced by the real pipeline in :mod:`lowlight.pipeline`;
exceptions are caught and shown in the window's status panel rather than
crashing the application.
"""
from __future__ import annotations

import os
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk

import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib import font_manager

# Register any user-local CJK font (e.g. a deb extracted under ~/.local) so
# Chinese labels render instead of missing-glyph boxes. Falls back silently to
# the default font when none is present.
_CJK_CANDIDATES = [
    "wqy-microhei.ttc", "wqy-zenhei.ttc",
    "NotoSansCJK-Regular.ttc", "NotoSansCJKsc-Regular.otf",
]
_FONT_SEARCH_ROOTS = [
    os.path.expanduser("~/.local/extracted/usr/share/fonts"),
    os.path.expanduser("~/.local/share/fonts"),
    "/usr/share/fonts", "/usr/local/share/fonts",
]


def _register_cjk_font() -> str | None:
    for root in _FONT_SEARCH_ROOTS:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirs, files in os.walk(root):
            for name in files:
                if name.lower() in _CJK_CANDIDATES:
                    path = os.path.join(dirpath, name)
                    try:
                        font_manager.fontManager.addfont(path)
                    except Exception:
                        continue
                    return font_manager.FontProperties(fname=path).get_name()
    return None


_CJK_FONT = _register_cjk_font()
if _CJK_FONT:
    matplotlib.rcParams["font.family"] = [_CJK_FONT, "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

from .color import linear_luma
from .lifting import LiftParams
from .pipeline import (PipelineConfig, export_results, read_image,
                       reset_session_cache, run_pipeline)


class LowlightWindow:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("暗光恢复窗口 / Low-light Restoration")
        root.geometry("1460x900")
        self._configure_tk_font()

        self.rgb01 = None          # sRGB-range float input
        self.image_path = None
        self.result = None
        self._busy = False
        self._run_started_at = None
        self._last_elapsed = None
        self._run_generation = 0
        # When True the pipeline runs on the calling thread; used by the
        # headless tests so result delivery is deterministic.
        self.synchronous = False

        self._build_controls()
        self._build_figure()
        self._set_status("请打开一张暗光照片。", level="info")

    def _configure_tk_font(self) -> None:
        """Point Tk's default fonts at a registered CJK family when present."""
        if not _CJK_FONT:
            return
        for name in ("TkDefaultFont", "TkTextFont", "TkFixedFont",
                     "TkMenuFont", "TkHeadingFont", "TkCaptionFont"):
            try:
                font = tkfont.nametofont(name)
                font.configure(family=_CJK_FONT)
            except tk.TclError:
                continue

    # ------------------------------------------------------------------ UI
    def _build_controls(self) -> None:
        bar = ttk.Frame(self.root, padding=8)
        bar.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Button(bar, text="打开图像…", command=self.on_open).pack(fill=tk.X, pady=2)
        self.path_var = tk.StringVar(value="(未选择图像)")
        ttk.Label(bar, textvariable=self.path_var, wraplength=280,
                  foreground="#555").pack(fill=tk.X, pady=(0, 8))

        box1 = ttk.LabelFrame(bar, text="去噪参数", padding=6)
        box1.pack(fill=tk.X, pady=4)
        self.strength_var = tk.DoubleVar(value=1.0)
        ttk.Scale(box1, from_=0.0, to=3.0, variable=self.strength_var,
                  command=lambda _=None: self._sync_label(self.strength_var, self.strength_lbl, 2)).pack(fill=tk.X)
        self.strength_lbl = ttk.Label(box1, text="1.00 ×σ（单位噪声域，与图像尺寸无关）")
        self.strength_lbl.pack(anchor="w")
        self.method_var = tk.StringVar(value="nlm")
        ttk.Radiobutton(box1, text="非局部均值 NLM（默认，细线保留好）",
                        variable=self.method_var, value="nlm").pack(anchor="w")
        ttk.Radiobutton(box1, text="Haar 小波收缩（无 OpenCV 时回退）",
                        variable=self.method_var, value="wavelet").pack(anchor="w")

        box2 = ttk.LabelFrame(bar, text="噪声标定（可声明；留空则自动估计）", padding=6)
        box2.pack(fill=tk.X, pady=4)
        self.use_cal_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(box2, text="使用我声明的标定值",
                        variable=self.use_cal_var).pack(anchor="w")
        self.alpha_var = tk.StringVar(value="0.004")
        self.sigma_var = tk.StringVar(value="0.006")
        ttk.Label(box2, text="信号相关系数 α（泊松/散粒）").pack(anchor="w")
        ttk.Entry(box2, textvariable=self.alpha_var, width=12).pack(anchor="w")
        ttk.Label(box2, text="固定偏置 σ_r（读出噪声，线性单位）").pack(anchor="w")
        ttk.Entry(box2, textvariable=self.sigma_var, width=12).pack(anchor="w")
        ttk.Label(box2, text="留空不勾选时：块统计 + Theil–Sen 稳健回归，\n"
                             "物理像素固定窗口，估计范围 α≥0、σ_r≥0；\n"
                             "无法测量时回退 σ_r≈1/510 并在下方提示。",
                  foreground="#666", wraplength=280, justify=tk.LEFT).pack(anchor="w", pady=(4, 0))

        box3 = ttk.LabelFrame(bar, text="受约束暗部提升", padding=6)
        box3.pack(fill=tk.X, pady=4)
        self.shadow_var = tk.DoubleVar(value=0.12)
        ttk.Scale(box3, from_=0.02, to=0.4, variable=self.shadow_var,
                  command=lambda _=None: self._sync_label(self.shadow_var, self.shadow_lbl, 3)).pack(fill=tk.X)
        self.shadow_lbl = ttk.Label(box3, text="阴影目标线性亮度 0.120")
        self.shadow_lbl.pack(anchor="w")
        self.gain_var = tk.DoubleVar(value=8.0)
        ttk.Scale(box3, from_=1.0, to=16.0, variable=self.gain_var,
                  command=lambda _=None: self._sync_label(self.gain_var, self.gain_lbl, 1)).pack(fill=tk.X)
        self.gain_lbl = ttk.Label(box3, text="增益硬上限 8.0×")
        self.gain_lbl.pack(anchor="w")
        ttk.Label(box3, text="增益随亮度平滑衰减到 1，高光与白平衡不动；\n"
                             "按通道比例施加以保留暗部细微色差。",
                  foreground="#666", wraplength=280, justify=tk.LEFT).pack(anchor="w", pady=(4, 0))

        ttk.Button(bar, text="运行恢复", command=self.on_run).pack(fill=tk.X, pady=(10, 2))
        ttk.Button(bar, text="查看暗部细节对比…",
                   command=self.on_show_shadow).pack(fill=tk.X, pady=2)
        ttk.Button(bar, text="导出原图/恢复图/残差/可靠性图与参数 JSON…",
                   command=self.on_export).pack(fill=tk.X, pady=2)

        run_box = ttk.LabelFrame(bar, text="运行状态", padding=6)
        run_box.pack(fill=tk.X, pady=4)
        self.run_state_var = tk.StringVar(value="空闲")
        self.run_state_lbl = ttk.Label(run_box, textvariable=self.run_state_var,
                                       foreground="#333")
        self.run_state_lbl.pack(anchor="w")
        self.run_elapsed_var = tk.StringVar(value="尚未运行")
        ttk.Label(run_box, textvariable=self.run_elapsed_var,
                  foreground="#666").pack(anchor="w")

        ttk.Separator(bar, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)
        self.status = tk.Text(bar, width=38, height=14, wrap=tk.WORD,
                              background="#f7f7f7", relief=tk.GROOVE)
        self.status.pack(fill=tk.BOTH, expand=True)
        self.status.tag_configure("info", foreground="#222")
        self.status.tag_configure("warn", foreground="#a06000")
        self.status.tag_configure("error", foreground="#b00020")

    def _build_figure(self) -> None:
        pane = ttk.Frame(self.root)
        pane.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.fig = Figure(figsize=(9, 9), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=pane)
        toolbar = NavigationToolbar2Tk(self.canvas, pane, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.axes = []
        # Bilingual in-axes titles; the registered CJK font renders Chinese.
        for i, title in enumerate(["原图 Original input",
                                   "恢复图像 Restored image",
                                   "残差 Residual (mid-grey = 0)",
                                   "可靠性 Reliability (white = reliable)"]):
            ax = self.fig.add_subplot(2, 2, i + 1)
            ax.set_title(title, fontsize=10)
            ax.axis("off")
            self.axes.append(ax)
        self.fig.tight_layout()

    @staticmethod
    def _sync_label(var: tk.Variable, label: ttk.Label, digits: int) -> None:
        label.configure(text=f"{float(var.get()):.{digits}f}")

    def _set_status(self, text: str, level: str = "info") -> None:
        self.status.configure(state=tk.NORMAL)
        self.status.delete("1.0", tk.END)
        self.status.insert(tk.END, text + "\n", level)
        self.status.configure(state=tk.DISABLED)

    def _append_status(self, text: str, level: str = "info") -> None:
        self.status.configure(state=tk.NORMAL)
        self.status.insert(tk.END, text + "\n", level)
        self.status.configure(state=tk.DISABLED)

    def _write_status_lines(self, lines: list[tuple[str, str]]) -> None:
        """Write ``(text, tag)`` lines so anomaly rows can be red individually."""
        self.status.configure(state=tk.NORMAL)
        self.status.delete("1.0", tk.END)
        for text, tag in lines:
            self.status.insert(tk.END, text + "\n", tag)
        self.status.configure(state=tk.DISABLED)

    def _set_run_state(self, state: str, detail: str = "") -> None:
        """Drive the visible run-state indicator (idle / running / done / error)."""
        icons = {"idle": "● 空闲", "running": "◐ 运算中…", "done": "√ 完成",
                 "error": "✗ 异常"}
        self.run_state_var.set(icons.get(state, state) + (f"  {detail}" if detail else ""))
        colors = {"idle": "#333", "running": "#0050a0", "done": "#1b6b1b",
                  "error": "#b00020"}
        self.run_state_lbl.configure(foreground=colors.get(state, "#333"))

    # ------------------------------------------------------------- actions
    def on_open(self) -> None:
        path = filedialog.askopenfilename(
            title="选择暗光图像",
            filetypes=[("图像文件", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"),
                       ("所有文件", "*.*")])
        if not path:
            return
        try:
            self.rgb01, meta = read_image(path)
        except Exception as exc:  # surfaced in-window, never a hard crash
            self._set_status(f"无法读取图像：\n{exc}", level="error")
            return
        self.image_path = path
        self.path_var.set(os.path.basename(path))
        self.result = None
        # A different image invalidates every cached stage; drop the session
        # cache so stale arrays from the previous photo can never be reused.
        reset_session_cache()
        self._draw_original()
        self._set_status(
            f"已加载 {meta['shape'][1]}×{meta['shape'][0]} 图像。\n"
            "可直接运行（自动估计噪声），或在左侧声明标定值。", level="info")

    def _parse_calibration(self) -> tuple[float | None, float | None, str | None]:
        if not self.use_cal_var.get():
            return None, None, None
        try:
            alpha = float(self.alpha_var.get())
            sigma = float(self.sigma_var.get())
        except ValueError:
            return None, None, "标定值必须是数字。"
        if alpha < 0 or sigma < 0:
            return None, None, "α 与 σ_r 必须非负。"
        return alpha, sigma, None

    def on_run(self) -> None:
        if self._busy:
            return
        if self.rgb01 is None:
            self._set_status("请先打开图像。", level="warn")
            return
        alpha, sigma, err = self._parse_calibration()
        if err is not None:
            self._set_status(err, level="error")
            return
        self._busy = True
        self._run_generation += 1
        self._run_started_at = time.perf_counter()
        self._set_run_state("running", f"第 {self._run_generation} 次")
        self.run_elapsed_var.set("已运行 0.0 s")
        self._set_status("正在运算（GAT 稳定化 → 边缘保持去噪 → 逆变换 → 约束提升）…",
                         level="info")
        cfg = PipelineConfig(
            strength=float(self.strength_var.get()),
            method=self.method_var.get(),
            alpha=alpha, sigma_r=sigma,
            lift=LiftParams(shadow_level=float(self.shadow_var.get()),
                            max_gain=float(self.gain_var.get())))
        if self.synchronous:
            self._run_worker(cfg, use_after=False)
        else:
            threading.Thread(target=self._run_worker, args=(cfg,),
                             daemon=True).start()

    def _run_worker(self, cfg: PipelineConfig, use_after: bool = True) -> None:
        try:
            result = run_pipeline(self.rgb01, cfg)
        except Exception as exc:  # all runtime failures reported in the UI
            if use_after:
                self.root.after(0, lambda: self._on_run_failed(exc))
            else:
                self._on_run_failed(exc)
            return
        if use_after:
            self.root.after(0, lambda: self._on_run_done(result))
        else:
            self._on_run_done(result)

    def _on_run_failed(self, exc: Exception) -> None:
        self._busy = False
        elapsed = (time.perf_counter() - self._run_started_at
                   if self._run_started_at else 0.0)
        self._set_run_state("error", f"{type(exc).__name__}")
        self.run_elapsed_var.set(f"失败于 {elapsed:.2f} s")
        self._set_status(f"恢复过程出现异常：\n{type(exc).__name__}: {exc}",
                         level="error")

    def _on_run_done(self, result) -> None:
        self._busy = False
        self.result = result
        elapsed = (time.perf_counter() - self._run_started_at
                   if self._run_started_at else 0.0)
        self._last_elapsed = elapsed
        self._draw_results()
        s = result.stats
        p = result.params
        timing = s.get("timing_s", {})
        self._set_run_state("done", f"{elapsed:.2f} s")
        self.run_elapsed_var.set(
            f"运行 {elapsed:.2f} s  |  预处理 {timing.get('patch_stats_s', 0.0):.2f} s"
            f"  |  GAT {timing.get('gat_s', 0.0):.2f} s"
            f"  |  去噪 {timing.get('denoise_s', 0.0):.2f} s")
        evalue = s.get("estimation") or {}
        rows = [(f"噪声来源：{ {'calibration':'用户声明','estimate':'自动估计','default':'默认回退'}[p.source] }", "info"),
                (f"α={p.alpha:.6g}  σ_r={p.sigma_r:.6g}（线性单位）", "info"),
                (f"可靠像素比例：{s['reliability']['reliable_fraction']:.2%}", "info"),
                (f"饱和：{s['reliability']['saturated']:.2%}  "
                 f"黑底不可恢复：{s['reliability']['black_floor']:.2%}", "info"),
                (f"阴影增益：{s['lift']['applied_shadow_gain']:.2f}× "
                 f"（触顶={s['lift']['gain_capped']}）", "info"),
                (f"暗部均值 {s['luma_mean_observed']:.4f} → {s['luma_mean_restored']:.4f}", "info"),
                (f"残差 RMS（线性）：{s['reliability']['residual_rms_linear']:.5f}", "info")]
        if evalue:
            cache_word = "命中缓存" if evalue.get("cache_hit") else "重新计算"
            approx = "（抽样近似）" if evalue.get("approximate") else ""
            rows.append((f"块统计：{evalue.get('n_blocks', 0)} 块，"
                         f"回归样本 {evalue.get('n_regression', 0)}{approx}，{cache_word}", "info"))
        if s.get("tiling", {}).get("enabled"):
            rows.append((f"分块去噪：halo={s['tiling']['nlm_halo']}px"
                         "（= 搜索窗//2 + 模板窗//2）", "info"))
        anom = s.get("restored_anomalies") or {}
        if anom:
            rows.append(
                (f"恢复异常：彩色斑点 {anom.get('speckle_fraction', 0.0):.2%}，"
                 f"触顶/底 {anom.get('clip_fraction', 0.0):.2%}，"
                 f"暗部残差 RMS {anom.get('dark_residual_rms', 0.0):.3f}",
                 "error" if anom.get("warnings") else "info"))
        for w in s["warnings"]:
            rows.append((f"提示：{w}", "warn"))
        self._write_status_lines(rows)
        # An output anomaly means the restored image is not trustworthy even if
        # the input flags look clean; mark the run state and open the dark
        # comparison so the damaged detail is in front of the user.
        has_anomaly = bool(anom.get("warnings"))
        if has_anomaly:
            self._set_run_state("error", "恢复图像异常")
            self._draw_shadow_strip()

    def on_export(self) -> None:
        if self.result is None:
            self._set_status("没有可导出的结果，请先运行恢复。", level="warn")
            return
        out_dir = filedialog.askdirectory(title="选择导出目录")
        if not out_dir:
            return
        try:
            paths = export_results(self.result, out_dir)
        except Exception as exc:
            self._set_status(f"导出失败：\n{exc}", level="error")
            return
        self._append_status("已导出：" + ", ".join(sorted(paths.keys())), level="info")
        messagebox.showinfo("导出完成", f"文件与 restoration_record.json 已写入：\n{out_dir}")

    # ------------------------------------------------------------- drawing
    def _draw_original(self) -> None:
        for ax in self.axes:
            ax.clear()
            ax.axis("off")
        self.axes[0].imshow(self.rgb01)
        self.axes[0].set_title("原图 Original input", fontsize=10)
        self.canvas.draw_idle()

    def _draw_results(self) -> None:
        r = self.result
        images = [r.display_original(), r.display_restored(),
                  r.display_residual(), r.display_reliability()]
        titles = ["原图 Original input", "恢复图像 Restored image",
                  "残差 Residual (mid-grey = 0)",
                  "可靠性 Reliability (white = reliable)"]
        for ax, img, title in zip(self.axes, images, titles):
            ax.clear()
            ax.imshow(img)
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def on_show_shadow(self) -> None:
        if self.result is None:
            self._set_status("请先运行恢复，再查看暗部细节。", level="warn")
            return
        self._draw_shadow_strip()

    def _draw_shadow_strip(self) -> None:
        """Dark-detail comparison window computed from the real arrays."""
        r = self.result
        luma = linear_luma(r.rgb_linear_obs)
        dark = luma < max(float(np.quantile(luma, 0.25)), 1.0e-6)
        if dark.sum() < 100:
            self._append_status("暗部像素过少，跳过暗部放大条。", level="warn")
            return
        ys, xs = np.where(dark)
        y0, y1 = max(ys.min() - 4, 0), min(ys.max() + 5, luma.shape[0])
        x0, x1 = max(xs.min() - 4, 0), min(xs.max() + 5, luma.shape[1])
        # Limit the strip to a manageable window around the darkest centroid.
        cy, cx = int(np.median(ys)), int(np.median(xs))
        half = 48
        y0, y1 = max(cy - half, 0), min(cy + half, luma.shape[0])
        x0, x1 = max(cx - half, 0), min(cx + half, luma.shape[1])
        win = tk.Toplevel(self.root)
        win.title("暗部细节（实际运算结果） / Shadow detail")
        win.transient(self.root)
        # Beside the main window when there is room, otherwise centred on the
        # screen, so the comparison never hangs off the right edge.
        win_w, win_h = 834, 420
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        right = self.root.winfo_x() + self.root.winfo_width() + 20
        if right + win_w <= screen_w:
            x = right
        else:
            x = max((screen_w - win_w) // 2, 0)
        y = max(self.root.winfo_y() + 40, 0)
        y = min(y, max(screen_h - win_h - 10, 0))
        win.geometry(f"{win_w}x{win_h}+{x}+{y}")
        fig = Figure(figsize=(8, 4), dpi=100)
        ax1, ax2 = fig.subplots(1, 2)
        ax1.imshow(r.display_original()[y0:y1, x0:x1])
        ax1.set_title("原图暗部 Original dark window")
        ax1.axis("off")
        ax2.imshow(r.display_restored()[y0:y1, x0:x1])
        ax2.set_title("恢复暗部 Restored dark window")
        ax2.axis("off")
        fig.tight_layout()
        FigureCanvasTkAgg(fig, master=win).get_tk_widget().pack(fill=tk.BOTH, expand=True)


def main() -> None:
    root = tk.Tk()
    LowlightWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
