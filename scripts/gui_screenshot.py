"""Headless GUI screenshot driver (manual verification helper)."""
import os
import sys
import time

import tkinter as tk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lowlight.gui import LowlightWindow
from lowlight.pipeline import read_image

root = tk.Tk()
win = LowlightWindow(root)
win.synchronous = True
img_path = sys.argv[1] if len(sys.argv) > 1 else "examples/dark_sample.png"
win.rgb01, _ = read_image(img_path)
win.path_var.set(os.path.basename(img_path))
win._draw_original()
root.update_idletasks(); root.update()
win.on_run()
root.update_idletasks(); root.update()
time.sleep(0.4)
root.update_idletasks(); root.update()

# Capture via an offscreen PostScript-free route: use the canvas screenshot.
from PIL import ImageGrab
x = root.winfo_rootx(); y = root.winfo_rooty()
w = root.winfo_width(); h = root.winfo_height()
shot = ImageGrab.grab(bbox=(x, y, x + w, y + h))
out = sys.argv[2] if len(sys.argv) > 2 else "runs/gui_screenshot.png"
shot.save(out)
print("saved", out, shot.size)
root.destroy()
