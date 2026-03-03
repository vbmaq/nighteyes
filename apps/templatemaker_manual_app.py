"""
Manual template creator for glint constellations.

Usage:
  python templatemaker_manual.py
  - Click "Browse" or "Load" to open an image.
  - Click on the image to add points (in order).
  - Use Delete/Reset to edit current points.
  - Use Save Template to append a new template to a JSON file.
  - Use Load Templates to open an existing templates JSON.
  - Select a saved template to Preview or Delete the whole template set.

Saved format (same schema as templates/chugh/default_templates.json):
{
  "templates": { "P1_DEFAULT": [[x,y],...], ... },
  "default_bank": ["P1_DEFAULT", ...],
  "sources": { "P1_DEFAULT": "image_name.jpg", ... }
}
"""

from __future__ import annotations

import json
import os
import re
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    from PIL import Image, ImageTk
except ImportError as exc:  # pragma: no cover - UI
    raise SystemExit("Pillow is required. Install with: python -m pip install pillow") from exc

import cv2
import numpy as np

from glint_pipeline import eval_gen as g

class TemplateMakerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Manual Template Creator")

        # Image state
        self.image_path: str | None = None
        self.image_dir: str | None = None
        self.base_name: str | None = None
        self.img = None  # PIL image
        self.photo = None  # ImageTk photo
        self.scale = 1.0
        self.zoom = 1.0

        # Points for current image
        self.points: list[tuple[float, float]] = []
        self.preview_points: list[tuple[float, float]] = []
        self.cand_raw: list[tuple[float, float]] = []
        self.cand_pool: list[tuple[float, float]] = []

        # Template file state
        self.templates_path: str | None = None
        self.image_config_path: str | None = None
        self.templates = {"templates": {}, "default_bank": [], "sources": {}}

        # UI
        self.overlay_ids: list[int] = []
        self.canvas_image_id: int | None = None
        self._build_ui()

    # UI construction -------------------------------------------------
    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=6)
        top.pack(fill=tk.X)

        self.path_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.path_var, width=70).pack(side=tk.LEFT, padx=(0, 6), fill=tk.X, expand=True)
        ttk.Button(top, text="Browse", command=self.browse_image).pack(side=tk.LEFT, padx=3)
        ttk.Button(top, text="Load", command=self.load_image_from_entry).pack(side=tk.LEFT, padx=3)

        top2 = ttk.Frame(self.root, padding=6)
        top2.pack(fill=tk.X)
        ttk.Button(top2, text="Load Templates", command=self.load_templates_file).pack(side=tk.LEFT, padx=3)
        ttk.Button(top2, text="Load Image Config", command=self.load_image_config).pack(side=tk.LEFT, padx=3)
        ttk.Button(top2, text="Save Template", command=self.save_template).pack(side=tk.LEFT, padx=3)
        ttk.Button(top2, text="Zoom +", command=self.zoom_in).pack(side=tk.LEFT, padx=3)
        ttk.Button(top2, text="Zoom -", command=self.zoom_out).pack(side=tk.LEFT, padx=3)
        ttk.Button(top2, text="Zoom 1:1", command=self.zoom_reset).pack(side=tk.LEFT, padx=3)
        self.templates_var = tk.StringVar(value="(no templates file)")
        ttk.Label(top2, textvariable=self.templates_var).pack(side=tk.LEFT, padx=8)

        main = ttk.Frame(self.root, padding=6)
        main.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(main, bg="#111", width=900, height=700, highlightthickness=1, highlightbackground="#444")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_click)

        sidebar = ttk.Frame(main, width=280, padding=(6, 0))
        sidebar.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(sidebar, text="Current points").pack(anchor=tk.W, pady=(0, 4))
        self.listbox_points = tk.Listbox(sidebar, selectmode=tk.EXTENDED, width=32, height=12)
        self.listbox_points.pack(fill=tk.BOTH, expand=False)

        btn_row = ttk.Frame(sidebar)
        btn_row.pack(fill=tk.X, pady=6)
        ttk.Button(btn_row, text="Delete Point", command=self.delete_selected_points).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=2
        )
        ttk.Button(btn_row, text="Reset Points", command=self.reset_points).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=2
        )

        ttk.Label(sidebar, text="Candidate preview").pack(anchor=tk.W, pady=(8, 2))
        self.show_cand_raw = tk.BooleanVar(value=True)
        self.show_cand_pool = tk.BooleanVar(value=False)
        ttk.Checkbutton(sidebar, text="Show raw candidates", variable=self.show_cand_raw, command=self.render_points).pack(anchor=tk.W)
        ttk.Checkbutton(sidebar, text="Show pooled candidates", variable=self.show_cand_pool, command=self.render_points).pack(anchor=tk.W)

        ttk.Label(sidebar, text="Candidate params").pack(anchor=tk.W, pady=(8, 2))
        self.param_vars = {}
        def add_param(label, key, default):
            row = ttk.Frame(sidebar)
            row.pack(fill=tk.X, pady=1)
            ttk.Label(row, text=label, width=14).pack(side=tk.LEFT)
            var = tk.StringVar(value=str(default))
            ttk.Entry(row, textvariable=var, width=10).pack(side=tk.LEFT)
            self.param_vars[key] = var

        add_param("percentile", "percentile", 99.7)
        add_param("kernel", "kernel", 11)
        add_param("enhance_mode", "enhance_mode", "tophat")
        add_param("dog_sigma1", "dog_sigma1", 1.0)
        add_param("dog_sigma2", "dog_sigma2", 2.2)
        add_param("denoise", "denoise", 1)
        add_param("denoise_k", "denoise_k", 0)
        add_param("clahe", "clahe", 1)
        add_param("clahe_clip", "clahe_clip", 2.0)
        add_param("clahe_tiles", "clahe_tiles", 8)
        add_param("gamma", "gamma", 1.0)
        add_param("unsharp", "unsharp", 0)
        add_param("unsharp_amount", "unsharp_amount", 1.0)
        add_param("unsharp_sigma", "unsharp_sigma", 1.0)
        add_param("clean_k", "clean_k", 3)
        add_param("open_iter", "open_iter", 1)
        add_param("close_iter", "close_iter", 0)
        add_param("min_area", "min_area", 8)
        add_param("max_area", "max_area", 250)
        add_param("min_circ", "min_circ", 0.45)
        add_param("max_pool", "max_pool", 30)

        ttk.Label(sidebar, text="Saved templates").pack(anchor=tk.W, pady=(8, 4))
        self.listbox_templates = tk.Listbox(sidebar, selectmode=tk.SINGLE, width=32, height=12)
        self.listbox_templates.pack(fill=tk.BOTH, expand=True)

        btn_row2 = ttk.Frame(sidebar)
        btn_row2.pack(fill=tk.X, pady=6)
        ttk.Button(btn_row2, text="Preview", command=self.preview_template).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=2
        )
        ttk.Button(btn_row2, text="Delete Template", command=self.delete_template).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=2
        )

        self.status_var = tk.StringVar(value="Load an image to begin.")
        ttk.Label(self.root, textvariable=self.status_var, padding=6).pack(fill=tk.X)

    # Image handling --------------------------------------------------
    def browse_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Images", "*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff"), ("All files", "*.*")],
        )
        if path:
            self.path_var.set(path)
            self.load_image(path)

    def load_image_from_entry(self) -> None:
        path = self.path_var.get().strip()
        if path:
            self.load_image(path)

    def load_image(self, path: str) -> None:
        if not os.path.isfile(path):
            messagebox.showerror("Error", f"File not found:\n{path}")
            return
        try:
            self.img = Image.open(path).convert("RGB")
        except Exception as exc:  # pragma: no cover - UI
            messagebox.showerror("Error", f"Could not open image:\n{exc}")
            return

        self.image_path = path
        self.image_dir = os.path.dirname(path)
        self.base_name = os.path.basename(path)
        self.points = []
        self.preview_points = []
        self.cand_raw = []
        self.cand_pool = []
        self.zoom = 1.0

        self._display_image()
        self._compute_candidates()
        self.render_points()
        self.status_var.set(f"Loaded {self.base_name}")

    def _display_image(self) -> None:
        if self.img is None:
            return
        max_w, max_h = 1100, 900
        w, h = self.img.size
        base_scale = min(max_w / w, max_h / h, 1.0)
        self.scale = base_scale * float(self.zoom)
        if self.scale != 1.0:
            disp_img = self.img.resize((int(w * self.scale), int(h * self.scale)), Image.LANCZOS)
        else:
            disp_img = self.img
        self.photo = ImageTk.PhotoImage(disp_img)
        self.canvas.config(width=disp_img.width, height=disp_img.height, scrollregion=(0, 0, disp_img.width, disp_img.height))
        self.canvas.delete("all")
        self.canvas_image_id = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.overlay_ids.clear()

    def zoom_in(self) -> None:
        if self.img is None:
            return
        self.zoom = min(8.0, self.zoom * 1.25)
        self._display_image()
        self.render_points()

    def zoom_out(self) -> None:
        if self.img is None:
            return
        self.zoom = max(0.2, self.zoom / 1.25)
        self._display_image()
        self.render_points()

    def zoom_reset(self) -> None:
        if self.img is None:
            return
        self.zoom = 1.0
        self._display_image()
        self.render_points()

    # Template file handling ------------------------------------------
    def load_templates_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Select templates JSON",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            data = json.loads(open(path, "r", encoding="utf-8-sig").read())
        except Exception as exc:
            messagebox.showerror("Error", f"Could not read templates file:\n{exc}")
            return
        if not isinstance(data, dict) or "templates" not in data or "default_bank" not in data:
            messagebox.showerror("Error", "Invalid templates file format.")
            return
        if "sources" not in data:
            data["sources"] = {}
        self.templates = data
        self.templates_path = path
        self.templates_var.set(os.path.basename(path))
        self._refresh_templates_list()

    def load_image_config(self) -> None:
        path = filedialog.askopenfilename(
            title="Select image config JSON",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        self.image_config_path = path
        try:
            cfg = json.loads(open(path, "r", encoding="utf-8-sig").read())
            if isinstance(cfg, dict):
                # update UI params
                def set_if(key, val):
                    if key in self.param_vars:
                        self.param_vars[key].set(str(val))
                if "thr_pct" in cfg:
                    thr = float(cfg["thr_pct"])
                    set_if("percentile", thr / 10.0 if thr > 100.0 else thr)
                set_if("kernel", cfg.get("tophat_k", self.param_vars["kernel"].get()))
                set_if("denoise", cfg.get("denoise", self.param_vars["denoise"].get()))
                set_if("denoise_k", cfg.get("denoise_k", self.param_vars["denoise_k"].get()))
                set_if("clahe", cfg.get("clahe", self.param_vars["clahe"].get()))
                set_if("clahe_clip", cfg.get("clahe_clip", self.param_vars["clahe_clip"].get()))
                set_if("clahe_tiles", cfg.get("clahe_tile", self.param_vars["clahe_tiles"].get()))
                set_if("enhance_mode", cfg.get("enhance_mode", self.param_vars["enhance_mode"].get()))
                set_if("dog_sigma1", cfg.get("dog_sigma1", self.param_vars["dog_sigma1"].get()))
                set_if("dog_sigma2", cfg.get("dog_sigma2", self.param_vars["dog_sigma2"].get()))
                set_if("gamma", cfg.get("gamma", self.param_vars["gamma"].get()))
                set_if("unsharp", cfg.get("unsharp", self.param_vars["unsharp"].get()))
                set_if("unsharp_amount", cfg.get("unsharp_amount", self.param_vars["unsharp_amount"].get()))
                set_if("unsharp_sigma", cfg.get("unsharp_sigma", self.param_vars["unsharp_sigma"].get()))
                set_if("clean_k", cfg.get("clean_k", self.param_vars["clean_k"].get()))
                set_if("open_iter", cfg.get("open_iter", self.param_vars["open_iter"].get()))
                set_if("close_iter", cfg.get("close_iter", self.param_vars["close_iter"].get()))
                set_if("min_area", cfg.get("min_area", self.param_vars["min_area"].get()))
                set_if("max_area", cfg.get("max_area", self.param_vars["max_area"].get()))
                set_if("min_circ", cfg.get("min_circ", self.param_vars["min_circ"].get()))
        except Exception:
            pass
        self._compute_candidates()
        self.render_points()

    def save_template(self) -> None:
        if not self.points:
            messagebox.showwarning("No points", "Add points before saving.")
            return
        if not self.templates_path:
            path = filedialog.asksaveasfilename(
                title="Save templates JSON",
                defaultextension=".json",
                filetypes=[("JSON", "*.json")],
                initialfile="default_templates.json",
            )
            if not path:
                return
            self.templates_path = path
            self.templates_var.set(os.path.basename(path))
            if not os.path.isfile(self.templates_path):
                self.templates = {"templates": {}, "default_bank": [], "sources": {}}

        # reload file from disk (prevents stale numbering if file was edited externally)
        try:
            data = json.loads(open(self.templates_path, "r", encoding="utf-8-sig").read())
            if isinstance(data, dict) and "templates" in data:
                if "default_bank" not in data:
                    data["default_bank"] = []
                if "sources" not in data:
                    data["sources"] = {}
                self.templates = data
        except Exception:
            pass

        # assign next P#_DEFAULT (scan templates + default_bank)
        existing = list(self.templates.get("templates", {}).keys()) + list(self.templates.get("default_bank", []))
        used = set()
        for name in existing:
            m = re.match(r"^P(\d+)_DEFAULT$", str(name))
            if m:
                used.add(int(m.group(1)))
        next_id = 1
        while next_id in used:
            next_id += 1
        name = f"P{next_id}_DEFAULT"

        self.templates.setdefault("templates", {})[name] = [[float(x), float(y)] for x, y in self.points]
        self.templates.setdefault("default_bank", []).append(name)
        self.templates.setdefault("sources", {})[name] = self.base_name or ""

        with open(self.templates_path, "w", encoding="utf-8") as f:
            json.dump(self.templates, f, indent=2)
        self._refresh_templates_list()
        self.status_var.set(f"Saved {name} ({len(self.points)} pts)")

    def delete_template(self) -> None:
        sel = self.listbox_templates.curselection()
        if not sel:
            return
        idx = sel[0]
        name = self.listbox_templates.get(idx)
        if not messagebox.askyesno("Delete", f"Delete template {name}?"):
            return
        self.templates["templates"].pop(name, None)
        if name in self.templates.get("default_bank", []):
            self.templates["default_bank"].remove(name)
        if "sources" in self.templates:
            self.templates["sources"].pop(name, None)
        if self.templates_path:
            with open(self.templates_path, "w", encoding="utf-8") as f:
                json.dump(self.templates, f, indent=2)
        self._refresh_templates_list()
        self.preview_points = []
        self.render_points()

    def preview_template(self) -> None:
        sel = self.listbox_templates.curselection()
        if not sel:
            return
        name = self.listbox_templates.get(sel[0])
        pts = self.templates.get("templates", {}).get(name)
        if not pts:
            return
        self.preview_points = [(float(x), float(y)) for x, y in pts]

        # attempt to load source image if available and current image differs
        src = self.templates.get("sources", {}).get(name, "")
        if src and self.image_dir:
            src_path = os.path.join(self.image_dir, src)
            if os.path.isfile(src_path) and self.base_name != src:
                self.load_image(src_path)
        self.render_points()
        self.status_var.set(f"Previewing {name}")

    # Annotation handling ----------------------------------------------
    def on_click(self, event) -> None:
        if self.img is None:
            return
        x, y = event.x / self.scale, event.y / self.scale
        self.points.append((x, y))
        self.render_points()

    def render_points(self) -> None:
        if self.canvas_image_id is None:
            return
        for item in self.overlay_ids:
            self.canvas.delete(item)
        self.overlay_ids.clear()

        # current points (green)
        self.listbox_points.delete(0, tk.END)
        for idx, (x, y) in enumerate(self.points):
            sx, sy = x * self.scale, y * self.scale
            r = 4
            oid = self.canvas.create_oval(sx - r, sy - r, sx + r, sy + r, fill="#5adb4c", outline="")
            self.overlay_ids.append(oid)
            self.listbox_points.insert(tk.END, f"{idx+1}: ({x:.1f}, {y:.1f})")

        # preview points (orange)
        for (x, y) in self.preview_points:
            sx, sy = x * self.scale, y * self.scale
            r = 4
            oid = self.canvas.create_oval(sx - r, sy - r, sx + r, sy + r, fill="#f0a34c", outline="")
            self.overlay_ids.append(oid)

        # candidate previews
        if self.show_cand_raw.get():
            for (x, y) in self.cand_raw:
                sx, sy = x * self.scale, y * self.scale
                r = 3
                oid = self.canvas.create_oval(sx - r, sy - r, sx + r, sy + r, outline="#3aa0ff", width=1)
                self.overlay_ids.append(oid)
        if self.show_cand_pool.get():
            for (x, y) in self.cand_pool:
                sx, sy = x * self.scale, y * self.scale
                r = 4
                oid = self.canvas.create_oval(sx - r, sy - r, sx + r, sy + r, outline="#ffd04c", width=2)
                self.overlay_ids.append(oid)

    def delete_selected_points(self) -> None:
        sel = list(self.listbox_points.curselection())
        if not sel:
            return
        for idx in sorted(sel, reverse=True):
            if 0 <= idx < len(self.points):
                self.points.pop(idx)
        self.render_points()

    def reset_points(self) -> None:
        self.points = []
        self.render_points()

    def _refresh_templates_list(self) -> None:
        self.listbox_templates.delete(0, tk.END)
        for name in self.templates.get("default_bank", []):
            self.listbox_templates.insert(tk.END, name)

    # Candidate preview -------------------------------------------------
    def _build_args_for_candidates(self):
        args = type("Args", (), {})()
        args.percentile = float(self.param_vars["percentile"].get())
        args.kernel = int(self.param_vars["kernel"].get())
        args.enhance_mode = str(self.param_vars["enhance_mode"].get())
        args.dog_sigma1 = float(self.param_vars["dog_sigma1"].get())
        args.dog_sigma2 = float(self.param_vars["dog_sigma2"].get())
        args.denoise = int(self.param_vars["denoise"].get())
        args.denoise_k = int(self.param_vars["denoise_k"].get())
        args.clahe = int(self.param_vars["clahe"].get())
        args.clahe_clip = float(self.param_vars["clahe_clip"].get())
        args.clahe_tiles = int(self.param_vars["clahe_tiles"].get())
        args.gamma = float(self.param_vars["gamma"].get())
        args.unsharp = int(self.param_vars["unsharp"].get())
        args.unsharp_amount = float(self.param_vars["unsharp_amount"].get())
        args.unsharp_sigma = float(self.param_vars["unsharp_sigma"].get())
        args.clean_k = int(self.param_vars["clean_k"].get())
        args.open_iter = int(self.param_vars["open_iter"].get())
        args.close_iter = int(self.param_vars["close_iter"].get())
        args.min_area = int(self.param_vars["min_area"].get())
        args.max_area = int(self.param_vars["max_area"].get())
        args.min_circ = float(self.param_vars["min_circ"].get())
        args.min_maxI = 200
        args.max_pool = int(self.param_vars["max_pool"].get())
        args.score2_mode = "contrast_support"
        args.support_M = 30
        args.support_tol = 0.10
        args.support_w = 0.15
        args.contrast_r_inner = 3
        args.contrast_r_outer1 = 5
        args.contrast_r_outer2 = 8
        args.dog_sigma1 = 1.0
        args.dog_sigma2 = 2.2
        args.template_bank_source = "default"
        args.template_bank_path = None
        args.template_build_mode = "procrustes"
        args.verbose_template = False
        return args

    def _compute_candidates(self) -> None:
        if self.img is None:
            return
        # build args and apply image_config if loaded
        args = self._build_args_for_candidates()
        if self.image_config_path:
            g._apply_image_config(args, self.image_config_path)
        # prepare expected distances for contrast_support
        try:
            bank = g.load_default_template_bank()
            d_expected = g.compute_expected_pairwise_distances(bank)
        except Exception:
            d_expected = None
        gray = cv2.cvtColor(np.array(self.img), cv2.COLOR_RGB2GRAY)
        params = g.scale_params_for_image(args, w=gray.shape[1], h=gray.shape[0])
        cand_xy, rows, cand_score2, cand_raw_count, cand_support = g.detect_candidates_one_pass(
            gray, params, args, d_expected=d_expected
        )
        self.cand_raw = [(float(x), float(y)) for x, y in cand_xy]
        # pool
        rows_sorted = sorted(
            zip(cand_xy, cand_score2),
            key=lambda p: p[1],
            reverse=True,
        )[: args.max_pool]
        self.cand_pool = [(float(p[0][0]), float(p[0][1])) for p in rows_sorted]


def main() -> None:
    root = tk.Tk()
    TemplateMakerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
