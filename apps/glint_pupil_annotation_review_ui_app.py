import json
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

try:
    from PIL import Image, ImageTk
except ImportError as exc:
    raise SystemExit("Pillow is required. Install with: python -m pip install pillow") from exc


def _is_valid_xy(pt: np.ndarray) -> bool:
    if pt is None:
        return False
    x, y = float(pt[0]), float(pt[1])
    if not np.isfinite(x) or not np.isfinite(y):
        return False
    return x >= 0 and y >= 0


class AnnotationReviewApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Glint + Pupil Annotation Review")

        self.folder: Optional[Path] = None
        self.files: list[Path] = []
        self.idx = 0
        self.playing = False
        self.after_id = None
        self.photo = None
        self.zoom = 1.0

        self.glints_path: Optional[Path] = None
        self.pupil_path: Optional[Path] = None
        self.glints_by_name: Dict[str, np.ndarray] = {}
        self.template_by_name: Dict[str, np.ndarray] = {}
        self.pupil_by_name: Dict[str, dict] = {}
        self.n_glints = 4

        self.selected_kind: Optional[str] = None
        self.selected_idx: Optional[int] = None
        self.drag_start: Optional[Tuple[float, float]] = None
        self.drag_mode: Optional[str] = None
        self.drag_base_glints: Optional[np.ndarray] = None
        self.drag_centroid: Optional[np.ndarray] = None

        self._build_ui()

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=6)
        top.pack(fill=tk.X)
        ttk.Button(top, text="Load Folder", command=self.load_folder).pack(side=tk.LEFT, padx=3)
        ttk.Button(top, text="Load Glints NPZ", command=self.load_glints_npz).pack(side=tk.LEFT, padx=3)
        ttk.Button(top, text="Load Pupil NPZ", command=self.load_pupil_npz).pack(side=tk.LEFT, padx=3)
        ttk.Button(top, text="Save Glints", command=self.save_glints_npz).pack(side=tk.LEFT, padx=3)
        ttk.Button(top, text="Save Pupils", command=self.save_pupil_npz).pack(side=tk.LEFT, padx=3)
        ttk.Button(top, text="Prev", command=self.prev_image).pack(side=tk.LEFT, padx=3)
        ttk.Button(top, text="Next", command=self.next_image).pack(side=tk.LEFT, padx=3)
        ttk.Button(top, text="Play", command=self.toggle_play).pack(side=tk.LEFT, padx=3)
        ttk.Button(top, text="Zoom +", command=self.zoom_in).pack(side=tk.LEFT, padx=3)
        ttk.Button(top, text="Zoom -", command=self.zoom_out).pack(side=tk.LEFT, padx=3)
        ttk.Button(top, text="Zoom 1:1", command=self.zoom_reset).pack(side=tk.LEFT, padx=3)

        self.edit_target = tk.StringVar(value="detected")
        ttk.Label(top, text="Edit").pack(side=tk.LEFT, padx=(8, 2))
        ttk.Combobox(top, textvariable=self.edit_target, values=["detected", "template", "pupil"], state="readonly", width=9).pack(side=tk.LEFT)
        ttk.Button(top, text="Delete", command=self.delete_selected).pack(side=tk.LEFT, padx=6)

        self.edit_constellation_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="Edit Constellation", variable=self.edit_constellation_var).pack(side=tk.LEFT, padx=6)
        self.rotate_constellation_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="Rotate", variable=self.rotate_constellation_var).pack(side=tk.LEFT, padx=2)
        self.scale_constellation_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="Scale", variable=self.scale_constellation_var).pack(side=tk.LEFT, padx=2)

        self.show_template_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Show Template", variable=self.show_template_var, command=self.render_current).pack(side=tk.LEFT, padx=6)

        self.status_var = tk.StringVar(value="No folder loaded.")
        ttk.Label(top, textvariable=self.status_var).pack(side=tk.LEFT, padx=8)

        main = ttk.Frame(self.root, padding=6)
        main.pack(fill=tk.BOTH, expand=True)
        self.slider = ttk.Scale(main, from_=0, to=0, orient=tk.HORIZONTAL, command=self.on_slider)
        self.slider.pack(fill=tk.X)
        self.canvas = tk.Canvas(main, bg="#111", width=900, height=700, highlightthickness=1, highlightbackground="#444")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_canvas_press)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<ButtonPress-3>", self.on_canvas_right_click)
        self.root.bind("+", self.on_scale_up)
        self.root.bind("-", self.on_scale_down)
        self.root.bind("=", self.on_scale_up)

    def load_folder(self) -> None:
        path = filedialog.askdirectory(title="Select image folder")
        if not path:
            return
        self.folder = Path(path)
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
        self.files = sorted([p for p in self.folder.iterdir() if p.suffix.lower() in exts])
        if not self.files:
            messagebox.showerror("Error", "No images found in folder")
            return
        self.idx = 0
        self.slider.configure(from_=0, to=max(0, len(self.files) - 1))
        self.render_current()

    def load_glints_npz(self) -> None:
        path = filedialog.askopenfilename(title="Select glints NPZ", filetypes=[("NPZ", "*.npz"), ("All files", "*.*")])
        if not path:
            return
        self.glints_path = Path(path)
        data = np.load(self.glints_path, allow_pickle=True)
        gmap = data["glints"].item()
        tmap = None
        if "template_xy" in data:
            try:
                tmap = data["template_xy"].item()
            except Exception:
                tmap = None
        self.glints_by_name = {}
        self.template_by_name = {}
        n_glints = 0
        for k, v in gmap.items():
            name = Path(str(k)).name
            arr = np.asarray(v, dtype=float).reshape(-1, 2)
            self.glints_by_name[name] = arr
            n_glints = max(n_glints, arr.shape[0])
            if tmap is not None and k in tmap:
                t_arr = np.asarray(tmap[k], dtype=float).reshape(-1, 2)
                self.template_by_name[name] = t_arr
        self.n_glints = max(1, n_glints)
        self.render_current()

    def load_pupil_npz(self) -> None:
        path = filedialog.askopenfilename(title="Select pupil NPZ", filetypes=[("NPZ", "*.npz"), ("All files", "*.*")])
        if not path:
            return
        self.pupil_path = Path(path)
        data = np.load(self.pupil_path, allow_pickle=True)
        filenames = data.get("filenames", None)
        ellipses = data.get("ellipses", None)
        if filenames is None or ellipses is None:
            messagebox.showerror("Error", "Pupil NPZ must contain 'filenames' and 'ellipses'")
            return
        self.pupil_by_name = {}
        for name, ell in zip(filenames, ellipses):
            fname = Path(str(name)).name
            if isinstance(ell, dict):
                self.pupil_by_name[fname] = dict(ell)
        self.render_current()

    def _get_glints_for_image(self, name: str) -> np.ndarray:
        arr = self.glints_by_name.get(name)
        if arr is None:
            return np.full((self.n_glints, 2), np.nan, dtype=float)
        return arr

    def _get_template_for_image(self, name: str) -> Optional[np.ndarray]:
        arr = self.template_by_name.get(name)
        if arr is None:
            return None
        return arr

    def _get_edit_target_kind(self) -> str:
        return self.edit_target.get()

    def _get_edit_array(self, name: str) -> Optional[np.ndarray]:
        kind = self._get_edit_target_kind()
        if kind == "template":
            arr = self.template_by_name.get(name)
            if arr is None:
                arr = np.full((self.n_glints, 2), np.nan, dtype=float)
                self.template_by_name[name] = arr
            return arr
        if kind == "detected":
            arr = self.glints_by_name.get(name)
            if arr is None:
                arr = np.full((self.n_glints, 2), np.nan, dtype=float)
                self.glints_by_name[name] = arr
            return arr
        return None

    def _set_edit_array(self, name: str, arr: np.ndarray) -> None:
        kind = self._get_edit_target_kind()
        if kind == "template":
            self.template_by_name[name] = arr
        elif kind == "detected":
            self.glints_by_name[name] = arr

    def _get_pupil_for_image(self, name: str) -> Optional[Tuple[float, float]]:
        entry = self.pupil_by_name.get(name)
        if not entry:
            return None
        centroid = entry.get("centroid", None)
        if not isinstance(centroid, (list, tuple)) or len(centroid) < 2:
            return None
        return float(centroid[0]), float(centroid[1])

    def render_current(self) -> None:
        if not self.files:
            return
        fp = self.files[self.idx]
        bgr = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if bgr is None:
            return

        glints = self._get_glints_for_image(fp.name)
        pupil = self._get_pupil_for_image(fp.name)
        template_xy = self._get_template_for_image(fp.name) if self.show_template_var.get() else None

        overlay = bgr.copy()
        # draw template positions (if present)
        if template_xy is not None:
            for i in range(min(self.n_glints, template_xy.shape[0])):
                pt = template_xy[i]
                if not _is_valid_xy(pt):
                    continue
                x, y = int(round(pt[0])), int(round(pt[1]))
                cv2.circle(overlay, (x, y), 7, (0, 255, 0), 2)
                cv2.putText(overlay, f"T{i}", (x + 6, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # draw detected glints
        for i in range(min(self.n_glints, glints.shape[0])):
            pt = glints[i]
            if not _is_valid_xy(pt):
                continue
            x, y = int(round(pt[0])), int(round(pt[1]))
            cv2.circle(overlay, (x, y), 6, (255, 0, 0), 2)
            cv2.putText(overlay, f"T{i}", (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # draw pupil
        if pupil is not None and np.isfinite(pupil[0]) and np.isfinite(pupil[1]) and pupil[0] >= 0 and pupil[1] >= 0:
            px, py = int(round(pupil[0])), int(round(pupil[1]))
            cv2.circle(overlay, (px, py), 8, (0, 200, 255), 2)
            cv2.putText(overlay, "P", (px + 6, py + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        if self.zoom != 1.0:
            z = float(self.zoom)
            overlay = cv2.resize(overlay, (int(overlay.shape[1] * z), int(overlay.shape[0] * z)), interpolation=cv2.INTER_LINEAR)

        rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.config(width=img.width, height=img.height, scrollregion=(0, 0, img.width, img.height))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.status_var.set(f"{self.idx + 1}/{len(self.files)}: {fp.name}")

    def on_slider(self, val) -> None:
        if not self.files:
            return
        try:
            idx = int(float(val))
        except Exception:
            return
        if idx != self.idx:
            self.idx = idx
            self.render_current()

    def _hit_test_points(self, arr: np.ndarray, x: float, y: float, radius: float = 10.0) -> Optional[int]:
        if not self.files:
            return None
        fp = self.files[self.idx]
        best = None
        best_d = float("inf")
        for i in range(min(self.n_glints, arr.shape[0])):
            pt = arr[i]
            if not _is_valid_xy(pt):
                continue
            d = float(np.hypot(pt[0] - x, pt[1] - y))
            if d < best_d:
                best_d = d
                best = i
        if best is None or best_d > radius:
            return None
        return best

    def _valid_mask(self, arr: np.ndarray) -> np.ndarray:
        if arr.size == 0:
            return np.zeros((0,), dtype=bool)
        return np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1]) & (arr[:, 0] >= 0) & (arr[:, 1] >= 0)

    def _hit_test_pupil(self, x: float, y: float, radius: float = 12.0) -> bool:
        if not self.files:
            return False
        fp = self.files[self.idx]
        pupil = self._get_pupil_for_image(fp.name)
        if pupil is None:
            return False
        if not np.isfinite(pupil[0]) or not np.isfinite(pupil[1]):
            return False
        d = float(np.hypot(pupil[0] - x, pupil[1] - y))
        return d <= radius

    def on_canvas_press(self, event) -> None:
        if not self.files:
            return
        x = event.x / float(self.zoom)
        y = event.y / float(self.zoom)
        if self.edit_target.get() == "pupil":
            if self._hit_test_pupil(x, y, radius=12.0 / max(self.zoom, 1e-6)):
                self.selected_kind = "pupil"
                self.selected_idx = None
                self.drag_start = (x, y)
                return
        arr = self._get_edit_array(self.files[self.idx].name)
        if arr is None:
            return
        hit = self._hit_test_points(arr, x, y, radius=12.0 / max(self.zoom, 1e-6))
        if hit is not None:
            self.selected_kind = "glint"
            self.selected_idx = hit
            self.drag_start = (x, y)
            if self.edit_constellation_var.get():
                self.drag_mode = "constellation"
                self.drag_base_glints = np.asarray(arr, dtype=float)
                valid = self._valid_mask(self.drag_base_glints)
                if np.any(valid):
                    self.drag_centroid = np.mean(self.drag_base_glints[valid], axis=0)
                else:
                    self.drag_centroid = np.array([x, y], dtype=float)
            else:
                self.drag_mode = "point"
            return

    def on_canvas_drag(self, event) -> None:
        if not self.files or self.selected_kind is None:
            return
        x = event.x / float(self.zoom)
        y = event.y / float(self.zoom)
        fp = self.files[self.idx]
        if self.selected_kind == "glint":
            arr = self._get_edit_array(fp.name)
            if arr is None:
                return
            arr = arr.copy()
            if self.drag_mode == "constellation" and self.drag_start is not None and self.drag_base_glints is not None:
                base = np.asarray(self.drag_base_glints, dtype=float)
                valid = self._valid_mask(base)
                new = base.copy()
                if np.any(valid):
                    if self.rotate_constellation_var.get():
                        centroid = self.drag_centroid if self.drag_centroid is not None else np.array([x, y], dtype=float)
                        v0 = np.array(self.drag_start, dtype=float) - centroid
                        v1 = np.array([x, y], dtype=float) - centroid
                        if np.hypot(v0[0], v0[1]) > 1e-6 and np.hypot(v1[0], v1[1]) > 1e-6:
                            a0 = float(np.arctan2(v0[1], v0[0]))
                            a1 = float(np.arctan2(v1[1], v1[0]))
                            ang = a1 - a0
                            c, s = float(np.cos(ang)), float(np.sin(ang))
                            R = np.array([[c, -s], [s, c]], dtype=float)
                            shifted = base[valid] - centroid
                            rotated = shifted @ R.T
                            new[valid] = rotated + centroid
                    elif self.scale_constellation_var.get():
                        centroid = self.drag_centroid if self.drag_centroid is not None else np.array([x, y], dtype=float)
                        v0 = np.array(self.drag_start, dtype=float) - centroid
                        v1 = np.array([x, y], dtype=float) - centroid
                        r0 = float(np.hypot(v0[0], v0[1]))
                        r1 = float(np.hypot(v1[0], v1[1]))
                        if r0 > 1e-6 and r1 > 1e-6:
                            s = r1 / r0
                            new[valid] = centroid + (base[valid] - centroid) * s
                    else:
                        dx = x - self.drag_start[0]
                        dy = y - self.drag_start[1]
                        new[valid] = base[valid] + np.array([dx, dy], dtype=float)
                self._set_edit_array(fp.name, new)
                self.render_current()
            elif self.selected_idx is not None:
                if arr.shape[0] <= self.selected_idx:
                    return
                arr[self.selected_idx] = (float(x), float(y))
                self._set_edit_array(fp.name, arr)
                self.render_current()
        elif self.selected_kind == "pupil":
            entry = self.pupil_by_name.get(fp.name, {})
            entry = dict(entry) if isinstance(entry, dict) else {}
            entry["centroid"] = (float(x), float(y))
            self.pupil_by_name[fp.name] = entry
            self.render_current()

    def on_canvas_release(self, event) -> None:
        self.selected_kind = None
        self.selected_idx = None
        self.drag_start = None
        self.drag_mode = None
        self.drag_base_glints = None
        self.drag_centroid = None

    def _scale_constellation(self, factor: float) -> None:
        if not self.files:
            return
        if self.edit_target.get() == "pupil":
            return
        fp = self.files[self.idx]
        arr = self._get_edit_array(fp.name)
        if arr is None:
            return
        arr = arr.copy()
        valid = self._valid_mask(arr)
        if not np.any(valid):
            return
        centroid = np.mean(arr[valid], axis=0)
        arr[valid] = centroid + (arr[valid] - centroid) * factor
        self._set_edit_array(fp.name, arr)
        self.render_current()

    def on_scale_up(self, _event=None) -> None:
        if not self.edit_constellation_var.get():
            return
        self._scale_constellation(1.02)

    def on_scale_down(self, _event=None) -> None:
        if not self.edit_constellation_var.get():
            return
        self._scale_constellation(0.98)

    def on_canvas_right_click(self, event) -> None:
        if not self.files:
            return
        if self.edit_target.get() == "pupil":
            return
        x = event.x / float(self.zoom)
        y = event.y / float(self.zoom)
        fp = self.files[self.idx]
        arr = self._get_edit_array(fp.name)
        if arr is None:
            return
        hit = self._hit_test_points(arr, x, y, radius=12.0 / max(self.zoom, 1e-6))
        if hit is None:
            return
        arr = arr.copy()
        arr[hit] = (-1.0, -1.0)
        self._set_edit_array(fp.name, arr)
        self.render_current()

    def delete_selected(self) -> None:
        if not self.files or self.selected_kind is None:
            return
        fp = self.files[self.idx]
        if self.selected_kind == "glint" and self.selected_idx is not None:
            arr = self._get_edit_array(fp.name)
            if arr is None:
                return
            arr = arr.copy()
            arr[self.selected_idx] = (-1.0, -1.0)
            self._set_edit_array(fp.name, arr)
        elif self.selected_kind == "pupil":
            entry = self.pupil_by_name.get(fp.name, {})
            entry = dict(entry) if isinstance(entry, dict) else {}
            entry["centroid"] = (-1.0, -1.0)
            entry["major_axis_length"] = None
            entry["minor_axis_length"] = None
            self.pupil_by_name[fp.name] = entry
        self.render_current()

    def save_glints_npz(self) -> None:
        if not self.glints_by_name:
            messagebox.showwarning("Save glints", "No glints loaded.")
            return
        out_path = filedialog.asksaveasfilename(
            title="Save glints",
            defaultextension=".npz",
            filetypes=[("NumPy NPZ", "*.npz"), ("All files", "*.*")],
            initialfile="glints_updated.npz",
        )
        if not out_path:
            return
        config = {"glints_source": str(self.glints_path) if self.glints_path else None}
        payload = {"glints": self.glints_by_name, "config": config}
        if self.template_by_name:
            payload["template_xy"] = self.template_by_name
        np.savez(out_path, **payload)
        messagebox.showinfo("Save glints", f"Saved: {Path(out_path).name}")

    def save_pupil_npz(self) -> None:
        if not self.pupil_by_name:
            messagebox.showwarning("Save pupils", "No pupil annotations loaded.")
            return
        out_path = filedialog.asksaveasfilename(
            title="Save pupil npz",
            defaultextension=".npz",
            filetypes=[("NumPy NPZ", "*.npz"), ("All files", "*.*")],
            initialfile="pupil_updated.npz",
        )
        if not out_path:
            return
        filenames = []
        ellipses = []
        for fp in self.files:
            name = fp.name
            entry = self.pupil_by_name.get(name, {})
            if not isinstance(entry, dict):
                entry = {}
            if "centroid" not in entry:
                entry["centroid"] = (-1.0, -1.0)
            filenames.append(name)
            ellipses.append(entry)
        np.savez(out_path, filenames=np.array(filenames, dtype=object), ellipses=np.array(ellipses, dtype=object))
        messagebox.showinfo("Save pupils", f"Saved: {Path(out_path).name}")

    def next_image(self) -> None:
        if not self.files:
            return
        self.idx = (self.idx + 1) % len(self.files)
        self.slider.set(self.idx)
        self.render_current()

    def prev_image(self) -> None:
        if not self.files:
            return
        self.idx = (self.idx - 1) % len(self.files)
        self.slider.set(self.idx)
        self.render_current()

    def toggle_play(self) -> None:
        if not self.files:
            return
        self.playing = not self.playing
        if self.playing:
            self._tick()

    def _tick(self) -> None:
        if not self.playing:
            return
        self.next_image()
        self.after_id = self.root.after(100, self._tick)

    def zoom_in(self) -> None:
        self.zoom = min(8.0, self.zoom * 1.25)
        self.render_current()

    def zoom_out(self) -> None:
        self.zoom = max(0.2, self.zoom / 1.25)
        self.render_current()

    def zoom_reset(self) -> None:
        self.zoom = 1.0
        self.render_current()


def main() -> None:
    root = tk.Tk()
    AnnotationReviewApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
