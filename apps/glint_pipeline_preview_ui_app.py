import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import json
import hashlib
import logging
import cv2
import numpy as np
import re
import threading
import multiprocessing as mp
import types
import tempfile
import zipfile
import shutil
import time
import sys
from collections import OrderedDict

from glint_pipeline import eval_gen as g
from glint_pipeline.temporal import MultiGlintTracker
from glint_pipeline.pupil_roi import compute_pupil_roi, map_points_to_full, resolve_pupil_roi_center

try:
    from PIL import Image, ImageTk
except ImportError as exc:
    raise SystemExit("Pillow is required. Install with: python -m pip install pillow") from exc


_LOGGER = logging.getLogger(__name__)
if not _LOGGER.handlers:
    logging.basicConfig(level=logging.INFO)

_OVERLAY_CACHE_MAX = 32

# Whitelist of args that affect compute results (detection/scoring/matching/temporal).
_COMPUTE_ATTRS = (
    "matcher",
    "template_mode",
    "score2_mode",
    "ml_model_path",
    "percentile",
    "kernel",
    "enhance_mode",
    "enhance_enable",
    "median_ksize",
    "denoise",
    "denoise_k",
    "clahe",
    "clahe_clip",
    "clahe_tiles",
    "gamma",
    "unsharp",
    "unsharp_amount",
    "unsharp_sigma",
    "minmax",
    "clean_k",
    "open_iter",
    "close_iter",
    "eps",
    "max_pool",
    "min_area",
    "max_area",
    "min_circ",
    "min_maxI",
    "cand_fallback",
    "cand_target_raw",
    "cand_fallback_passes",
    "cand_fallback_percentiles",
    "cand_fallback_kernel_add",
    "cand_merge_eps",
    "support_M",
    "support_tol",
    "support_w",
    "contrast_r_inner",
    "contrast_r_outer1",
    "contrast_r_outer2",
    "dog_sigma1",
    "dog_sigma2",
    "ratio_tol",
    "pivot_P",
    "max_seeds",
    "grow_resid_max",
    "layout_prior",
    "layout_lambda",
    "layout_mode",
    "sla_layout_prior",
    "sla_layout_lambda",
    "sla_layout_mode",
    "sla_semantic_prior",
    "sla_semantic_mode",
    "sla_semantic_lambda",
    "sla_semantic_hard",
    "sla_mirror_reject",
    "sla_top2_margin",
    "sla_base_ratio_min",
    "sla_side_margin",
    "sla_scale_min",
    "sla_scale_max",
    "sla_g0_top2",
    "sla_w_seed_score2",
    "sla_w_seed_geom",
    "max_seeds_per_pivot",
    "sla_adaptive_ratio_tol",
    "sla_ratio_tol_min",
    "sla_ratio_tol_refN",
    "temporal",
    "temporal_prior",
    "temporal_gate_px",
    "temporal_max_missed",
    "temporal_lambda",
    "temporal_w_scale",
    "temporal_w_rot",
    "temporal_w_trans",
    "temporal_use_tracks_for_matching",
    "temporal_roi_radius",
    "vote_M",
    "vote_ratio_tol",
    "vote_max_hyp",
    "vote_w_score2",
    "min_k",
    "iters",
    "seed",
    "scale_min",
    "scale_max",
    "disable_scale_gate",
    "matching",
    "appearance_tiebreak",
    "roi_mode",
    "roi_border_frac",
    "roi_border_px",
    "pupil_roi",
    "pupil_roi_size",
    "pupil_roi_pad_mode",
    "pupil_roi_pad_value",
    "pupil_roi_fail_policy",
    "pupil_source",
    "pupil_axis_mode",
    "pupil_dark_thresh",
    "pupil_min_area",
    "pupil_rmin",
    "pupil_rmax",
    "pupil_fallback_center",
    "pupil_method",
    "pupil_radii",
    "pupil_sigma_frac",
    "pupil_fail_open",
    "pupil_force_gate",
    "pupil_npz",
    "auto_scale",
    "ref_width",
    "min_kernel",
    "min_inliers",
    "template_bank_source",
    "template_bank_path",
    "bank_select_metric",
    "template_build_mode",
    "mirror",
)


def _stable_json_dumps(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, default=str)


def _hash_payload(payload: dict) -> str:
    blob = _stable_json_dumps(payload)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def _file_fingerprint(path: str | None) -> dict | None:
    if not path:
        return None
    p = Path(path)
    try:
        stat = p.stat()
        return {"path": str(p.resolve()), "mtime_ns": stat.st_mtime_ns, "size": stat.st_size}
    except Exception:
        return {"path": str(p)}


def _default_template_path() -> str:
    try:
        return str(Path(__file__).resolve().parents[1] / "templates" / "default_templates.json")
    except Exception:
        return "templates/default_templates.json"


def _compute_key_payload(args, templates_path: str | None, image_config_path: str | None, pupil_npz_path: str | None) -> dict:
    payload = {}
    for key in _COMPUTE_ATTRS:
        payload[key] = getattr(args, key, None)
    if getattr(args, "template_bank_source", None) == "custom":
        payload["template_bank_file"] = _file_fingerprint(getattr(args, "template_bank_path", None) or templates_path)
    else:
        payload["template_bank_file"] = _file_fingerprint(_default_template_path())
    payload["image_config_file"] = _file_fingerprint(image_config_path or getattr(args, "image_config", None))
    payload["pupil_npz_file"] = _file_fingerprint(pupil_npz_path or getattr(args, "pupil_npz", None))
    return payload


def _overlay_key_payload(args, overrides_key, corr_mode: bool) -> dict:
    return {
        "show_overlay": bool(getattr(args, "show_overlay", True)),
        "preview_enhanced": bool(getattr(args, "preview_enhanced", False)),
        "pupil_roi_debug": bool(getattr(args, "pupil_roi_debug", False)),
        "match_tol": float(getattr(args, "match_tol", 0.0)),
        "corr_mode": bool(corr_mode),
        "overrides": overrides_key,
    }


_MP_SAVE_STATE = {}


class _Tooltip:
    def __init__(self, root: tk.Tk, delay_ms: int = 500, wraplength: int = 440) -> None:
        self.root = root
        self.delay_ms = int(delay_ms)
        self.wraplength = int(wraplength)
        self._after_id = None
        self._tipwin = None
        self._text = ""
        self._x = 0
        self._y = 0

    def bind(self, widget, text: str) -> None:
        if not text:
            return
        widget.bind("<Enter>", lambda e, w=widget, t=text: self._schedule(w, t, e), add="+")
        widget.bind("<Leave>", lambda _e: self.hide(), add="+")
        widget.bind("<ButtonPress>", lambda _e: self.hide(), add="+")

    def _schedule(self, _widget, text: str, event) -> None:
        self._text = text
        self._x = int(getattr(event, "x_root", 0)) + 16
        self._y = int(getattr(event, "y_root", 0)) + 12
        self._cancel()
        self._after_id = self.root.after(self.delay_ms, self._show)

    def _cancel(self) -> None:
        if self._after_id is None:
            return
        try:
            self.root.after_cancel(self._after_id)
        except Exception:
            pass
        self._after_id = None

    def hide(self) -> None:
        self._cancel()
        if self._tipwin is None:
            return
        try:
            self._tipwin.destroy()
        except Exception:
            pass
        self._tipwin = None

    def _show(self) -> None:
        self._after_id = None
        if self._tipwin is not None or not self._text:
            return
        try:
            tw = tk.Toplevel(self.root)
            tw.wm_overrideredirect(True)
            tw.wm_geometry(f"+{self._x}+{self._y}")
            label = tk.Label(
                tw,
                text=self._text,
                justify=tk.LEFT,
                background="#ffffe0",
                relief=tk.SOLID,
                borderwidth=1,
                wraplength=self.wraplength,
                font=("Segoe UI", 9),
            )
            label.pack(ipadx=6, ipady=4)
            self._tipwin = tw
        except Exception:
            self._tipwin = None


def _mp_save_init(
    args_dict,
    template,
    bank_templates,
    ratio_index_bank,
    ratio_index_single,
    d_expected,
    overrides_by_image,
):
    args = types.SimpleNamespace(**args_dict)
    args.temporal = False
    args.temporal_prior = False
    pupil_npz_map = None
    if getattr(args, "pupil_npz", None):
        try:
            pupil_npz_map = g.load_pupil_npz(args.pupil_npz)
        except Exception:
            pupil_npz_map = None
    pupil_radii_raw = [int(x) for x in str(args.pupil_radii).split(",") if x.strip().isdigit()]
    if not pupil_radii_raw:
        pupil_radii_raw = [12, 16, 20, 24, 28, 32]
    _MP_SAVE_STATE.clear()
    _MP_SAVE_STATE.update(
        dict(
            args=args,
            template=template,
            bank_templates=bank_templates,
            ratio_index_bank=ratio_index_bank,
            ratio_index_single=ratio_index_single,
            d_expected=d_expected,
            overrides_by_image=overrides_by_image,
            pupil_npz_map=pupil_npz_map,
            pupil_radii_raw=pupil_radii_raw,
            n_glints=int(template.shape[0]) if template is not None else 0,
        )
    )


def _mp_save_process_one(fp_str: str):
    state = _MP_SAVE_STATE
    args = state["args"]
    fp = Path(fp_str)
    bgr = cv2.imread(str(fp), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    if bool(getattr(args, "mirror", False)):
        bgr = cv2.flip(bgr, 1)
    gray_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    params = g.scale_params_for_image(args, w=gray_full.shape[1], h=gray_full.shape[0])
    H, W = gray_full.shape[:2]
    image_center = (0.5 * W, 0.5 * H)

    pcx = pcy = pr = None
    pupil_detected = False
    pupil_ok = False
    pupil_source_used = "none"
    if args.pupil_roi and args.pupil_source != "none":
        pcx, pcy, pr, pupil_detected, pupil_ok, pupil_source_used = g.detect_pupil_center_for_frame(
            gray_full, None, fp.name, W, H, args, params, state["pupil_radii_raw"], state["pupil_npz_map"]
        )

    roi_active = False
    roi_info = None
    roi_decision = None
    gray = gray_full
    if args.pupil_roi:
        roi_decision = resolve_pupil_roi_center(
            (pcx, pcy) if pcx is not None and pcy is not None else None,
            pr,
            W,
            H,
            args.pupil_roi_fail_policy,
            None,
        )
        if roi_decision.action == "skip":
            return None
        if roi_decision.action == "use" and roi_decision.center is not None:
            roi_info = compute_pupil_roi(
                gray_full,
                roi_decision.center,
                size=int(args.pupil_roi_size),
                pad_mode=str(args.pupil_roi_pad_mode),
                pad_value=int(args.pupil_roi_pad_value),
            )
            gray = roi_info.roi_img
            roi_active = True

    cand_xy_pass0, rows_pass0, cand_score2_pass0, cand_raw_pass0, cand_support_pass0 = g.detect_candidates_one_pass(
        gray, params, args, d_expected=state["d_expected"]
    )
    cand_xy_raw = cand_xy_pass0
    cand_score2_raw = cand_score2_pass0
    cand_support_raw = cand_support_pass0
    cand_raw_merged = int(cand_raw_pass0)

    if args.cand_fallback and cand_raw_pass0 < args.cand_target_raw:
        try:
            fallback_percentiles = [float(x) for x in args.cand_fallback_percentiles.split(",") if x.strip()]
        except Exception:
            fallback_percentiles = []
        if not fallback_percentiles:
            fallback_percentiles = [args.percentile]

        cand_xy_list = [cand_xy_pass0]
        cand_score2_list = [cand_score2_pass0]
        cand_support_list = [cand_support_pass0]
        for i in range(1, int(args.cand_fallback_passes) + 1):
            idx = min(i - 1, len(fallback_percentiles) - 1)
            perc = fallback_percentiles[idx]
            kernel_add = int(args.cand_fallback_kernel_add) * i
            cand_xy_i, rows_i, cand_score2_i, cand_raw_i, cand_support_i = g.detect_candidates_one_pass(
                gray, params, args, d_expected=state["d_expected"], percentile_override=perc, kernel_add=kernel_add
            )
            cand_xy_list.append(cand_xy_i)
            cand_score2_list.append(cand_score2_i)
            cand_support_list.append(cand_support_i)
            cand_xy_raw, cand_score2_raw, cand_support_raw = g.merge_candidates(
                cand_xy_list, cand_score2_list, cand_support_list, merge_eps=float(args.cand_merge_eps)
            )
            cand_raw_merged = int(len(cand_xy_raw))
            if cand_raw_merged >= args.cand_target_raw:
                break

    if roi_active and roi_info is not None:
        cand_xy_raw = map_points_to_full(cand_xy_raw, roi_info.offset_x, roi_info.offset_y)
        if cand_xy_raw.size > 0:
            in_bounds = (
                (cand_xy_raw[:, 0] >= 0)
                & (cand_xy_raw[:, 0] < W)
                & (cand_xy_raw[:, 1] >= 0)
                & (cand_xy_raw[:, 1] < H)
            )
            cand_xy_raw = cand_xy_raw[in_bounds]
            cand_score2_raw = cand_score2_raw[in_bounds]
            cand_support_raw = cand_support_raw[in_bounds]
        cand_raw_merged = int(len(cand_xy_raw))

    rows_sorted = sorted(
        zip(cand_xy_raw, cand_score2_raw),
        key=lambda p: p[1],
        reverse=True,
    )[: args.max_pool]
    cand_xy = np.array([p[0] for p in rows_sorted], dtype=float) if rows_sorted else np.empty((0, 2), dtype=float)
    cand_score2 = np.array([p[1] for p in rows_sorted], dtype=float) if rows_sorted else np.empty((0,), dtype=float)
    roi_shape = gray_full.shape if roi_active else gray.shape
    cand_xy, cand_score2, roi_margin = g.filter_candidates_roi(
        cand_xy, cand_score2, roi_shape, mode=args.roi_mode,
        border_frac=args.roi_border_frac, border_px=args.roi_border_px
    )
    roi_center = image_center
    roi_radius_estimate = max(1.0, 0.5 * (min(W, H) - 2 * roi_margin))
    pcx_gate = pcx
    pcy_gate = pcy
    pr_gate = pr
    if roi_decision is not None and roi_decision.action == "use" and roi_decision.center is not None:
        pcx_gate, pcy_gate = roi_decision.center
        if roi_decision.radius is not None and np.isfinite(roi_decision.radius):
            pr_gate = roi_decision.radius
    if args.pupil_roi and len(cand_xy) >= 2 and args.pupil_source != "none":
        if not pupil_ok:
            if args.pupil_fail_open:
                pcx_gate = pcy_gate = pr_gate = None
            else:
                if args.pupil_fallback_center == "image":
                    pcx_gate, pcy_gate = image_center
                    pr_gate = 0.25 * min(W, H)
                else:
                    pcx_gate, pcy_gate = roi_center
                    pr_gate = roi_radius_estimate
        if pcx_gate is not None and pcy_gate is not None and pr_gate is not None:
            pupil_mask = g.gate_candidates_by_pupil(
                cand_xy, (pcx_gate, pcy_gate), pr_gate, args.pupil_rmin, args.pupil_rmax
            )
            mask_sum = int(pupil_mask.sum())
            if mask_sum >= args.min_k or args.pupil_force_gate:
                cand_xy = cand_xy[pupil_mask]
                cand_score2 = cand_score2[pupil_mask]

    best = None
    best_key = None
    chosen_template_idx = None
    T_hat = None
    matches = None
    best_template_xy = state["template"]
    if len(cand_xy) >= args.min_k:
        if args.template_mode == "single":
            best = g.run_matcher_for_template(
                state["template"], cand_xy, cand_score2, args, None, params["eps_eff"], state["ratio_index_single"],
                cand_raw_count=cand_raw_merged
            )
            best_template_xy = state["template"]
        else:
            bank = state["bank_templates"] if state["bank_templates"] is not None else g.load_template_bank(args)
            for bi, template_xy in enumerate(bank):
                ratio_idx = None
                if state["ratio_index_bank"] is not None and bi < len(state["ratio_index_bank"]):
                    ratio_idx = state["ratio_index_bank"][bi]
                res = g.run_matcher_for_template(
                    template_xy, cand_xy, cand_score2, args, None, params["eps_eff"], ratio_idx,
                    cand_raw_count=cand_raw_merged
                )
                if res is None:
                    continue
                key = g.score_match_result(
                    res, cand_score2, args.appearance_tiebreak, args.bank_select_metric, s_expected=None
                )
                if best is None or key > best_key:
                    best = res
                    best_key = key
                    chosen_template_idx = bi
                    best_template_xy = template_xy

    if best is not None:
        s_fit, _, _, matches, T_hat, app_sum = best

    overrides = state["overrides_by_image"].get(fp.name, {})
    cand_xy_disp = cand_xy.copy() if isinstance(cand_xy, np.ndarray) else np.array(cand_xy, dtype=float)
    matches_disp = list(matches) if matches is not None else []
    if overrides and T_hat is not None:
        for ti, pt in overrides.items():
            x, y = pt
            cand_xy_disp = np.vstack([cand_xy_disp, np.array([[x, y]], dtype=float)])
            new_ci = len(cand_xy_disp) - 1
            d = float(np.linalg.norm(T_hat[int(ti)] - cand_xy_disp[new_ci]))
            replaced = False
            for mi, (tti, _, _) in enumerate(matches_disp):
                if int(tti) == int(ti):
                    matches_disp[mi] = (int(ti), int(new_ci), d)
                    replaced = True
                    break
            if not replaced:
                matches_disp.append((int(ti), int(new_ci), d))

    n_glints = int(state["n_glints"])
    glint_xy = np.full((n_glints, 2), np.nan, dtype=float)
    if matches_disp is not None:
        for ti, ci, _ in matches_disp:
            if 0 <= int(ti) < n_glints and 0 <= int(ci) < len(cand_xy_disp):
                glint_xy[int(ti)] = cand_xy_disp[int(ci)]

    if T_hat is not None:
        template_xy = np.array(T_hat, dtype=float)
    else:
        template_xy = np.full((n_glints, 2), np.nan, dtype=float)

    return {"name": fp.name, "glint_xy": glint_xy, "template_xy": template_xy}


def _run_save_multiproc(files, args, template, bank_templates, ratio_index_bank, ratio_index_single, d_expected, overrides_by_image, workers):
    glints_by_image = {}
    template_by_image = {}
    ctx = mp.get_context("spawn")
    args_dict = vars(args)
    with ctx.Pool(
        processes=int(workers),
        initializer=_mp_save_init,
        initargs=(args_dict, template, bank_templates, ratio_index_bank, ratio_index_single, d_expected, overrides_by_image),
    ) as pool:
        file_list = [str(p) for p in files]
        for res in pool.imap_unordered(_mp_save_process_one, file_list):
            if res is None:
                continue
            glints_by_image[res["name"]] = res["glint_xy"]
            template_by_image[res["name"]] = res["template_xy"]
    return glints_by_image, template_by_image


class PreviewApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Glint Pipeline Preview")

        
        self.folder: Path | None = None
        self.files: list[Path] = []
        self.zip_temp_dir: Path | None = None
        self.idx = 0
        self.playing = False
        self.after_id = None
        self.render_request_id = None
        self.photo = None
        self.canvas_image_id = None
        self.canvas_size = None
        self.zoom = 1.0
        self.mirror = False
        self.template = None
        self.bank_templates = None
        self.ratio_index_bank = None
        self.ratio_index_single = None
        self.d_expected = None
        self.templates_path: str | None = None
        self.image_config_path: str | None = None
        self.pupil_npz_path: str | None = None
        self.pupil_npz_map = None
        self.current_fp = None
        self.current_matches = None
        self.current_cand_xy = None
        self.overrides_by_image = {}
        self.glints_by_image = {}
        self.template_by_image = {}
        self.drag_ti = None
        self.drag_mode = None
        self.drag_start = None
        self.drag_base_glints = None
        self.drag_centroid = None
        self.temporal_tracker = None
        self.prev_params = None
        self.last_idx = None
        self.last_good_pupil = None
        self.bulk_processing = False
        # Two-level caching: compute results per frame+compute_key, overlay LRU for renders.
        self.compute_cache = {}
        self.overlay_cache = OrderedDict()
        self.overlay_cache_max = _OVERLAY_CACHE_MAX
        self.compute_key = None
        self.cache_lock = threading.Lock()
        self.cache_thread = None
        self.cache_token = 0
        self.cache_building = False
        self.cache_debounce_id = None
        self.cache_progress_var = tk.DoubleVar(value=0.0)
        self.save_workers_var = tk.IntVar(value=0)
        self.default_settings_path = Path(r"C:\Users\vbmaq\Documents\virnet2\data\templates\chugh\preview_ui_settings_b2_081_win.json")
        self.current_preview_base = None
        self.current_preview_base_name = None
        self.current_preview_base_cache_key = None
        self.args_dirty = True
        self.args_cached = None
        self.args_cached_key = None
        self.ondemand_lock = threading.Lock()
        self.ondemand_inflight = set()

        self._build_ui()
        if self.default_settings_path.exists():
            try:
                data = json.loads(self.default_settings_path.read_text(encoding="utf-8"))
                vars_data = data.get("vars", {})
                for k, v in vars_data.items():
                    if k in self.vars:
                        try:
                            self.vars[k].set(v)
                        except Exception:
                            pass
                self.templates_path = data.get("templates_path")
                self.image_config_path = data.get("image_config_path")
                self.pupil_npz_path = data.get("pupil_npz_path")
                if self.pupil_npz_path:
                    try:
                        self.pupil_npz_map = g.load_pupil_npz(self.pupil_npz_path)
                    except Exception:
                        pass
                self.cfg_var.set(f"settings: {self.default_settings_path.name}")
            except Exception:
                pass

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=6)
        top.pack(fill=tk.X)
        ttk.Button(top, text="Load Folder", command=self.load_folder).pack(side=tk.LEFT, padx=3)
        ttk.Button(top, text="Load Zip", command=self.load_zip).pack(side=tk.LEFT, padx=3)
        ttk.Button(top, text="Save NPZ", command=self.save_npz).pack(side=tk.LEFT, padx=3)
        ttk.Button(top, text="Prev", command=self.prev_image).pack(side=tk.LEFT, padx=3)
        ttk.Button(top, text="Next", command=self.next_image).pack(side=tk.LEFT, padx=3)
        ttk.Button(top, text="Zoom +", command=self.zoom_in).pack(side=tk.LEFT, padx=3)
        ttk.Button(top, text="Zoom -", command=self.zoom_out).pack(side=tk.LEFT, padx=3)
        ttk.Button(top, text="Zoom 1:1", command=self.zoom_reset).pack(side=tk.LEFT, padx=3)
        ttk.Label(top, text="Save workers").pack(side=tk.LEFT, padx=(10, 2))
        ttk.Entry(top, textvariable=self.save_workers_var, width=4).pack(side=tk.LEFT)
        self.mirror_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="Mirror", variable=self.mirror_var, command=self._request_render).pack(side=tk.LEFT, padx=6)
        self.play_btn = ttk.Button(top, text="Play", command=self.toggle_play)
        self.play_btn.pack(side=tk.LEFT, padx=3)
        self.status_var = tk.StringVar(value="No folder loaded.")
        ttk.Label(top, textvariable=self.status_var).pack(side=tk.LEFT, padx=8)
        self.cache_progress = ttk.Progressbar(
            top,
            orient=tk.HORIZONTAL,
            length=140,
            mode="determinate",
            maximum=100.0,
            variable=self.cache_progress_var,
        )
        self.cache_progress.pack(side=tk.LEFT, padx=6)

        main = ttk.Frame(self.root, padding=6)
        main.pack(fill=tk.BOTH, expand=True)

        # Paned layout: resizable controls panel + preview
        paned = ttk.Panedwindow(main, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # Controls panel with scroll
        ctrl_container = ttk.Frame(paned, width=360, padding=(0, 0, 6, 0))
        ctrl_canvas = tk.Canvas(ctrl_container, borderwidth=0, highlightthickness=0, width=360)
        ctrl_scroll = ttk.Scrollbar(ctrl_container, orient=tk.VERTICAL, command=ctrl_canvas.yview)
        ctrl_canvas.configure(yscrollcommand=ctrl_scroll.set)
        ctrl_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        ctrl_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ctrl = ttk.Frame(ctrl_canvas)
        ctrl_canvas.create_window((0, 0), window=ctrl, anchor=tk.NW)

        def _on_ctrl_config(_event=None):
            ctrl_canvas.configure(scrollregion=ctrl_canvas.bbox("all"))
            ctrl_canvas.itemconfigure(0, width=ctrl_canvas.winfo_width())

        ctrl.bind("<Configure>", _on_ctrl_config)
        ctrl_canvas.bind("<Configure>", _on_ctrl_config)

        def _on_mousewheel(event):
            ctrl_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        ctrl_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        tooltip = _Tooltip(self.root)
        field_help = {
            "preview_enhanced": "If enabled, the background preview image uses the enhanced version of the frame.\nThis affects only how the preview looks (not the candidate/match computation).",
            "show_overlay": "Draw the detected candidates and matched glints on the image.\nDisable to show the plain grayscale preview (faster).",
            "matcher": "Matching algorithm.\n- ransac: RANSAC-based constellation fit\n- star: star-tracker style voting\n- hybrid: combines ransac + star\n- sla: structured layout-aware matcher",
            "template_mode": "Template source.\n- single: use one template\n- bank: choose best template from a bank",
            "matching": "Final assignment step for matching template points to candidates.\n- greedy: fast heuristic assignment\n- hungarian: optimal assignment (slower)",
            "match_tol": "Pixel tolerance used for visualization (and GT metrics in eval script) when judging a match as correct.",
            "min_inliers": "Minimum number of matched points required to accept a hypothesis.",
            "appearance_tiebreak": "When enabled, uses candidate score2 as an additional tiebreaker between hypotheses.",
            "eps": "Inlier threshold (pixels) used by matchers to consider a candidate consistent with the hypothesis.",
            "max_pool": "Maximum number of candidates (highest score2) passed into matching.",
            "layout_prior": "Enable a soft prior that prefers physically plausible glint layouts.",
            "layout_lambda": "Strength of the layout prior penalty (higher = stronger).",
            "seed": "Random seed used by matchers that sample hypotheses (e.g., RANSAC).",
            "scale_min": "Minimum allowed scale for similarity transform hypotheses.",
            "scale_max": "Maximum allowed scale for similarity transform hypotheses.",
            "disable_scale_gate": "Disable the scale_min/scale_max gating (allows any scale).",
            "min_k": "Number of points used per RANSAC hypothesis (typically 3).",
            "iters": "Number of RANSAC iterations (more = slower but can be more robust).",
            "vote_M": "Star matcher: shortlist size per template point (higher = more hypotheses).",
            "vote_ratio_tol": "Star matcher: tolerance on log-scale ratio (gating of hypothesized scales).",
            "vote_max_hyp": "Star matcher: maximum number of hypotheses to verify.",
            "vote_w_score2": "Star matcher: weight of score2 in vote ranking (0 = geometry-only).",
            "thr_pct": "Percentile threshold for candidate mask (higher = fewer candidates).",
            "kernel": "Top-hat kernel size (odd). Larger finds broader bright spots but can merge structures.",
            "min_area": "Minimum connected-component area (pixels) to keep a candidate blob.",
            "max_area": "Maximum connected-component area (pixels) to keep a candidate blob.",
            "min_circ": "Minimum circularity (0..1) for candidate blobs (higher = more circular).",
            "cand_fallback": "If enabled, runs additional candidate passes if too few raw candidates are found.",
            "cand_target_raw": "Target raw candidate count before fallback stops (used with cand_fallback).",
            "cand_fallback_passes": "Maximum number of fallback passes (used with cand_fallback).",
            "cand_fallback_percentiles": "Comma-separated percentiles to try for fallback passes (e.g. 99.5,99,98.5,98).",
            "score2_mode": "How candidates are scored/ranked before matching.\n- heuristic: simple score\n- contrast: contrast-based\n- contrast_support: contrast + neighborhood support\n- ml_cc: ML model over candidate connected components",
            "ml_model_path": "Path to the ML model file.\nOnly used when score2_mode = ml_cc.",
            "support_M": "Support scoring: number of neighbors to consider.\nOnly used when score2_mode = contrast_support.",
            "support_tol": "Support scoring: tolerance parameter.\nOnly used when score2_mode = contrast_support.",
            "support_w": "Support scoring: weight of support term.\nOnly used when score2_mode = contrast_support.",
            "contrast_r_inner": "Contrast scoring: inner radius (pixels).\nUsed when score2_mode is contrast/contrast_support.",
            "contrast_r_outer1": "Contrast scoring: outer radius 1 (pixels).\nUsed when score2_mode is contrast/contrast_support.",
            "contrast_r_outer2": "Contrast scoring: outer radius 2 (pixels).\nUsed when score2_mode is contrast/contrast_support.",
            "enhance_mode": "Enhancement mode applied before candidate detection (when enhance_enable is on).",
            "enhance_enable": "Enable/disable enhancement pipeline (tophat/dog/highpass).",
            "median_ksize": "Median blur kernel size (odd).",
            "denoise": "Enable denoising as part of enhancement (median blur).",
            "denoise_k": "Override median blur kernel size (0 uses median_ksize). Only used when denoise is enabled.",
            "clahe": "Enable CLAHE (contrast-limited adaptive histogram equalization).",
            "clahe_clip": "CLAHE clip limit. Only used when CLAHE is enabled.",
            "clahe_tiles": "CLAHE tile grid size. Only used when CLAHE is enabled.",
            "gamma_enable": "Enable gamma correction.",
            "gamma": "Gamma correction (>1 darkens, <1 brightens). Only used when gamma_enable is enabled.",
            "unsharp": "Enable unsharp mask sharpening.",
            "unsharp_amount": "Unsharp strength. Only used when unsharp is enabled.",
            "unsharp_sigma": "Unsharp blur sigma. Only used when unsharp is enabled.",
            "minmax": "Enable min-max normalization during enhancement.",
            "clean_k": "Morphology cleanup kernel size for candidate mask.",
            "open_iter": "Morphology open iterations for cleanup.",
            "close_iter": "Morphology close iterations for cleanup.",
            "dog_sigma1": "DoG enhancement: sigma1. Only used when enhance_mode = dog.",
            "dog_sigma2": "DoG enhancement: sigma2. Only used when enhance_mode = dog.",
            "pupil_roi": "Enable pupil-centered ROI cropping for candidate detection (reduces search area).",
            "pupil_roi_size": "ROI size (pixels). Only used when pupil_roi is enabled.",
            "pupil_roi_pad_mode": "Padding mode if ROI goes out of bounds. Only used when pupil_roi is enabled.",
            "pupil_roi_pad_value": "Constant padding value (0..255). Only used when pupil_roi_pad_mode = constant.",
            "pupil_roi_fail_policy": "What to do if pupil center is unavailable.\n- skip: skip frame\n- full_frame: fall back to full image\n- last_good: reuse last good pupil center",
            "pupil_roi_debug": "Draw the ROI rectangle on the preview (debug).",
            "pupil_source": "Where pupil center comes from (auto/labels/naive/swirski/npz/none).",
            "pupil_axis_mode": "Pupil axis mode for interpreting pupil size inputs (auto/radius/diameter).",
            "pivot_P": "SLA matcher: consider top-P pivot candidates by score2.",
            "ratio_tol": "SLA matcher: log-ratio tolerance for geometric consistency.",
            "max_seeds": "SLA matcher: maximum seed hypotheses to try.",
            "grow_resid_max": "SLA matcher: max median residual when growing. Leave blank to use eps.",
            "sla_layout_prior": "SLA-only: enable layout prior inside SLA.",
            "sla_layout_lambda": "SLA-only: strength of SLA layout prior penalty.",
            "sla_semantic_prior": "SLA-only: enable semantic geometry prior.",
            "sla_semantic_mode": "SLA-only: which semantic rules apply (full vs top_only).",
            "sla_semantic_lambda": "SLA-only: semantic penalty weight.",
            "temporal": "Enable temporal tracking/smoothing across frames.",
            "temporal_gate_px": "Temporal tracking: gating radius in pixels.",
            "temporal_max_missed": "Temporal tracking: max consecutive missed frames before a track is dropped.",
            "temporal_lambda": "Temporal tracking: smoothing strength (higher = smoother but less responsive).",
            "temporal_w_scale": "Temporal tracking: weight on scale changes.",
            "temporal_w_rot": "Temporal tracking: weight on rotation changes.",
            "temporal_w_trans": "Temporal tracking: weight on translation changes.",
        }

        self.vars = {}
        self.field_rows = {}
        self.field_pack_opts = {}
        self.field_parent = {}
        self.field_order_by_parent = {}
        self.group_frames = {}
        self.group_pack_opts = {}
        self.group_order = []

        def _register_group(key: str, frame, pack_opts: dict) -> None:
            self.group_frames[key] = frame
            self.group_pack_opts[key] = pack_opts
            self.group_order.append(key)

        def _set_group_visible(key: str, visible: bool) -> None:
            frame = self.group_frames.get(key)
            if frame is None:
                return
            is_packed = frame.winfo_manager() == "pack"
            if visible:
                if is_packed:
                    return
                before_widget = None
                try:
                    idx = self.group_order.index(key)
                except ValueError:
                    idx = -1
                if idx >= 0:
                    for next_key in self.group_order[idx + 1 :]:
                        next_frame = self.group_frames.get(next_key)
                        if next_frame is not None and next_frame.winfo_manager() == "pack":
                            before_widget = next_frame
                            break
                if before_widget is not None:
                    frame.pack(before=before_widget, **self.group_pack_opts.get(key, {"fill": tk.X}))
                else:
                    frame.pack(**self.group_pack_opts.get(key, {"fill": tk.X}))
            else:
                if not is_packed:
                    return
                frame.pack_forget()

        def _register_field_row(key: str, row: ttk.Frame, parent):
            self.field_rows[key] = row
            self.field_parent[key] = parent
            self.field_pack_opts[key] = {"fill": tk.X, "pady": 2}
            self.field_order_by_parent.setdefault(parent, []).append(key)

        def _set_field_visible(key: str, visible: bool) -> None:
            row = self.field_rows.get(key)
            if row is None:
                return
            is_packed = row.winfo_manager() == "pack"
            if visible:
                if is_packed:
                    return
                parent = self.field_parent.get(key)
                order = self.field_order_by_parent.get(parent, [])
                before_widget = None
                try:
                    idx = order.index(key)
                except ValueError:
                    idx = -1
                if idx >= 0:
                    for next_key in order[idx + 1 :]:
                        next_row = self.field_rows.get(next_key)
                        if next_row is not None and next_row.winfo_manager() == "pack":
                            before_widget = next_row
                            break
                if before_widget is not None:
                    row.pack(before=before_widget, **self.field_pack_opts.get(key, {"fill": tk.X, "pady": 2}))
                else:
                    row.pack(**self.field_pack_opts.get(key, {"fill": tk.X, "pady": 2}))
            else:
                if not is_packed:
                    return
                row.pack_forget()

        def _update_dependent_visibility(_evt=None) -> None:
            score2_mode = self.vars.get("score2_mode").get() if "score2_mode" in self.vars else ""
            enhance_mode = self.vars.get("enhance_mode").get() if "enhance_mode" in self.vars else ""
            denoise = bool(self.vars.get("denoise").get()) if "denoise" in self.vars else False
            clahe = bool(self.vars.get("clahe").get()) if "clahe" in self.vars else False
            gamma_enable = bool(self.vars.get("gamma_enable").get()) if "gamma_enable" in self.vars else False
            unsharp = bool(self.vars.get("unsharp").get()) if "unsharp" in self.vars else False
            cand_fallback = bool(self.vars.get("cand_fallback").get()) if "cand_fallback" in self.vars else False
            pupil_roi = bool(self.vars.get("pupil_roi").get()) if "pupil_roi" in self.vars else False
            pad_mode = self.vars.get("pupil_roi_pad_mode").get() if "pupil_roi_pad_mode" in self.vars else ""
            matcher = self.vars.get("matcher").get() if "matcher" in self.vars else ""
            temporal = bool(self.vars.get("temporal").get()) if "temporal" in self.vars else False

            _set_field_visible("ml_model_path", score2_mode == "ml_cc")
            for k in ("contrast_r_inner", "contrast_r_outer1", "contrast_r_outer2"):
                _set_field_visible(k, score2_mode in ("contrast", "contrast_support"))
            for k in ("support_M", "support_tol", "support_w"):
                _set_field_visible(k, score2_mode == "contrast_support")

            _set_field_visible("denoise_k", denoise)
            for k in ("clahe_clip", "clahe_tiles"):
                _set_field_visible(k, clahe)
            _set_field_visible("gamma", gamma_enable)
            for k in ("unsharp_amount", "unsharp_sigma"):
                _set_field_visible(k, unsharp)
            for k in ("dog_sigma1", "dog_sigma2"):
                _set_field_visible(k, enhance_mode == "dog")

            for k in ("cand_target_raw", "cand_fallback_passes", "cand_fallback_percentiles"):
                _set_field_visible(k, cand_fallback)

            for k in (
                "pupil_roi_size",
                "pupil_roi_pad_mode",
                "pupil_roi_pad_value",
                "pupil_roi_fail_policy",
                "pupil_roi_debug",
                "pupil_source",
                "pupil_axis_mode",
            ):
                _set_field_visible(k, pupil_roi)
            _set_field_visible("pupil_roi_pad_value", pupil_roi and pad_mode == "constant")

            _set_field_visible("min_inliers", matcher in ("star", "sla", "hybrid"))
            _set_group_visible("scale", matcher in ("ransac", "star", "hybrid"))
            _set_group_visible("ransac", matcher in ("ransac", "hybrid"))
            _set_group_visible("star", matcher in ("star", "hybrid"))

            for k in ("sla_layout_prior", "sla_layout_lambda", "sla_semantic_prior", "sla_semantic_mode", "sla_semantic_lambda"):
                _set_field_visible(k, matcher == "sla")
            _set_group_visible("sla", matcher == "sla")

            for k in ("temporal_gate_px", "temporal_max_missed", "temporal_lambda", "temporal_w_scale", "temporal_w_rot", "temporal_w_trans"):
                _set_field_visible(k, temporal)

        def add_field(label, key, default, field_type="entry", values=None, parent=None):
            parent = ctrl if parent is None else parent
            row = ttk.Frame(parent)
            row.pack(fill=tk.X, pady=2)
            _register_field_row(key, row, parent)
            lbl = ttk.Label(row, text=label, width=20)
            lbl.pack(side=tk.LEFT)
            tip = field_help.get(key, "")
            tooltip.bind(lbl, tip)
            tooltip.bind(row, tip)
            if field_type == "combobox":
                var = tk.StringVar(value=str(default))
                cb = ttk.Combobox(row, textvariable=var, values=values, width=14, state="readonly")
                cb.pack(side=tk.LEFT, fill=tk.X, expand=True)
                self.vars[key] = var
                tooltip.bind(cb, tip)
            elif field_type == "check":
                var = tk.BooleanVar(value=bool(default))
                chk = ttk.Checkbutton(row, variable=var)
                chk.pack(side=tk.LEFT)
                self.vars[key] = var
                tooltip.bind(chk, tip)
            else:
                var = tk.StringVar(value=str(default))
                ent = ttk.Entry(row, textvariable=var, width=16)
                ent.pack(side=tk.LEFT, fill=tk.X, expand=True)
                self.vars[key] = var
                tooltip.bind(ent, tip)

        # Template/image config load
        cfg_row = ttk.Frame(ctrl)
        cfg_row.pack(fill=tk.X, pady=2)
        ttk.Button(cfg_row, text="Load Templates", command=self.load_templates_json).pack(side=tk.LEFT, padx=2)
        ttk.Button(cfg_row, text="Load Image Config", command=self.load_image_config).pack(side=tk.LEFT, padx=2)
        ttk.Button(cfg_row, text="Load Pupil NPZ", command=self.load_pupil_npz).pack(side=tk.LEFT, padx=2)
        ttk.Button(cfg_row, text="Save Settings", command=self.save_settings).pack(side=tk.LEFT, padx=2)
        ttk.Button(cfg_row, text="Load Settings", command=self.load_settings).pack(side=tk.LEFT, padx=2)
        self.cfg_var = tk.StringVar(value="(no templates/image config loaded)")
        self.default_settings_path = Path(r"C:\Users\vbmaq\Documents\virnet2\data\templates\chugh\preview_ui_settings_b2_081_win.json")
        ttk.Label(ctrl, textvariable=self.cfg_var, wraplength=300).pack(anchor=tk.W, pady=(2, 6))

        # Correction toggle (always visible, near top)
        self.corr_mode = tk.BooleanVar(value=False)
        self.corr_target = tk.StringVar(value="glint")
        self.corr_rotate = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctrl, text="Correction mode", variable=self.corr_mode, command=self._toggle_correction).pack(anchor=tk.W, pady=(2, 6))

        grp_display = ttk.LabelFrame(ctrl, text="Display", padding=6)
        grp_display.pack(fill=tk.X, pady=(0, 8))
        _register_group("display", grp_display, {"fill": tk.X, "pady": (0, 8)})
        add_field("preview_enhanced", "preview_enhanced", False, "check", parent=grp_display)
        add_field("show_overlay", "show_overlay", True, "check", parent=grp_display)

        grp_enhance = ttk.LabelFrame(ctrl, text="Enhancement / Cleanup", padding=6)
        grp_enhance.pack(fill=tk.X, pady=(0, 8))
        _register_group("enhance", grp_enhance, {"fill": tk.X, "pady": (0, 8)})
        add_field("enhance_mode", "enhance_mode", "tophat", "combobox", ["tophat", "dog", "highpass"], parent=grp_enhance)
        add_field("enhance_enable", "enhance_enable", True, "check", parent=grp_enhance)
        add_field("median_ksize", "median_ksize", 3, parent=grp_enhance)
        add_field("denoise", "denoise", True, "check", parent=grp_enhance)
        add_field("denoise_k", "denoise_k", 0, parent=grp_enhance)
        add_field("clahe", "clahe", True, "check", parent=grp_enhance)
        add_field("clahe_clip", "clahe_clip", 2.0, parent=grp_enhance)
        add_field("clahe_tiles", "clahe_tiles", 8, parent=grp_enhance)
        add_field("gamma_enable", "gamma_enable", True, "check", parent=grp_enhance)
        add_field("gamma", "gamma", 11.0, parent=grp_enhance)
        add_field("unsharp", "unsharp", False, "check", parent=grp_enhance)
        add_field("unsharp_amount", "unsharp_amount", 1.0, parent=grp_enhance)
        add_field("unsharp_sigma", "unsharp_sigma", 1.0, parent=grp_enhance)
        add_field("minmax", "minmax", True, "check", parent=grp_enhance)
        add_field("clean_k", "clean_k", 3, parent=grp_enhance)
        add_field("open_iter", "open_iter", 1, parent=grp_enhance)
        add_field("close_iter", "close_iter", 0, parent=grp_enhance)
        add_field("dog_sigma1", "dog_sigma1", 1.0, parent=grp_enhance)
        add_field("dog_sigma2", "dog_sigma2", 2.2, parent=grp_enhance)

        grp_scoring = ttk.LabelFrame(ctrl, text="Scoring (contrast / support / ML)", padding=6)
        grp_scoring.pack(fill=tk.X, pady=(0, 8))
        _register_group("scoring", grp_scoring, {"fill": tk.X, "pady": (0, 8)})
        add_field(
            "score2_mode",
            "score2_mode",
            "contrast_support",
            "combobox",
            ["heuristic", "contrast", "contrast_support", "ml_cc"],
            parent=grp_scoring,
        )
        add_field("ml_model_path", "ml_model_path", "", parent=grp_scoring)
        add_field("support_M", "support_M", 30, parent=grp_scoring)
        add_field("support_tol", "support_tol", 0.10, parent=grp_scoring)
        add_field("support_w", "support_w", 0.15, parent=grp_scoring)
        add_field("contrast_r_inner", "contrast_r_inner", 3, parent=grp_scoring)
        add_field("contrast_r_outer1", "contrast_r_outer1", 5, parent=grp_scoring)
        add_field("contrast_r_outer2", "contrast_r_outer2", 8, parent=grp_scoring)

        grp_candidates = ttk.LabelFrame(ctrl, text="Candidates / Thresholding", padding=6)
        grp_candidates.pack(fill=tk.X, pady=(0, 8))
        _register_group("candidates", grp_candidates, {"fill": tk.X, "pady": (0, 8)})
        add_field("thr_pct", "percentile", 99.7, parent=grp_candidates)
        add_field("kernel", "kernel", 11, parent=grp_candidates)
        add_field("min_area", "min_area", 8, parent=grp_candidates)
        add_field("max_area", "max_area", 250, parent=grp_candidates)
        add_field("min_circ", "min_circ", 0.45, parent=grp_candidates)
        add_field("cand_fallback", "cand_fallback", True, "check", parent=grp_candidates)
        add_field("cand_target_raw", "cand_target_raw", 12, parent=grp_candidates)
        add_field("cand_fallback_passes", "cand_fallback_passes", 4, parent=grp_candidates)
        add_field("cand_fallback_percentiles", "cand_fallback_percentiles", "99.5,99,98.5,98", parent=grp_candidates)

        grp_matching = ttk.LabelFrame(ctrl, text="Matching", padding=6)
        grp_matching.pack(fill=tk.X, pady=(0, 8))
        _register_group("matching", grp_matching, {"fill": tk.X, "pady": (0, 8)})
        add_field("matcher", "matcher", "hybrid", "combobox", ["ransac", "star", "hybrid", "sla"], parent=grp_matching)
        add_field("template_mode", "template_mode", "bank", "combobox", ["single", "bank"], parent=grp_matching)
        add_field("matching", "matching", "greedy", "combobox", ["greedy", "hungarian"], parent=grp_matching)
        add_field("match_tol", "match_tol", 10.0, parent=grp_matching)
        add_field("min_inliers", "min_inliers", 3, parent=grp_matching)
        add_field("appearance_tiebreak", "appearance_tiebreak", False, "check", parent=grp_matching)
        add_field("eps", "eps", 6.0, parent=grp_matching)
        add_field("max_pool", "max_pool", 30, parent=grp_matching)
        add_field("layout_prior", "layout_prior", False, "check", parent=grp_matching)
        add_field("layout_lambda", "layout_lambda", 0.25, parent=grp_matching)

        grp_scale = ttk.LabelFrame(ctrl, text="Scale / Gating", padding=6)
        grp_scale.pack(fill=tk.X, pady=(0, 8))
        _register_group("scale", grp_scale, {"fill": tk.X, "pady": (0, 8)})
        add_field("seed", "seed", 0, parent=grp_scale)
        add_field("scale_min", "scale_min", 0.6, parent=grp_scale)
        add_field("scale_max", "scale_max", 1.6, parent=grp_scale)
        add_field("disable_scale_gate", "disable_scale_gate", False, "check", parent=grp_scale)

        grp_ransac = ttk.LabelFrame(ctrl, text="RANSAC", padding=6)
        grp_ransac.pack(fill=tk.X, pady=(0, 8))
        _register_group("ransac", grp_ransac, {"fill": tk.X, "pady": (0, 8)})
        add_field("min_k", "min_k", 3, parent=grp_ransac)
        add_field("iters", "iters", 4000, parent=grp_ransac)

        grp_star = ttk.LabelFrame(ctrl, text="Star", padding=6)
        grp_star.pack(fill=tk.X, pady=(0, 8))
        _register_group("star", grp_star, {"fill": tk.X, "pady": (0, 8)})
        add_field("vote_M", "vote_M", 8, parent=grp_star)
        add_field("vote_ratio_tol", "vote_ratio_tol", 0.12, parent=grp_star)
        add_field("vote_max_hyp", "vote_max_hyp", 2000, parent=grp_star)
        add_field("vote_w_score2", "vote_w_score2", 0.0, parent=grp_star)

        grp_sla = ttk.LabelFrame(ctrl, text="SLA", padding=6)
        grp_sla.pack(fill=tk.X, pady=(0, 8))
        _register_group("sla", grp_sla, {"fill": tk.X, "pady": (0, 8)})
        add_field("pivot_P", "pivot_P", 8, parent=grp_sla)
        add_field("ratio_tol", "ratio_tol", 0.12, parent=grp_sla)
        add_field("max_seeds", "max_seeds", 200, parent=grp_sla)
        add_field("grow_resid_max", "grow_resid_max", "", parent=grp_sla)
        add_field("sla_layout_prior", "sla_layout_prior", False, "check", parent=grp_sla)
        add_field("sla_layout_lambda", "sla_layout_lambda", 0.25, parent=grp_sla)
        add_field("sla_semantic_prior", "sla_semantic_prior", False, "check", parent=grp_sla)
        add_field("sla_semantic_mode", "sla_semantic_mode", "full", "combobox", ["full", "top_only"], parent=grp_sla)
        add_field("sla_semantic_lambda", "sla_semantic_lambda", 1.5, parent=grp_sla)

        grp_pupil = ttk.LabelFrame(ctrl, text="Pupil ROI", padding=6)
        grp_pupil.pack(fill=tk.X, pady=(0, 8))
        _register_group("pupil", grp_pupil, {"fill": tk.X, "pady": (0, 8)})
        add_field("pupil_roi", "pupil_roi", False, "check", parent=grp_pupil)
        add_field("pupil_roi_size", "pupil_roi_size", 80, parent=grp_pupil)
        add_field("pupil_roi_pad_mode", "pupil_roi_pad_mode", "reflect", "combobox", ["reflect", "constant", "edge"], parent=grp_pupil)
        add_field("pupil_roi_pad_value", "pupil_roi_pad_value", 0, parent=grp_pupil)
        add_field("pupil_roi_fail_policy", "pupil_roi_fail_policy", "skip", "combobox", ["skip", "full_frame", "last_good"], parent=grp_pupil)
        add_field("pupil_roi_debug", "pupil_roi_debug", False, "check", parent=grp_pupil)
        add_field(
            "pupil_source",
            "pupil_source",
            "none",
            "combobox",
            ["auto", "labels", "naive", "swirski", "npz", "none"],
            parent=grp_pupil,
        )
        add_field("pupil_axis_mode", "pupil_axis_mode", "auto", "combobox", ["auto", "radius", "diameter"], parent=grp_pupil)

        grp_temporal = ttk.LabelFrame(ctrl, text="Temporal", padding=6)
        grp_temporal.pack(fill=tk.X, pady=(0, 8))
        _register_group("temporal", grp_temporal, {"fill": tk.X, "pady": (0, 8)})
        add_field("temporal", "temporal", False, "check", parent=grp_temporal)
        add_field("temporal_gate_px", "temporal_gate_px", 25.0, parent=grp_temporal)
        add_field("temporal_max_missed", "temporal_max_missed", 5, parent=grp_temporal)
        add_field("temporal_lambda", "temporal_lambda", 0.25, parent=grp_temporal)
        add_field("temporal_w_scale", "temporal_w_scale", 1.0, parent=grp_temporal)
        add_field("temporal_w_rot", "temporal_w_rot", 1.0, parent=grp_temporal)
        add_field("temporal_w_trans", "temporal_w_trans", 1.0, parent=grp_temporal)

        # Preview panel
        self.canvas = tk.Canvas(paned, bg="#111", width=900, height=700, highlightthickness=1, highlightbackground="#444")
        self.canvas.bind("<ButtonPress-1>", self.on_canvas_press)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)

        paned.add(ctrl_container, weight=0)
        paned.add(self.canvas, weight=1)

        # Correction panel (hidden unless enabled)
        self.corr_frame = ttk.Frame(ctrl, padding=(0, 4, 0, 0))
        mode_row = ttk.Frame(self.corr_frame)
        mode_row.pack(fill=tk.X, pady=(2, 2))
        ttk.Label(mode_row, text="Edit target").pack(side=tk.LEFT)
        mode_cb = ttk.Combobox(
            mode_row,
            textvariable=self.corr_target,
            values=["glint", "constellation"],
            width=14,
            state="readonly",
        )
        mode_cb.pack(side=tk.LEFT, padx=6)
        ttk.Checkbutton(self.corr_frame, text="Rotate constellation", variable=self.corr_rotate).pack(anchor=tk.W)
        ttk.Label(self.corr_frame, text="Detections (select then click image)").pack(anchor=tk.W, pady=(4, 2))
        self.det_list = tk.Listbox(self.corr_frame, selectmode=tk.SINGLE, width=32, height=8)
        self.det_list.pack(fill=tk.X, expand=False)
        btn_row = ttk.Frame(self.corr_frame)
        btn_row.pack(fill=tk.X, pady=4)
        ttk.Button(btn_row, text="Clear corrections", command=self.clear_corrections).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        self.corr_frame.pack_forget()

        # settings change -> rebuild cache (debounced)
        for var in self.vars.values():
            try:
                var.trace_add("write", self._on_settings_change)
            except Exception:
                pass
        try:
            self.mirror_var.trace_add("write", self._on_settings_change)
        except Exception:
            pass
        # dependent field visibility
        for k in (
            "score2_mode",
            "enhance_mode",
            "denoise",
            "clahe",
            "gamma_enable",
            "unsharp",
            "cand_fallback",
            "pupil_roi",
            "pupil_roi_pad_mode",
            "matcher",
            "temporal",
        ):
            var = self.vars.get(k)
            if var is None:
                continue
            try:
                var.trace_add("write", lambda *_a: self.root.after_idle(_update_dependent_visibility))
            except Exception:
                pass
        self.root.after_idle(_update_dependent_visibility)

    def _request_render(self, delay_ms: int = 0) -> None:
        if self.render_request_id is not None:
            try:
                self.root.after_cancel(self.render_request_id)
            except Exception:
                pass
        self.render_request_id = self.root.after(int(delay_ms), self._run_requested_render)

    def _run_requested_render(self) -> None:
        self.render_request_id = None
        self.render_current()

    def _set_canvas_rgb(self, rgb: np.ndarray) -> None:
        if self.zoom != 1.0:
            z = float(self.zoom)
            rgb = cv2.resize(
                rgb,
                (int(rgb.shape[1] * z), int(rgb.shape[0] * z)),
                interpolation=cv2.INTER_LINEAR,
            )
        img = Image.fromarray(rgb)
        self.photo = ImageTk.PhotoImage(img)
        if self.canvas_size != (img.width, img.height):
            self.canvas_size = (img.width, img.height)
            self.canvas.config(width=img.width, height=img.height, scrollregion=(0, 0, img.width, img.height))
        if self.canvas_image_id is None:
            self.canvas_image_id = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        else:
            self.canvas.itemconfig(self.canvas_image_id, image=self.photo)

    def _overrides_key(self, name: str):
        overrides = self.overrides_by_image.get(name)
        if not overrides:
            return None
        items = []
        for ti, (x, y) in overrides.items():
            try:
                items.append((int(ti), round(float(x), 3), round(float(y), 3)))
            except Exception:
                continue
        items.sort()
        return tuple(items) if items else None

    def _apply_overrides(self, name: str, cand_xy: np.ndarray, matches, T_hat):
        overrides = self.overrides_by_image.get(name, {})
        cand_xy_disp = cand_xy.copy() if isinstance(cand_xy, np.ndarray) else np.array(cand_xy, dtype=float)
        matches_disp = list(matches) if matches is not None else []
        if overrides and T_hat is not None:
            for ti, pt in overrides.items():
                x, y = pt
                cand_xy_disp = np.vstack([cand_xy_disp, np.array([[x, y]], dtype=float)])
                new_ci = len(cand_xy_disp) - 1
                d = float(np.linalg.norm(T_hat[int(ti)] - cand_xy_disp[new_ci]))
                replaced = False
                for mi, (tti, _, _) in enumerate(matches_disp):
                    if int(tti) == int(ti):
                        matches_disp[mi] = (int(ti), int(new_ci), d)
                        replaced = True
                        break
                if not replaced:
                    matches_disp.append((int(ti), int(new_ci), d))
        return cand_xy_disp, matches_disp

    def _frame_id(self, fp: Path) -> str:
        try:
            return str(fp.resolve())
        except Exception:
            return str(fp)

    def _preview_key(self, args, compute_key: str) -> str:
        return f"{compute_key}|preview={int(bool(getattr(args, 'preview_enhanced', False)))}"

    def _compute_key(self, args) -> str:
        payload = _compute_key_payload(args, self.templates_path, self.image_config_path, self.pupil_npz_path)
        return _hash_payload(payload)

    def _overlay_key(self, args, frame_name: str) -> str:
        overrides_key = self._overrides_key(frame_name)
        payload = _overlay_key_payload(args, overrides_key, bool(self.corr_mode.get()))
        return _hash_payload(payload)

    def _get_compute_entry(self, frame_id: str, compute_key: str):
        with self.cache_lock:
            return self.compute_cache.get(frame_id, {}).get(compute_key)

    def _set_compute_entry(self, frame_id: str, compute_key: str, entry: dict) -> None:
        with self.cache_lock:
            self.compute_cache.setdefault(frame_id, {})[compute_key] = entry

    def _overlay_cache_get(self, key):
        with self.cache_lock:
            if key in self.overlay_cache:
                self.overlay_cache.move_to_end(key)
                return self.overlay_cache[key]
        return None

    def _overlay_cache_set(self, key, value) -> None:
        with self.cache_lock:
            self.overlay_cache[key] = value
            self.overlay_cache.move_to_end(key)
            while len(self.overlay_cache) > int(self.overlay_cache_max):
                self.overlay_cache.popitem(last=False)

    def _ensure_current_preview_base(self, fp: Path, args, preview_key: str) -> np.ndarray:
        if (
            self.current_preview_base is not None
            and self.current_preview_base_name == fp.name
            and self.current_preview_base_cache_key == preview_key
        ):
            return self.current_preview_base
        bgr = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to read image: {fp}")
        if bool(getattr(args, "mirror", False)):
            bgr = cv2.flip(bgr, 1)
        gray_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        params = g.scale_params_for_image(args, w=gray_full.shape[1], h=gray_full.shape[0])
        preview_base = gray_full
        if getattr(args, "preview_enhanced", False):
            kernel_eff = int(params["kernel_eff"])
            median_ksize_eff = int(params["median_ksize_eff"])
            preview_base = g.enhance_for_glints(
                gray_full,
                kernel_size=kernel_eff,
                median_ksize=median_ksize_eff,
                clahe_clip=args.clahe_clip,
                clahe_tiles=args.clahe_tiles,
                denoise=int(getattr(args, "denoise", 1)),
                denoise_k=int(getattr(args, "denoise_k", 0)),
                clahe_enable=int(getattr(args, "clahe", 1)),
                gamma=float(getattr(args, "gamma", 1.0)),
                unsharp=int(getattr(args, "unsharp", 0)),
                unsharp_amount=float(getattr(args, "unsharp_amount", 1.0)),
                unsharp_sigma=float(getattr(args, "unsharp_sigma", 1.0)),
                enhance_mode=str(getattr(args, "enhance_mode", "tophat")),
                dog_sigma1=float(getattr(args, "dog_sigma1", 1.0)),
                dog_sigma2=float(getattr(args, "dog_sigma2", 2.2)),
                minmax=int(getattr(args, "minmax", 1)),
                enhance_enable=int(getattr(args, "enhance_enable", 1)),
            )
        self.current_preview_base = preview_base
        self.current_preview_base_name = fp.name
        self.current_preview_base_cache_key = preview_key
        return preview_base

    def _quick_preview_rgb(self, fp: Path, args, preview_key: str) -> np.ndarray | None:
        bgr = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        if bool(getattr(args, "mirror", False)):
            bgr = cv2.flip(bgr, 1)
        gray_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        self.current_preview_base = gray_full
        self.current_preview_base_name = fp.name
        self.current_preview_base_cache_key = preview_key
        return cv2.cvtColor(gray_full, cv2.COLOR_GRAY2RGB)

    def _start_ondemand_cache_compute(self, fp: Path, frame_idx: int, args, compute_key: str) -> None:
        frame_id = self._frame_id(fp)
        key = (compute_key, frame_id)
        with self.ondemand_lock:
            if len(self.ondemand_inflight) >= 2:
                return
            if key in self.ondemand_inflight:
                return
            self.ondemand_inflight.add(key)

        args_snapshot = types.SimpleNamespace(**vars(args))
        with self.cache_lock:
            token = self.cache_token

        def worker():
            try:
                template, bank_templates, ratio_index_bank, ratio_index_single, d_expected = self._prepare_templates_local(args_snapshot)
                pupil_npz_map = self.pupil_npz_map
                pupil_radii_raw = [int(x) for x in str(getattr(args_snapshot, "pupil_radii", "")).split(",") if x.strip().isdigit()]
                if not pupil_radii_raw:
                    pupil_radii_raw = [12, 16, 20, 24, 28, 32]
                cache_state = {
                    "template": template,
                    "bank_templates": bank_templates,
                    "ratio_index_bank": ratio_index_bank,
                    "ratio_index_single": ratio_index_single,
                    "d_expected": d_expected,
                    "prev_params": None,
                    "temporal_tracker": None,
                    "last_good_pupil": None,
                    "pupil_npz_map": pupil_npz_map,
                    "pupil_radii_raw": pupil_radii_raw,
                }
                entry = self._compute_cache_entry(fp, frame_idx, args_snapshot, cache_state)
                if entry is None:
                    return
                with self.cache_lock:
                    if token != self.cache_token or compute_key != self.compute_key:
                        return
                    self.compute_cache.setdefault(frame_id, {})[compute_key] = entry
            except Exception:
                _LOGGER.exception("On-demand cache compute failed for %s", frame_id)
            finally:
                with self.ondemand_lock:
                    self.ondemand_inflight.discard(key)
                try:
                    self.root.after(0, self._request_render)
                except Exception:
                    pass

        threading.Thread(target=worker, daemon=True).start()

    def _render_overlay_for_entry(
        self,
        fp: Path,
        frame_id: str,
        entry: dict,
        args,
        compute_key: str,
    ):
        overlay_key = self._overlay_key(args, fp.name)
        overlay_cache_key = (frame_id, compute_key, overlay_key)
        overlay_rgb = self._overlay_cache_get(overlay_cache_key)
        overlay_hit = overlay_rgb is not None

        status = entry.get("status", "ok")
        cand_xy_base = entry.get("cand_xy_base")
        matches_base = entry.get("matches_base")
        T_hat = entry.get("T_hat")
        tracked_xy = entry.get("tracked_xy")

        if status == "ok" and cand_xy_base is None:
            raise RuntimeError("Cache entry missing cand_xy_base")

        if status == "ok":
            cand_xy_disp, matches_disp = self._apply_overrides(fp.name, cand_xy_base, matches_base, T_hat)
        else:
            cand_xy_disp = np.empty((0, 2), dtype=float)
            matches_disp = []

        if overlay_rgb is None:
            preview_key = self._preview_key(args, compute_key)
            preview_base = self._ensure_current_preview_base(fp, args, preview_key)
            if status != "ok":
                overlay = cv2.cvtColor(preview_base, cv2.COLOR_GRAY2BGR)
                if bool(getattr(args, "show_overlay", True)):
                    cv2.putText(
                        overlay,
                        "SKIPPED (pupil ROI)",
                        (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 200, 255),
                        2,
                        cv2.LINE_AA,
                    )
                overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            else:
                title = (
                    f"{fp.name} | matcher={args.matcher} | inliers={entry.get('inliers', 0)} | "
                    f"err={entry.get('mean_err', float('nan')):.2f}px"
                )
                if args.temporal:
                    title += " | temporal"

                roi_rect = entry.get("roi_rect")
                use_temporal_display = bool(args.temporal) and not self.corr_mode.get()
                matches_draw = matches_disp
                if not args.show_overlay:
                    overlay = cv2.cvtColor(preview_base, cv2.COLOR_GRAY2BGR)
                else:
                    T_hat_draw = T_hat
                    if use_temporal_display and tracked_xy is not None:
                        if np.isfinite(tracked_xy).any():
                            T_hat_draw = tracked_xy
                            T_hat_draw = np.where(np.isfinite(T_hat_draw), T_hat_draw, -1e6)
                        matches_draw = []
                    overlay = g.draw_overlay(
                        preview_base,
                        cand_xy_disp,
                        T_hat_draw,
                        matches_draw,
                        title_text=title,
                        gt_xy=None,
                        match_tol=args.match_tol,
                    )
                if args.pupil_roi_debug and roi_rect is not None:
                    ox, oy, sz = roi_rect
                    cv2.rectangle(
                        overlay,
                        (int(round(ox)), int(round(oy))),
                        (int(round(ox + sz)), int(round(oy + sz))),
                        (0, 200, 255),
                        2,
                    )
                overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            self._overlay_cache_set(overlay_cache_key, overlay_rgb)

        return overlay_rgb, matches_disp, cand_xy_disp, overlay_hit

    def _get_args_and_compute_key(self):
        if not self.args_dirty and self.args_cached is not None and self.args_cached_key is not None:
            return self.args_cached, self.args_cached_key
        args = self._get_args()
        compute_key = self._compute_key(args)
        self.args_cached = args
        self.args_cached_key = compute_key
        self.args_dirty = False
        return args, compute_key

    def _get_args(self):
        # Build a Namespace compatible with g.run_matcher_for_template and helpers
        args = type("Args", (), {})()
        def getv(key, cast):
            v = self.vars[key].get()
            return cast(v)
        def getb(key):
            return bool(self.vars[key].get())

        args.matcher = self.vars["matcher"].get()
        args.template_mode = self.vars["template_mode"].get()
        args.score2_mode = self.vars["score2_mode"].get()
        args.ml_model_path = self.vars["ml_model_path"].get().strip() or None
        args.percentile = getv("percentile", float)
        args.kernel = getv("kernel", int)
        args.enhance_mode = self.vars["enhance_mode"].get()
        args.enhance_enable = 1 if getb("enhance_enable") else 0
        args.median_ksize = getv("median_ksize", int)
        args.denoise = 1 if getb("denoise") else 0
        args.denoise_k = getv("denoise_k", int)
        args.clahe = 1 if getb("clahe") else 0
        args.clahe_clip = getv("clahe_clip", float)
        args.clahe_tiles = getv("clahe_tiles", int)
        if getb("gamma_enable"):
            args.gamma = getv("gamma", float)
        else:
            args.gamma = 1.0
        args.unsharp = 1 if getb("unsharp") else 0
        args.unsharp_amount = getv("unsharp_amount", float)
        args.unsharp_sigma = getv("unsharp_sigma", float)
        args.minmax = 1 if getb("minmax") else 0
        args.preview_enhanced = getb("preview_enhanced")
        args.show_overlay = getb("show_overlay")
        args.clean_k = getv("clean_k", int)
        args.open_iter = getv("open_iter", int)
        args.close_iter = getv("close_iter", int)
        args.eps = getv("eps", float)
        args.max_pool = getv("max_pool", int)
        args.min_area = getv("min_area", int)
        args.max_area = getv("max_area", int)
        args.min_circ = getv("min_circ", float)
        args.min_maxI = 200
        args.cand_fallback = getb("cand_fallback")
        args.cand_target_raw = getv("cand_target_raw", int)
        args.cand_fallback_passes = getv("cand_fallback_passes", int)
        args.cand_fallback_percentiles = self.vars["cand_fallback_percentiles"].get()
        args.cand_fallback_kernel_add = 0
        args.cand_merge_eps = 2.0
        args.support_M = getv("support_M", int)
        args.support_tol = getv("support_tol", float)
        args.support_w = getv("support_w", float)
        args.contrast_r_inner = getv("contrast_r_inner", int)
        args.contrast_r_outer1 = getv("contrast_r_outer1", int)
        args.contrast_r_outer2 = getv("contrast_r_outer2", int)
        args.dog_sigma1 = getv("dog_sigma1", float)
        args.dog_sigma2 = getv("dog_sigma2", float)
        args.ratio_tol = getv("ratio_tol", float)
        args.pivot_P = getv("pivot_P", int)
        args.max_seeds = getv("max_seeds", int)
        grow_s = self.vars.get("grow_resid_max").get().strip() if "grow_resid_max" in self.vars else ""
        if not grow_s:
            args.grow_resid_max = None
        else:
            try:
                args.grow_resid_max = float(grow_s)
            except Exception:
                args.grow_resid_max = None
        args.layout_prior = getb("layout_prior")
        args.layout_lambda = getv("layout_lambda", float)
        args.layout_mode = "image"
        args.layout_debug = False
        args.sla_layout_prior = getb("sla_layout_prior")
        args.sla_layout_lambda = getv("sla_layout_lambda", float)
        args.sla_layout_mode = "image"
        args.sla_layout_debug = False
        args.sla_semantic_prior = getb("sla_semantic_prior")
        args.sla_semantic_mode = self.vars.get("sla_semantic_mode", tk.StringVar(value="full")).get()
        args.sla_semantic_lambda = getv("sla_semantic_lambda", float)
        args.sla_semantic_hard = False
        args.sla_mirror_reject = True
        args.sla_top2_margin = 0.0
        args.sla_base_ratio_min = 0.80
        args.sla_side_margin = 0.0
        args.sla_semantic_debug = False
        args.sla_scale_min = 0.01
        args.sla_scale_max = 500.0
        args.sla_g0_top2 = False
        args.sla_w_seed_score2 = 1.0
        args.sla_w_seed_geom = 1.0
        args.max_seeds_per_pivot = 80
        args.sla_adaptive_ratio_tol = True
        args.sla_ratio_tol_min = 0.06
        args.sla_ratio_tol_refN = 12
        args.temporal = getb("temporal")
        args.temporal_prior = args.temporal
        args.temporal_gate_px = getv("temporal_gate_px", float)
        args.temporal_max_missed = getv("temporal_max_missed", int)
        args.temporal_lambda = getv("temporal_lambda", float)
        args.temporal_w_scale = getv("temporal_w_scale", float)
        args.temporal_w_rot = getv("temporal_w_rot", float)
        args.temporal_w_trans = getv("temporal_w_trans", float)
        args.temporal_use_tracks_for_matching = True
        args.temporal_roi_radius = 0.0
        args.vote_M = getv("vote_M", int)
        args.vote_ratio_tol = getv("vote_ratio_tol", float)
        args.vote_max_hyp = getv("vote_max_hyp", int)
        args.vote_w_score2 = getv("vote_w_score2", float)
        args.min_k = getv("min_k", int)
        args.iters = getv("iters", int)
        args.seed = getv("seed", int)
        args.scale_min = getv("scale_min", float)
        args.scale_max = getv("scale_max", float)
        args.disable_scale_gate = getb("disable_scale_gate")
        args.matching = self.vars.get("matching", tk.StringVar(value="greedy")).get()
        args.appearance_tiebreak = getb("appearance_tiebreak")
        args.roi_mode = "none"
        args.roi_border_frac = 0.06
        args.roi_border_px = None
        args.pupil_roi = getb("pupil_roi")
        args.pupil_roi_size = getv("pupil_roi_size", int)
        args.pupil_roi_pad_mode = self.vars["pupil_roi_pad_mode"].get()
        args.pupil_roi_pad_value = getv("pupil_roi_pad_value", int)
        args.pupil_roi_fail_policy = self.vars["pupil_roi_fail_policy"].get()
        args.pupil_roi_debug = getb("pupil_roi_debug")
        args.pupil_source = self.vars["pupil_source"].get()
        args.pupil_axis_mode = self.vars["pupil_axis_mode"].get()
        args.pupil_dark_thresh = 60
        args.pupil_min_area = 150
        args.pupil_rmin = 0.3
        args.pupil_rmax = 1.2
        args.pupil_fallback_center = "image"
        args.pupil_method = "naive"
        args.pupil_radii = "12,16,20,24,28,32"
        args.pupil_sigma_frac = 0.35
        args.pupil_fail_open = True
        args.pupil_force_gate = False
        args.debug_pupil = False
        args.pupil_npz = self.pupil_npz_path
        args.auto_scale = True
        args.ref_width = 640
        args.min_kernel = 3
        args.min_inliers = getv("min_inliers", int)
        args.post_id_resolve = False
        args.template_bank_source = "default"
        args.template_bank_path = None
        args.bank_select_metric = "strict"
        args.template_build_mode = "procrustes"
        args.verbose_template = False
        args.match_tol = getv("match_tol", float)
        args.mirror = bool(self.mirror_var.get())

        # overrides from loaded configs
        if self.templates_path:
            args.template_bank_source = "custom"
            args.template_bank_path = self.templates_path
        if self.image_config_path:
            args.image_config = self.image_config_path
            g._apply_image_config(args, self.image_config_path)
        if args.pupil_npz:
            args.pupil_roi = True
            args.pupil_source = "npz"
            args.pupil_roi_fail_policy = "full_frame"
        else:
            args.pupil_roi = False

        return args

    def _on_settings_change(self, *_args) -> None:
        self.args_dirty = True
        self._request_render(0)
        self._schedule_cache_rebuild()

    def _schedule_cache_rebuild(self) -> None:
        if not self.files or self.bulk_processing:
            return
        if self.cache_debounce_id is not None:
            try:
                self.root.after_cancel(self.cache_debounce_id)
            except Exception:
                pass
        self.cache_debounce_id = self.root.after(400, self._start_cache_rebuild)

    def _start_cache_rebuild(self) -> None:
        if not self.files or self.bulk_processing:
            return
        args_snapshot = self._get_args()
        compute_key = self._compute_key(args_snapshot)
        with self.cache_lock:
            if compute_key == self.compute_key and self.compute_cache:
                return
            self.compute_key = compute_key
            self.compute_cache = {}
            self.overlay_cache.clear()
            self.cache_token += 1
            token = self.cache_token
            self.cache_building = True
        self.status_var.set("Precomputing cache (0%)...")
        self.cache_progress_var.set(0.0)
        files_snapshot = list(self.files)

        def worker():
            try:
                template, bank_templates, ratio_index_bank, ratio_index_single, d_expected = self._prepare_templates_local(args_snapshot)
                pupil_npz_map = None
                if getattr(args_snapshot, "pupil_npz", None):
                    try:
                        pupil_npz_map = g.load_pupil_npz(args_snapshot.pupil_npz)
                    except Exception:
                        pupil_npz_map = None
                pupil_radii_raw = [int(x) for x in str(args_snapshot.pupil_radii).split(",") if x.strip().isdigit()]
                if not pupil_radii_raw:
                    pupil_radii_raw = [12, 16, 20, 24, 28, 32]
                cache_state = {
                    "template": template,
                    "bank_templates": bank_templates,
                    "ratio_index_bank": ratio_index_bank,
                    "ratio_index_single": ratio_index_single,
                    "d_expected": d_expected,
                    "prev_params": None,
                    "temporal_tracker": None,
                    "last_good_pupil": None,
                    "pupil_npz_map": pupil_npz_map,
                    "pupil_radii_raw": pupil_radii_raw,
                }
                total = len(files_snapshot)
                for i, fp in enumerate(files_snapshot):
                    with self.cache_lock:
                        if token != self.cache_token or compute_key != self.compute_key:
                            return
                    frame_id = self._frame_id(fp)
                    try:
                        entry = self._compute_cache_entry(fp, i, args_snapshot, cache_state)
                    except Exception:
                        _LOGGER.exception("Cache compute failed for %s", frame_id)
                        continue
                    if entry is not None:
                        with self.cache_lock:
                            if token != self.cache_token or compute_key != self.compute_key:
                                return
                            self.compute_cache.setdefault(frame_id, {})[compute_key] = entry
                    if (i + 1) % 2 == 0:
                        time.sleep(0)
                    if (i + 1) % 10 == 0 or i + 1 == total:
                        pct = int(round((i + 1) * 100.0 / max(total, 1)))
                        self.root.after(0, lambda p=pct, a=i + 1, t=total: self.status_var.set(
                            f"Precomputing cache ({p}%)... {a}/{t}"
                        ))
                        self.root.after(0, lambda p=pct: self.cache_progress_var.set(float(p)))
            finally:
                with self.cache_lock:
                    if token == self.cache_token:
                        self.cache_building = False
                if token == self.cache_token:
                    self.root.after(0, lambda: self.cache_progress_var.set(100.0))
                    self.root.after(0, self._request_render)

        self.cache_thread = threading.Thread(target=worker, daemon=True)
        self.cache_thread.start()

    def _prepare_templates_local(self, args):
        if args.template_bank_source == "custom":
            bank = g.load_template_bank(args)
            P_list = bank
        else:
            P_list = g.load_default_template_bank()
        if args.template_build_mode == "median":
            template = g.build_template_median(P_list)
        else:
            template, _ = g.build_template_from_labeled_sets(P_list, iters=10, tol=1e-6, verbose=False)
        bank_templates = None
        if args.template_mode == "bank":
            bank_templates = P_list
        ratio_index_bank = None
        ratio_index_single = None
        if args.matcher == "sla":
            if bank_templates is not None:
                ratio_index_bank = [g.build_ratio_index(t) for t in bank_templates]
            else:
                ratio_index_single = g.build_ratio_index(template)
        d_expected = None
        if args.score2_mode == "contrast_support":
            bank_for_support = bank_templates if bank_templates is not None else [template]
            d_expected = g.compute_expected_pairwise_distances(bank_for_support)
        return template, bank_templates, ratio_index_bank, ratio_index_single, d_expected

    def _compute_cache_entry(self, fp: Path, frame_idx: int, args, cache_state: dict) -> dict | None:
        bgr = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        if bool(getattr(args, "mirror", False)):
            bgr = cv2.flip(bgr, 1)
        gray_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        params = g.scale_params_for_image(args, w=gray_full.shape[1], h=gray_full.shape[0])
        H, W = gray_full.shape[:2]
        image_center = (0.5 * W, 0.5 * H)
        pupil_radii_raw = cache_state["pupil_radii_raw"]

        pcx = pcy = pr = None
        pupil_detected = False
        pupil_ok = False
        pupil_source_used = "none"
        if args.pupil_roi and args.pupil_source != "none":
            pcx, pcy, pr, pupil_detected, pupil_ok, pupil_source_used = g.detect_pupil_center_for_frame(
                gray_full, None, fp.name, W, H, args, params, pupil_radii_raw, cache_state["pupil_npz_map"]
            )

        roi_active = False
        roi_info = None
        roi_decision = None
        gray = gray_full
        if args.pupil_roi:
            roi_decision = resolve_pupil_roi_center(
                (pcx, pcy) if pcx is not None and pcy is not None else None,
                pr,
                W,
                H,
                args.pupil_roi_fail_policy,
                cache_state["last_good_pupil"],
            )
            if roi_decision.action == "skip":
                return {
                    "status": "skipped",
                    "reason": "pupil_roi_skip",
                    "cand_xy_base": np.empty((0, 2), dtype=float),
                    "matches_base": [],
                    "T_hat": None,
                    "tracked_xy": None,
                    "inliers": 0,
                    "mean_err": float("nan"),
                    "roi_rect": None,
                }
            if roi_decision.action == "use" and roi_decision.center is not None:
                roi_info = compute_pupil_roi(
                    gray_full,
                    roi_decision.center,
                    size=int(args.pupil_roi_size),
                    pad_mode=str(args.pupil_roi_pad_mode),
                    pad_value=int(args.pupil_roi_pad_value),
                )
                gray = roi_info.roi_img
                roi_active = True
                if roi_decision.center is not None:
                    cache_state["last_good_pupil"] = (roi_decision.center[0], roi_decision.center[1], pr)

        cand_xy_pass0, rows_pass0, cand_score2_pass0, cand_raw_pass0, cand_support_pass0 = g.detect_candidates_one_pass(
            gray, params, args, d_expected=cache_state["d_expected"]
        )
        cand_xy_raw = cand_xy_pass0
        cand_score2_raw = cand_score2_pass0
        cand_support_raw = cand_support_pass0
        cand_raw_merged = int(cand_raw_pass0)

        if args.cand_fallback and cand_raw_pass0 < args.cand_target_raw:
            try:
                fallback_percentiles = [float(x) for x in args.cand_fallback_percentiles.split(",") if x.strip()]
            except Exception:
                fallback_percentiles = []
            if not fallback_percentiles:
                fallback_percentiles = [args.percentile]

            cand_xy_list = [cand_xy_pass0]
            cand_score2_list = [cand_score2_pass0]
            cand_support_list = [cand_support_pass0]
            for i in range(1, int(args.cand_fallback_passes) + 1):
                idx = min(i - 1, len(fallback_percentiles) - 1)
                perc = fallback_percentiles[idx]
                kernel_add = int(args.cand_fallback_kernel_add) * i
                cand_xy_i, rows_i, cand_score2_i, cand_raw_i, cand_support_i = g.detect_candidates_one_pass(
                    gray, params, args, d_expected=cache_state["d_expected"], percentile_override=perc, kernel_add=kernel_add
                )
                cand_xy_list.append(cand_xy_i)
                cand_score2_list.append(cand_score2_i)
                cand_support_list.append(cand_support_i)
                cand_xy_raw, cand_score2_raw, cand_support_raw = g.merge_candidates(
                    cand_xy_list, cand_score2_list, cand_support_list, merge_eps=float(args.cand_merge_eps)
                )
                cand_raw_merged = int(len(cand_xy_raw))
                if cand_raw_merged >= args.cand_target_raw:
                    break

        if roi_active and roi_info is not None:
            cand_xy_raw = map_points_to_full(cand_xy_raw, roi_info.offset_x, roi_info.offset_y)
            if cand_xy_raw.size > 0:
                in_bounds = (
                    (cand_xy_raw[:, 0] >= 0)
                    & (cand_xy_raw[:, 0] < W)
                    & (cand_xy_raw[:, 1] >= 0)
                    & (cand_xy_raw[:, 1] < H)
                )
                cand_xy_raw = cand_xy_raw[in_bounds]
                cand_score2_raw = cand_score2_raw[in_bounds]
                cand_support_raw = cand_support_raw[in_bounds]
            cand_raw_merged = int(len(cand_xy_raw))

        rows_sorted = sorted(
            zip(cand_xy_raw, cand_score2_raw),
            key=lambda p: p[1],
            reverse=True,
        )[: args.max_pool]
        cand_xy = np.array([p[0] for p in rows_sorted], dtype=float) if rows_sorted else np.empty((0, 2), dtype=float)
        cand_score2 = np.array([p[1] for p in rows_sorted], dtype=float) if rows_sorted else np.empty((0,), dtype=float)

        roi_shape = gray_full.shape if roi_active else gray.shape
        cand_xy, cand_score2, roi_margin = g.filter_candidates_roi(
            cand_xy, cand_score2, roi_shape, mode=args.roi_mode,
            border_frac=args.roi_border_frac, border_px=args.roi_border_px
        )
        roi_center = image_center
        roi_radius_estimate = max(1.0, 0.5 * (min(W, H) - 2 * roi_margin))
        pcx_gate = pcx
        pcy_gate = pcy
        pr_gate = pr
        if roi_decision is not None and roi_decision.action == "use" and roi_decision.center is not None:
            pcx_gate, pcy_gate = roi_decision.center
            if roi_decision.radius is not None and np.isfinite(roi_decision.radius):
                pr_gate = roi_decision.radius
        if args.pupil_roi and len(cand_xy) >= 2 and args.pupil_source != "none":
            if not pupil_ok:
                if args.pupil_fail_open:
                    pcx_gate = pcy_gate = pr_gate = None
                else:
                    if args.pupil_fallback_center == "image":
                        pcx_gate, pcy_gate = image_center
                        pr_gate = 0.25 * min(W, H)
                    else:
                        pcx_gate, pcy_gate = roi_center
                        pr_gate = roi_radius_estimate
            if pcx_gate is not None and pcy_gate is not None and pr_gate is not None:
                pupil_mask = g.gate_candidates_by_pupil(
                    cand_xy, (pcx_gate, pcy_gate), pr_gate, args.pupil_rmin, args.pupil_rmax
                )
                mask_sum = int(pupil_mask.sum())
                if mask_sum >= args.min_k or args.pupil_force_gate:
                    cand_xy = cand_xy[pupil_mask]
                    cand_score2 = cand_score2[pupil_mask]

        best = None
        best_key = None
        chosen_template_idx = None
        T_hat = None
        matches = None
        inliers = 0
        mean_err = float("nan")
        app_sum = float("nan")
        s_fit = float("nan")
        best_template_xy = cache_state["template"]
        if len(cand_xy) >= args.min_k:
            if args.template_mode == "single":
                best = g.run_matcher_for_template(
                    cache_state["template"], cand_xy, cand_score2, args, None, params["eps_eff"], cache_state["ratio_index_single"],
                    cand_raw_count=cand_raw_merged
                )
                best_template_xy = cache_state["template"]
            else:
                bank = cache_state["bank_templates"] if cache_state["bank_templates"] is not None else g.load_template_bank(args)
                for bi, template_xy in enumerate(bank):
                    ratio_idx = None
                    if cache_state["ratio_index_bank"] is not None and bi < len(cache_state["ratio_index_bank"]):
                        ratio_idx = cache_state["ratio_index_bank"][bi]
                    res = g.run_matcher_for_template(
                        template_xy, cand_xy, cand_score2, args, None, params["eps_eff"], ratio_idx,
                        cand_raw_count=cand_raw_merged
                    )
                    if res is None:
                        continue
                    if args.temporal and cache_state["prev_params"] is not None:
                        key = g.score_match_result_temporal(
                            template_xy, res, cand_score2, args.appearance_tiebreak,
                            bank_select_metric=args.bank_select_metric, prev_params=cache_state["prev_params"], args=args
                        )
                    else:
                        key = g.score_match_result(
                            res, cand_score2, args.appearance_tiebreak, args.bank_select_metric, s_expected=None
                        )
                    if best is None or key > best_key:
                        best = res
                        best_key = key
                        chosen_template_idx = bi
                        best_template_xy = template_xy

        if best is not None:
            s_fit, _, _, matches, T_hat, app_sum = best
            inliers = len(matches) if matches else 0
            mean_err = float(np.mean([m[2] for m in matches])) if matches else float("nan")
            if args.temporal:
                cache_state["prev_params"] = g.extract_similarity_params(best_template_xy, best)

        tracked_xy = None
        if args.temporal:
            if cache_state["temporal_tracker"] is None:
                cache_state["temporal_tracker"] = MultiGlintTracker(
                    n_tracks=4,
                    gate_px=args.temporal_gate_px,
                    max_missed=args.temporal_max_missed,
                )
            meas_labeled = np.full((4, 2), np.nan, dtype=float)
            for ti, ci, _ in (matches or []):
                if int(ti) >= 4:
                    continue
                meas_labeled[int(ti)] = cand_xy[int(ci)]
            if not np.isfinite(meas_labeled).any() and len(cand_xy) > 0:
                k = min(4, len(cand_xy))
                meas_labeled[:k] = cand_xy[:k]
            if not np.isfinite(meas_labeled).any() and T_hat is not None and T_hat.shape[0] >= 4:
                meas_labeled[:4] = T_hat[:4]
            tracked_xy, _meta = cache_state["temporal_tracker"].step_labeled(meas_labeled, frame_idx)

        roi_rect = None
        if roi_info is not None:
            roi_rect = (float(roi_info.offset_x), float(roi_info.offset_y), float(int(args.pupil_roi_size)))

        return {
            "status": "ok",
            "cand_xy_base": cand_xy,
            "matches_base": matches,
            "T_hat": T_hat,
            "tracked_xy": tracked_xy,
            "inliers": inliers,
            "mean_err": mean_err,
            "roi_rect": roi_rect,
        }

    def _invalidate_cache_for_frame(self, frame_id: str) -> None:
        with self.cache_lock:
            self.compute_cache.pop(frame_id, None)
            if self.overlay_cache:
                stale = [key for key in self.overlay_cache.keys() if key[0] == frame_id]
                for key in stale:
                    self.overlay_cache.pop(key, None)

    def load_templates_json(self) -> None:
        path = filedialog.askopenfilename(
            title="Select templates JSON",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        self.templates_path = path
        self.args_dirty = True
        self.cfg_var.set(f"templates: {Path(path).name}")
        self._start_template_prep()

    def load_image_config(self) -> None:
        path = filedialog.askopenfilename(
            title="Select image config JSON",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        self.image_config_path = path
        self.args_dirty = True
        self.cfg_var.set(f"image_config: {Path(path).name}")
        self._request_render()
        self._schedule_cache_rebuild()

    def load_pupil_npz(self) -> None:
        path = filedialog.askopenfilename(
            title="Select pupil NPZ",
            filetypes=[("NPZ", "*.npz"), ("All files", "*.*")],
        )
        if not path:
            return
        self.status_var.set("Loading pupil NPZ...")
        self.root.update_idletasks()

        def worker():
            try:
                pupil_map = g.load_pupil_npz(path)
            except Exception as exc:
                self.root.after(0, lambda exc=exc: messagebox.showerror("Error", f"Failed to load pupil NPZ: {exc}"))
                return
            def apply_loaded():
                self.pupil_npz_map = pupil_map
                self.pupil_npz_path = path
                self.args_dirty = True
                # Avoid UI "freeze" if filenames don't match: fall back to full frame.
                if "pupil_roi_fail_policy" in self.vars:
                    self.vars["pupil_roi_fail_policy"].set("full_frame")
                self.cfg_var.set(f"pupil_npz: {Path(path).name}")
                self._request_render()
                self._schedule_cache_rebuild()
            self.root.after(0, apply_loaded)

        threading.Thread(target=worker, daemon=True).start()

    def save_settings(self) -> None:
        out_path = filedialog.asksaveasfilename(
            title="Save UI settings",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
            initialfile="preview_ui_settings.json",
        )
        if not out_path:
            return
        data = {
            "vars": {k: v.get() for k, v in self.vars.items()},
            "templates_path": self.templates_path,
            "image_config_path": self.image_config_path,
            "pupil_npz_path": self.pupil_npz_path,
        }
        Path(out_path).write_text(json.dumps(data, indent=2), encoding="utf-8")
        self.status_var.set(f"Saved settings: {Path(out_path).name}")

    def load_settings(self) -> None:
        in_path = filedialog.askopenfilename(
            title="Load UI settings",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not in_path:
            return
        try:
            data = json.loads(Path(in_path).read_text(encoding="utf-8"))
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to load settings: {exc}")
            return
        vars_data = data.get("vars", {})
        for k, v in vars_data.items():
            if k in self.vars:
                try:
                    self.vars[k].set(v)
                except Exception:
                    pass
        self.templates_path = data.get("templates_path")
        self.image_config_path = data.get("image_config_path")
        self.pupil_npz_path = data.get("pupil_npz_path")
        self.args_dirty = True
        if self.pupil_npz_path:
            try:
                self.pupil_npz_map = g.load_pupil_npz(self.pupil_npz_path)
            except Exception as exc:
                messagebox.showerror("Error", f"Failed to load pupil NPZ: {exc}")
        self.cfg_var.set(f"settings: {Path(in_path).name}")
        self._request_render()
        self._schedule_cache_rebuild()

    def load_folder(self) -> None:
        path = filedialog.askdirectory(title="Select image folder")
        if not path:
            return
        if self.zip_temp_dir is not None and self.zip_temp_dir.exists():
            try:
                shutil.rmtree(self.zip_temp_dir)
            except Exception:
                pass
            self.zip_temp_dir = None
        self.folder = Path(path)
        files = g.iter_images(self.folder)
        self.files = sorted(files, key=self._natural_key)
        if not self.files:
            messagebox.showerror("Error", "No images found in folder")
            return
        self.idx = 0
        self._reset_temporal()
        self._start_template_prep()

    def load_zip(self) -> None:
        path = filedialog.askopenfilename(
            title="Select ZIP with images",
            filetypes=[("ZIP", "*.zip"), ("All files", "*.*")],
        )
        if not path:
            return
        zip_path = Path(path)
        if not zip_path.exists():
            messagebox.showerror("Error", f"ZIP not found: {zip_path}")
            return
        if self.zip_temp_dir is not None and self.zip_temp_dir.exists():
            try:
                shutil.rmtree(self.zip_temp_dir)
            except Exception:
                pass
        temp_root = Path(tempfile.mkdtemp(prefix="glint_zip_"))
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                members = [m for m in zf.infolist() if not m.is_dir()]
                img_members = [m for m in members if Path(m.filename).suffix.lower() in exts]
                if not img_members:
                    messagebox.showerror("Error", "No images found in ZIP")
                    shutil.rmtree(temp_root, ignore_errors=True)
                    return
                for idx, m in enumerate(img_members):
                    name = Path(m.filename).name
                    stem = f"{idx:06d}_{name}"
                    out_path = temp_root / stem
                    with zf.open(m, "r") as src, open(out_path, "wb") as dst:
                        shutil.copyfileobj(src, dst)
        except Exception as exc:
            shutil.rmtree(temp_root, ignore_errors=True)
            messagebox.showerror("Error", f"Failed to read ZIP: {exc}")
            return

        self.zip_temp_dir = temp_root
        self.folder = temp_root
        files = g.iter_images(self.folder)
        self.files = sorted(files, key=self._natural_key)
        if not self.files:
            messagebox.showerror("Error", "No images extracted from ZIP")
            return
        self.idx = 0
        self._reset_temporal()
        self._start_template_prep()

    def _start_template_prep(self) -> None:
        self.status_var.set("Preparing templates...")
        self.root.update_idletasks()

        def worker():
            try:
                self._prepare_templates()
            except Exception as exc:
                self.root.after(0, lambda exc=exc: messagebox.showerror("Error", f"Template prep failed: {exc}"))
                return
            self.root.after(0, self._request_render)
            self.root.after(0, self._schedule_cache_rebuild)

        threading.Thread(target=worker, daemon=True).start()

    @staticmethod
    def _natural_key(p: Path):
        name = p.stem
        nums = [int(x) for x in re.findall(r"\d+", name)]
        if nums:
            return (nums, name.lower(), p.suffix.lower())
        return ([], name.lower(), p.suffix.lower())

    def _reset_temporal(self) -> None:
        self.temporal_tracker = None
        self.prev_params = None
        self.last_idx = None
        self.last_good_pupil = None

    def _prepare_templates(self, args=None) -> None:
        if args is None:
            args = self._get_args()
        if args.template_bank_source == "custom":
            bank = g.load_template_bank(args)
            P_list = bank
        else:
            P_list = g.load_default_template_bank()
        if args.template_build_mode == "median":
            template = g.build_template_median(P_list)
        else:
            template, _ = g.build_template_from_labeled_sets(P_list, iters=10, tol=1e-6, verbose=False)
        self.template = template
        self.bank_templates = None
        if args.template_mode == "bank":
            self.bank_templates = P_list
        self.ratio_index_bank = None
        self.ratio_index_single = None
        if args.matcher == "sla":
            if self.bank_templates is not None:
                self.ratio_index_bank = [g.build_ratio_index(t) for t in self.bank_templates]
            else:
                self.ratio_index_single = g.build_ratio_index(self.template)
        self.d_expected = None
        if args.score2_mode == "contrast_support":
            bank_for_support = self.bank_templates if self.bank_templates is not None else [self.template]
            self.d_expected = g.compute_expected_pairwise_distances(bank_for_support)

    def _process_frame(self, fp: Path, frame_idx: int, update_tracker: bool, draw: bool, args_override=None) -> None:
        bgr = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if bgr is None:
            return
        args = args_override if args_override is not None else self._get_args()
        frame_id = self._frame_id(fp)
        compute_key = self._compute_key(args)
        if bool(getattr(args, "mirror", False)):
            bgr = cv2.flip(bgr, 1)
        gray_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        params = g.scale_params_for_image(args, w=gray_full.shape[1], h=gray_full.shape[0])
        H, W = gray_full.shape[:2]
        image_center = (0.5 * W, 0.5 * H)
        pupil_radii_raw = [int(x) for x in args.pupil_radii.split(",") if x.strip().isdigit()]
        if not pupil_radii_raw:
            pupil_radii_raw = [12, 16, 20, 24, 28, 32]

        pcx = pcy = pr = None
        pupil_detected = False
        pupil_ok = False
        pupil_source_used = "none"
        if args.pupil_roi and args.pupil_source != "none":
            pcx, pcy, pr, pupil_detected, pupil_ok, pupil_source_used = g.detect_pupil_center_for_frame(
                gray_full, None, fp.name, W, H, args, params, pupil_radii_raw, self.pupil_npz_map
            )

        roi_active = False
        roi_info = None
        roi_decision = None
        gray = gray_full
        if args.pupil_roi:
            roi_decision = resolve_pupil_roi_center(
                (pcx, pcy) if pcx is not None and pcy is not None else None,
                pr,
                W,
                H,
                args.pupil_roi_fail_policy,
                self.last_good_pupil,
            )
            if roi_decision.action == "skip":
                entry = {
                    "status": "skipped",
                    "reason": "pupil_roi_skip",
                    "cand_xy_base": np.empty((0, 2), dtype=float),
                    "matches_base": [],
                    "T_hat": None,
                    "tracked_xy": None,
                    "inliers": 0,
                    "mean_err": float("nan"),
                    "roi_rect": None,
                }
                with self.cache_lock:
                    if compute_key == self.compute_key:
                        self.compute_cache.setdefault(frame_id, {})[compute_key] = entry
                if draw:
                    overlay_rgb, matches_disp, cand_xy_disp, _overlay_hit = self._render_overlay_for_entry(
                        fp, frame_id, entry, args, compute_key
                    )
                    self._set_canvas_rgb(overlay_rgb)
                    self.status_var.set(f"{self.idx+1}/{len(self.files)}: {fp.name} (skipped)")
                    self.current_fp = fp
                    self.current_matches = matches_disp
                    self.current_cand_xy = cand_xy_disp
                    self._refresh_detection_list()
                return
            if roi_decision.action == "use" and roi_decision.center is not None:
                roi_info = compute_pupil_roi(
                    gray_full,
                    roi_decision.center,
                    size=int(args.pupil_roi_size),
                    pad_mode=str(args.pupil_roi_pad_mode),
                    pad_value=int(args.pupil_roi_pad_value),
                )
                gray = roi_info.roi_img
                roi_active = True
                if roi_decision.radius is not None:
                    self.last_good_pupil = (roi_decision.center[0], roi_decision.center[1], roi_decision.radius)
                else:
                    self.last_good_pupil = (roi_decision.center[0], roi_decision.center[1], float("nan"))

        cand_xy_pass0, rows_pass0, cand_score2_pass0, cand_raw_pass0, cand_support_pass0 = g.detect_candidates_one_pass(
            gray, params, args, d_expected=self.d_expected
        )
        cand_xy_raw = cand_xy_pass0
        cand_score2_raw = cand_score2_pass0
        cand_support_raw = cand_support_pass0
        cand_raw_merged = int(cand_raw_pass0)

        if args.cand_fallback and cand_raw_pass0 < args.cand_target_raw:
            try:
                fallback_percentiles = [float(x) for x in args.cand_fallback_percentiles.split(",") if x.strip()]
            except Exception:
                fallback_percentiles = []
            if not fallback_percentiles:
                fallback_percentiles = [args.percentile]

            cand_xy_list = [cand_xy_pass0]
            cand_score2_list = [cand_score2_pass0]
            cand_support_list = [cand_support_pass0]
            for i in range(1, int(args.cand_fallback_passes) + 1):
                idx = min(i - 1, len(fallback_percentiles) - 1)
                perc = fallback_percentiles[idx]
                kernel_add = int(args.cand_fallback_kernel_add) * i
                cand_xy_i, rows_i, cand_score2_i, cand_raw_i, cand_support_i = g.detect_candidates_one_pass(
                    gray, params, args, d_expected=self.d_expected, percentile_override=perc, kernel_add=kernel_add
                )
                cand_xy_list.append(cand_xy_i)
                cand_score2_list.append(cand_score2_i)
                cand_support_list.append(cand_support_i)
                cand_xy_raw, cand_score2_raw, cand_support_raw = g.merge_candidates(
                    cand_xy_list, cand_score2_list, cand_support_list, merge_eps=float(args.cand_merge_eps)
                )
                cand_raw_merged = int(len(cand_xy_raw))
                if cand_raw_merged >= args.cand_target_raw:
                    break

        if roi_active and roi_info is not None:
            cand_xy_raw = map_points_to_full(cand_xy_raw, roi_info.offset_x, roi_info.offset_y)
            if cand_xy_raw.size > 0:
                in_bounds = (
                    (cand_xy_raw[:, 0] >= 0)
                    & (cand_xy_raw[:, 0] < W)
                    & (cand_xy_raw[:, 1] >= 0)
                    & (cand_xy_raw[:, 1] < H)
                )
                cand_xy_raw = cand_xy_raw[in_bounds]
                cand_score2_raw = cand_score2_raw[in_bounds]
                cand_support_raw = cand_support_raw[in_bounds]
            cand_raw_merged = int(len(cand_xy_raw))

        # pool
        rows_sorted = sorted(
            zip(cand_xy_raw, cand_score2_raw),
            key=lambda p: p[1],
            reverse=True,
        )[: args.max_pool]
        cand_xy = np.array([p[0] for p in rows_sorted], dtype=float) if rows_sorted else np.empty((0, 2), dtype=float)
        cand_score2 = np.array([p[1] for p in rows_sorted], dtype=float) if rows_sorted else np.empty((0,), dtype=float)

        # match
        best = None
        best_key = None
        chosen_template_idx = None
        T_hat = None
        matches = None
        inliers = 0
        mean_err = float("nan")
        app_sum = float("nan")
        s_fit = float("nan")
        best_template_xy = self.template

        if len(cand_xy) >= args.min_k:
            if args.template_mode == "single":
                best = g.run_matcher_for_template(
                    self.template, cand_xy, cand_score2, args, None, params["eps_eff"], self.ratio_index_single,
                    cand_raw_count=cand_raw_merged
                )
                best_template_xy = self.template
            else:
                bank = self.bank_templates if self.bank_templates is not None else g.load_template_bank(args)
                for bi, template_xy in enumerate(bank):
                    ratio_idx = None
                    if self.ratio_index_bank is not None and bi < len(self.ratio_index_bank):
                        ratio_idx = self.ratio_index_bank[bi]
                    res = g.run_matcher_for_template(
                        template_xy, cand_xy, cand_score2, args, None, params["eps_eff"], ratio_idx,
                        cand_raw_count=cand_raw_merged
                    )
                    if res is None:
                        continue
                    if args.temporal and self.prev_params is not None:
                        key = g.score_match_result_temporal(
                            template_xy, res, cand_score2, args.appearance_tiebreak,
                            bank_select_metric=args.bank_select_metric, prev_params=self.prev_params, args=args
                        )
                    else:
                        key = g.score_match_result(
                            res, cand_score2, args.appearance_tiebreak, args.bank_select_metric, s_expected=None
                        )
                    if best is None or key > best_key:
                        best = res
                        best_key = key
                        chosen_template_idx = bi
                        best_template_xy = template_xy

        if best is not None:
            s_fit, _, _, matches, T_hat, app_sum = best
            inliers = len(matches) if matches else 0
            mean_err = float(np.mean([m[2] for m in matches])) if matches else float("nan")
            if args.temporal:
                self.prev_params = g.extract_similarity_params(best_template_xy, best)

        if T_hat is not None:
            self.template_by_image[fp.name] = np.array(T_hat, dtype=float)
        else:
            self.template_by_image[fp.name] = np.full((best_template_xy.shape[0], 2), np.nan, dtype=float)

        tracked_xy = None
        if update_tracker and args.temporal:
            if self.temporal_tracker is None:
                self.temporal_tracker = MultiGlintTracker(
                    n_tracks=4,
                    gate_px=args.temporal_gate_px,
                    max_missed=args.temporal_max_missed,
                )
            meas_labeled = np.full((4, 2), np.nan, dtype=float)
            for ti, ci, _ in (matches or []):
                if int(ti) >= 4:
                    continue
                meas_labeled[int(ti)] = cand_xy[int(ci)]
            # If no labeled matches, fall back to top candidates (unordered)
            if not np.isfinite(meas_labeled).any() and len(cand_xy) > 0:
                k = min(4, len(cand_xy))
                meas_labeled[:k] = cand_xy[:k]
            # If still empty but we have a hypothesis, seed from T_hat
            if not np.isfinite(meas_labeled).any() and T_hat is not None and T_hat.shape[0] >= 4:
                meas_labeled[:4] = T_hat[:4]
            tracked_xy, _meta = self.temporal_tracker.step_labeled(meas_labeled, frame_idx)

        cand_xy_disp, matches_disp = self._apply_overrides(fp.name, cand_xy, matches, T_hat)

        if update_tracker and args.temporal:
            if np.isfinite(tracked_xy).any():
                self.glints_by_image[fp.name] = tracked_xy
            elif T_hat is not None:
                # Fallback: keep direct frame detections if tracker produced no active tracks.
                glint_xy = np.full((T_hat.shape[0], 2), np.nan, dtype=float)
                for ti, ci, _ in matches_disp:
                    if 0 <= int(ti) < glint_xy.shape[0]:
                        glint_xy[int(ti)] = cand_xy_disp[int(ci)]
                self.glints_by_image[fp.name] = glint_xy
            else:
                self.glints_by_image[fp.name] = tracked_xy
        elif T_hat is not None:
            glint_xy = np.full((T_hat.shape[0], 2), np.nan, dtype=float)
            for ti, ci, _ in matches_disp:
                if 0 <= int(ti) < glint_xy.shape[0]:
                    glint_xy[int(ti)] = cand_xy_disp[int(ci)]
            self.glints_by_image[fp.name] = glint_xy

        roi_rect = None
        if roi_info is not None:
            roi_rect = (float(roi_info.offset_x), float(roi_info.offset_y), float(int(args.pupil_roi_size)))

        entry = {
            "status": "ok",
            "cand_xy_base": cand_xy,
            "matches_base": matches,
            "T_hat": T_hat,
            "tracked_xy": tracked_xy,
            "inliers": inliers,
            "mean_err": mean_err,
            "roi_rect": roi_rect,
        }
        with self.cache_lock:
            if compute_key == self.compute_key:
                self.compute_cache.setdefault(frame_id, {})[compute_key] = entry

        if not draw:
            return

        overlay_rgb, matches_disp, cand_xy_disp, _overlay_hit = self._render_overlay_for_entry(
            fp, frame_id, entry, args, compute_key
        )
        self._set_canvas_rgb(overlay_rgb)
        self.status_var.set(f"{self.idx+1}/{len(self.files)}: {fp.name} (computed)")
        self.current_fp = fp
        self.current_matches = matches_disp
        self.current_cand_xy = cand_xy_disp
        self._refresh_detection_list()

    def render_current(self) -> None:
        if self.bulk_processing:
            return
        if not self.files:
            return
        args, compute_key = self._get_args_and_compute_key()
        fp = self.files[self.idx]
        frame_id = self._frame_id(fp)

        entry = self._get_compute_entry(frame_id, compute_key)
        if args.temporal and self.last_idx is not None and self.idx < self.last_idx and entry is None:
            self._reset_temporal()
            for i in range(self.idx):
                self._process_frame(self.files[i], i, update_tracker=True, draw=False, args_override=args)
            entry = self._get_compute_entry(frame_id, compute_key)

        if entry is not None:
            try:
                overlay_rgb, matches_disp, cand_xy_disp, overlay_hit = self._render_overlay_for_entry(
                    fp, frame_id, entry, args, compute_key
                )
            except Exception:
                _LOGGER.exception("Cached render failed for %s", frame_id)
                overlay_rgb = None
            if overlay_rgb is None:
                self._process_frame(fp, self.idx, update_tracker=True, draw=True, args_override=args)
                self.last_idx = self.idx
                return
            status_suffix = "cached overlay" if overlay_hit else "cached compute"
            if entry.get("status") == "skipped":
                status_suffix = "skipped"
            self._set_canvas_rgb(overlay_rgb)
            self.status_var.set(f"{self.idx+1}/{len(self.files)}: {fp.name} ({status_suffix})")
            self.current_fp = fp
            self.current_matches = matches_disp
            self.current_cand_xy = cand_xy_disp
            self._refresh_detection_list()
            self.last_idx = self.idx
            return

        with self.cache_lock:
            cache_building = self.cache_building
        if cache_building:
            self._start_ondemand_cache_compute(fp, self.idx, args, compute_key)
            quick_rgb = None
            try:
                preview_key = self._preview_key(args, compute_key)
                if self.current_preview_base_name != fp.name or self.current_preview_base_cache_key != preview_key:
                    quick_rgb = self._quick_preview_rgb(fp, args, preview_key)
            except Exception:
                _LOGGER.exception("Quick preview failed for %s", frame_id)
                quick_rgb = None
            if quick_rgb is not None:
                self._set_canvas_rgb(quick_rgb)
                self.current_fp = fp
                self.current_matches = []
                self.current_cand_xy = np.empty((0, 2), dtype=float)
                self._refresh_detection_list()
            self.status_var.set(f"{self.idx+1}/{len(self.files)}: {fp.name} (caching...)")
            self._request_render(100)
            return

        self._process_frame(fp, self.idx, update_tracker=True, draw=True, args_override=args)
        self.last_idx = self.idx

    def next_image(self) -> None:
        if self.bulk_processing:
            return
        if not self.files:
            return
        self.idx = (self.idx + 1) % len(self.files)
        self._request_render()

    def prev_image(self) -> None:
        if self.bulk_processing:
            return
        if not self.files:
            return
        self.idx = (self.idx - 1) % len(self.files)
        self._request_render()

    def toggle_play(self) -> None:
        if self.bulk_processing:
            return
        self.playing = not self.playing
        self.play_btn.config(text="Pause" if self.playing else "Play")
        if self.playing:
            self._tick()
        else:
            if self.after_id is not None:
                self.root.after_cancel(self.after_id)
                self.after_id = None

    def _tick(self) -> None:
        if not self.playing:
            return
        self.next_image()
        self.after_id = self.root.after(100, self._tick)

    def zoom_in(self) -> None:
        self.zoom = min(8.0, self.zoom * 1.25)
        self._request_render()

    def zoom_out(self) -> None:
        self.zoom = max(0.2, self.zoom / 1.25)
        self._request_render()

    def zoom_reset(self) -> None:
        self.zoom = 1.0
        self._request_render()

    # Correction mode --------------------------------------------------
    def _toggle_correction(self) -> None:
        if self.corr_mode.get():
            self.corr_frame.pack(fill=tk.X, pady=(8, 0))
            self._refresh_detection_list()
        else:
            self.corr_frame.pack_forget()
        self._request_render()

    def _refresh_detection_list(self) -> None:
        if not self.corr_mode.get():
            return
        self.det_list.delete(0, tk.END)
        if not self.current_matches or self.current_cand_xy is None:
            return
        for ti, ci, d in self.current_matches:
            x, y = self.current_cand_xy[int(ci)]
            self.det_list.insert(tk.END, f"T{int(ti)} - ({x:.1f},{y:.1f})")

    def _hit_test_glint(self, x: float, y: float, radius: float = 12.0):
        if not self.current_matches or self.current_cand_xy is None:
            return None
        best = None
        best_d = float("inf")
        for ti, ci, _ in self.current_matches:
            px, py = self.current_cand_xy[int(ci)]
            d = float(np.hypot(px - x, py - y))
            if d < best_d:
                best_d = d
                best = int(ti)
        if best is None or best_d > radius:
            return None
        return best

    def _current_glint_positions(self):
        if not self.current_matches or self.current_cand_xy is None:
            return None
        gl = np.full((4, 2), np.nan, dtype=float)
        for ti, ci, _ in self.current_matches:
            if 0 <= int(ti) < 4 and 0 <= int(ci) < len(self.current_cand_xy):
                gl[int(ti)] = np.asarray(self.current_cand_xy[int(ci)], dtype=float).reshape(2)
        if not np.isfinite(gl).any():
            return None
        return gl

    def on_canvas_press(self, event) -> None:
        if not self.corr_mode.get() or self.current_fp is None:
            return
        if self.corr_target.get() == "constellation":
            gl = self._current_glint_positions()
            if gl is None:
                return
            valid = np.isfinite(gl[:, 0]) & np.isfinite(gl[:, 1])
            if not np.any(valid):
                return
            centroid = np.mean(gl[valid], axis=0)
            x = event.x / float(self.zoom)
            y = event.y / float(self.zoom)
            self.drag_ti = None
            self.drag_mode = "rotate" if self.corr_rotate.get() else "translate"
            self.drag_start = (float(x), float(y))
            self.drag_base_glints = gl.copy()
            self.drag_centroid = centroid.reshape(2)
            return
        x = event.x / float(self.zoom)
        y = event.y / float(self.zoom)
        hit = self._hit_test_glint(x, y, radius=12.0 / max(self.zoom, 1e-6))
        if hit is None:
            return
        self.drag_ti = int(hit)
        for i in range(self.det_list.size()):
            if self.det_list.get(i).startswith(f"T{self.drag_ti} "):
                self.det_list.selection_clear(0, tk.END)
                self.det_list.selection_set(i)
                self.det_list.activate(i)
                break
        overrides = self.overrides_by_image.setdefault(self.current_fp.name, {})
        overrides[self.drag_ti] = (float(x), float(y))
        self._request_render(0)

    def on_canvas_drag(self, event) -> None:
        if not self.corr_mode.get() or self.current_fp is None:
            return
        if self.corr_target.get() == "constellation" and self.drag_mode and self.drag_start and self.drag_base_glints is not None:
            x = event.x / float(self.zoom)
            y = event.y / float(self.zoom)
            base = np.asarray(self.drag_base_glints, dtype=float)
            centroid = np.asarray(self.drag_centroid, dtype=float).reshape(2)
            if self.drag_mode == "rotate":
                v0 = np.array(self.drag_start, dtype=float) - centroid
                v1 = np.array([x, y], dtype=float) - centroid
                ang0 = float(np.arctan2(v0[1], v0[0]))
                ang1 = float(np.arctan2(v1[1], v1[0]))
                theta = ang1 - ang0
                c, s = float(np.cos(theta)), float(np.sin(theta))
                R = np.array([[c, -s], [s, c]], dtype=float)
                new_pts = (base - centroid) @ R.T + centroid
            else:
                dx = float(x - self.drag_start[0])
                dy = float(y - self.drag_start[1])
                new_pts = base + np.array([dx, dy], dtype=float)
            overrides = self.overrides_by_image.setdefault(self.current_fp.name, {})
            for ti in range(min(4, new_pts.shape[0])):
                if np.isfinite(new_pts[ti]).all():
                    overrides[int(ti)] = (float(new_pts[ti, 0]), float(new_pts[ti, 1]))
            self._request_render(16)
            return
        if self.drag_ti is None:
            return
        x = event.x / float(self.zoom)
        y = event.y / float(self.zoom)
        overrides = self.overrides_by_image.setdefault(self.current_fp.name, {})
        overrides[self.drag_ti] = (float(x), float(y))
        self._request_render(16)

    def on_canvas_release(self, event) -> None:
        if not self.corr_mode.get():
            return
        self.drag_ti = None
        self.drag_mode = None
        self.drag_start = None
        self.drag_base_glints = None
        self.drag_centroid = None

    def clear_corrections(self) -> None:
        if self.current_fp is None:
            return
        self.overrides_by_image.pop(self.current_fp.name, None)
        self._request_render(0)

    def _args_to_config(self, args) -> dict:
        cfg = {k: getattr(args, k) for k in dir(args) if not k.startswith("_")}
        cfg["templates_path"] = self.templates_path
        cfg["image_config_path"] = self.image_config_path
        cfg["mirror"] = bool(self.mirror_var.get())
        return cfg

    def save_npz(self) -> None:
        if not self.files:
            messagebox.showwarning("Save NPZ", "No images loaded.")
            return
        if self.bulk_processing:
            messagebox.showinfo("Save NPZ", "Processing is already running. Please wait.")
            return
        initial_dir = str(self.folder) if self.folder else "."
        out_path = filedialog.asksaveasfilename(
            title="Save glints",
            defaultextension=".npz",
            filetypes=[("NumPy NPZ", "*.npz")],
            initialdir=initial_dir,
            initialfile="glints.npz",
        )
        if not out_path:
            return
        args_snapshot = self._get_args()
        files_snapshot = list(self.files)
        try:
            workers = int(self.save_workers_var.get())
        except Exception:
            workers = 0
        use_mp = workers > 1

        self.bulk_processing = True
        self.playing = False
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        self.play_btn.config(text="Play")
        self.status_var.set("Processing full dataset for save (0%)...")
        self.root.update_idletasks()

        def worker():
            try:
                self._prepare_templates(args_snapshot)
                self._reset_temporal()
                self.glints_by_image = {}
                self.template_by_image = {}
                total = len(files_snapshot)
                if use_mp:
                    if args_snapshot.temporal:
                        args_snapshot.temporal = False
                        args_snapshot.temporal_prior = False
                        self.root.after(0, lambda: self.status_var.set(
                            "MP save uses per-frame detection (temporal disabled)."
                        ))
                    self.glints_by_image, self.template_by_image = _run_save_multiproc(
                        files_snapshot,
                        args_snapshot,
                        self.template,
                        self.bank_templates,
                        self.ratio_index_bank,
                        self.ratio_index_single,
                        self.d_expected,
                        self.overrides_by_image,
                        workers,
                    )
                else:
                    for i, fp in enumerate(files_snapshot):
                        self._process_frame(fp, i, update_tracker=True, draw=False, args_override=args_snapshot)
                        if (i + 1) % 10 == 0 or i + 1 == total:
                            pct = int(round((i + 1) * 100.0 / max(total, 1)))
                            self.root.after(0, lambda p=pct, a=i + 1, t=total: self.status_var.set(
                                f"Processing full dataset for save ({p}%)... {a}/{t}"
                            ))
                config = self._args_to_config(args_snapshot)
                np.savez(out_path, glints=self.glints_by_image, template_xy=self.template_by_image, config=config)
            except Exception as exc:
                self.root.after(0, lambda exc=exc: messagebox.showerror("Save NPZ", f"Failed: {exc}"))
            else:
                self.root.after(0, lambda: messagebox.showinfo("Save NPZ", f"Saved: {out_path}"))
                self.root.after(0, lambda: self.status_var.set(f"Saved full dataset: {Path(out_path).name}"))
            finally:
                self.bulk_processing = False
                self.root.after(0, self._request_render)

        threading.Thread(target=worker, daemon=True).start()


def main() -> None:
    root = tk.Tk()
    PreviewApp(root)
    root.mainloop()


def _run_self_checks() -> bool:
    ok = True
    print("Running cache self-checks...")

    # 1) Frame id should be unique for same filename in different folders.
    base_dir = Path(tempfile.mkdtemp())
    a = base_dir / "a" / "0001.png"
    b = base_dir / "b" / "0001.png"
    a.parent.mkdir(parents=True, exist_ok=True)
    b.parent.mkdir(parents=True, exist_ok=True)
    frame_id_a = str(a.resolve())
    frame_id_b = str(b.resolve())
    if frame_id_a == frame_id_b:
        print("FAIL: frame_id collision for different paths.")
        ok = False
    else:
        print("OK: frame_id uniqueness.")

    # 2) Overlay-only changes should not change compute key.
    base_args = types.SimpleNamespace(
        matcher="sla",
        template_mode="single",
        score2_mode="contrast_support",
        percentile=99.5,
        kernel=11,
        median_ksize=3,
        eps=6.0,
    )
    key_base = _hash_payload(_compute_key_payload(base_args, None, None, None))
    overlay_args = types.SimpleNamespace(**vars(base_args))
    overlay_args.show_overlay = False
    overlay_args.preview_enhanced = True
    key_overlay = _hash_payload(_compute_key_payload(overlay_args, None, None, None))
    if key_base != key_overlay:
        print("FAIL: overlay-only change altered compute key.")
        ok = False
    else:
        print("OK: overlay-only change does not affect compute key.")

    # 3) Compute parameter change should alter compute key.
    changed_args = types.SimpleNamespace(**vars(base_args))
    changed_args.eps = 7.0
    key_changed = _hash_payload(_compute_key_payload(changed_args, None, None, None))
    if key_base == key_changed:
        print("FAIL: compute parameter change did not alter compute key.")
        ok = False
    else:
        print("OK: compute parameter change alters compute key.")

    return ok


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        raise SystemExit(0 if _run_self_checks() else 1)
    main()
