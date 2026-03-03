import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import json
import cv2
import numpy as np
import re
import threading
import multiprocessing as mp
import types
import tempfile
import zipfile
import shutil

from glint_pipeline import eval_gen as g
from glint_pipeline.temporal import MultiGlintTracker
from glint_pipeline.pupil_roi import compute_pupil_roi, map_points_to_full, resolve_pupil_roi_center

try:
    from PIL import Image, ImageTk
except ImportError as exc:
    raise SystemExit("Pillow is required. Install with: python -m pip install pillow") from exc


_MP_SAVE_STATE = {}


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
        self.photo = None
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
        self.frame_cache = {}
        self.cache_key = None
        self.cache_lock = threading.Lock()
        self.cache_thread = None
        self.cache_token = 0
        self.cache_building = False
        self.cache_debounce_id = None
        self.cache_progress_var = tk.DoubleVar(value=0.0)
        self.save_workers_var = tk.IntVar(value=0)
        self.default_settings_path = Path(r"C:\Users\vbmaq\Documents\virnet2\data\templates\chugh\preview_ui_settings_b2_081_win.json")

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
        ttk.Checkbutton(top, text="Mirror", variable=self.mirror_var, command=self.render_current).pack(side=tk.LEFT, padx=6)
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

        self.vars = {}
        def add_field(label, key, default, field_type="entry", values=None):
            row = ttk.Frame(ctrl)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label, width=20).pack(side=tk.LEFT)
            if field_type == "combobox":
                var = tk.StringVar(value=str(default))
                cb = ttk.Combobox(row, textvariable=var, values=values, width=14, state="readonly")
                cb.pack(side=tk.LEFT, fill=tk.X, expand=True)
                self.vars[key] = var
            elif field_type == "check":
                var = tk.BooleanVar(value=bool(default))
                chk = ttk.Checkbutton(row, variable=var)
                chk.pack(side=tk.LEFT)
                self.vars[key] = var
            else:
                var = tk.StringVar(value=str(default))
                ent = ttk.Entry(row, textvariable=var, width=16)
                ent.pack(side=tk.LEFT, fill=tk.X, expand=True)
                self.vars[key] = var

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

        add_field("matcher", "matcher", "hybrid", "combobox", ["ransac", "star", "hybrid", "sla"])
        add_field("template_mode", "template_mode", "bank", "combobox", ["single", "bank"])
        add_field("score2_mode", "score2_mode", "contrast_support", "combobox", ["heuristic", "contrast", "contrast_support", "ml_cc"])
        add_field("ml_model_path", "ml_model_path", "")
        add_field("thr_pct", "percentile", 99.7)
        add_field("kernel", "kernel", 11)
        add_field("enhance_mode", "enhance_mode", "tophat", "combobox", ["tophat", "dog", "highpass"])
        add_field("enhance_enable", "enhance_enable", True, "check")
        add_field("median_ksize", "median_ksize", 3)
        add_field("denoise", "denoise", True, "check")
        add_field("denoise_k", "denoise_k", 0)
        add_field("clahe", "clahe", True, "check")
        add_field("clahe_clip", "clahe_clip", 2.0)
        add_field("clahe_tiles", "clahe_tiles", 8)
        add_field("gamma_enable", "gamma_enable", True, "check")
        add_field("gamma", "gamma", 11.0)
        add_field("unsharp", "unsharp", False, "check")
        add_field("unsharp_amount", "unsharp_amount", 1.0)
        add_field("unsharp_sigma", "unsharp_sigma", 1.0)
        add_field("minmax", "minmax", True, "check")
        add_field("preview_enhanced", "preview_enhanced", False, "check")
        add_field("show_overlay", "show_overlay", True, "check")
        add_field("clean_k", "clean_k", 3)
        add_field("open_iter", "open_iter", 1)
        add_field("close_iter", "close_iter", 0)
        add_field("eps", "eps", 6.0)
        add_field("max_pool", "max_pool", 30)
        add_field("min_area", "min_area", 8)
        add_field("max_area", "max_area", 250)
        add_field("min_circ", "min_circ", 0.45)
        add_field("pupil_roi", "pupil_roi", False, "check")
        add_field("pupil_roi_size", "pupil_roi_size", 80)
        add_field("pupil_roi_pad_mode", "pupil_roi_pad_mode", "reflect", "combobox", ["reflect", "constant", "edge"])
        add_field("pupil_roi_pad_value", "pupil_roi_pad_value", 0)
        add_field("pupil_roi_fail_policy", "pupil_roi_fail_policy", "skip", "combobox", ["skip", "full_frame", "last_good"])
        add_field("pupil_roi_debug", "pupil_roi_debug", False, "check")
        add_field("pupil_source", "pupil_source", "none", "combobox", ["auto", "labels", "naive", "swirski", "npz", "none"])
        add_field("pupil_axis_mode", "pupil_axis_mode", "auto", "combobox", ["auto", "radius", "diameter"])
        add_field("cand_fallback", "cand_fallback", True, "check")
        add_field("cand_target_raw", "cand_target_raw", 12)
        add_field("cand_fallback_passes", "cand_fallback_passes", 4)
        add_field("cand_fallback_percentiles", "cand_fallback_percentiles", "99.5,99,98.5,98")
        add_field("support_M", "support_M", 30)
        add_field("support_tol", "support_tol", 0.10)
        add_field("support_w", "support_w", 0.15)
        add_field("contrast_r_inner", "contrast_r_inner", 3)
        add_field("contrast_r_outer1", "contrast_r_outer1", 5)
        add_field("contrast_r_outer2", "contrast_r_outer2", 8)
        add_field("dog_sigma1", "dog_sigma1", 1.0)
        add_field("dog_sigma2", "dog_sigma2", 2.2)
        add_field("ratio_tol", "ratio_tol", 0.12)
        add_field("pivot_P", "pivot_P", 8)
        add_field("max_seeds", "max_seeds", 200)
        add_field("layout_prior", "layout_prior", False, "check")
        add_field("layout_lambda", "layout_lambda", 0.25)
        add_field("sla_layout_prior", "sla_layout_prior", False, "check")
        add_field("sla_layout_lambda", "sla_layout_lambda", 0.25)
        add_field("sla_semantic_prior", "sla_semantic_prior", False, "check")
        add_field("sla_semantic_mode", "sla_semantic_mode", "full", "combobox", ["full", "top_only"])
        add_field("sla_semantic_lambda", "sla_semantic_lambda", 1.5)
        add_field("temporal", "temporal", False, "check")
        add_field("temporal_gate_px", "temporal_gate_px", 25.0)
        add_field("temporal_max_missed", "temporal_max_missed", 5)
        add_field("temporal_lambda", "temporal_lambda", 0.25)
        add_field("temporal_w_scale", "temporal_w_scale", 1.0)
        add_field("temporal_w_rot", "temporal_w_rot", 1.0)
        add_field("temporal_w_trans", "temporal_w_trans", 1.0)

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
        args.vote_M = 8
        args.vote_ratio_tol = 0.12
        args.vote_max_hyp = 2000
        args.vote_w_score2 = 0.0
        args.min_k = 3
        args.iters = 4000
        args.seed = 0
        args.scale_min = 0.6
        args.scale_max = 1.6
        args.disable_scale_gate = False
        args.matching = "greedy"
        args.appearance_tiebreak = False
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
        args.min_inliers = 3
        args.post_id_resolve = False
        args.template_bank_source = "default"
        args.template_bank_path = None
        args.bank_select_metric = "strict"
        args.template_build_mode = "procrustes"
        args.verbose_template = False
        args.match_tol = 10.0
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

    def _settings_cache_key(self, args) -> str:
        cfg = {k: getattr(args, k) for k in dir(args) if not k.startswith("_")}
        cfg["templates_path"] = self.templates_path
        cfg["image_config_path"] = self.image_config_path
        cfg["pupil_npz_path"] = self.pupil_npz_path
        return json.dumps(cfg, sort_keys=True, default=str)

    def _on_settings_change(self, *_args) -> None:
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
        cache_key = self._settings_cache_key(args_snapshot)
        if cache_key == self.cache_key and self.frame_cache:
            return
        self.cache_key = cache_key
        self.frame_cache = {}
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
                    if token != self.cache_token:
                        return
                    entry = self._compute_cache_entry(fp, i, args_snapshot, cache_state)
                    if entry is not None:
                        with self.cache_lock:
                            self.frame_cache[fp.name] = entry
                    if (i + 1) % 10 == 0 or i + 1 == total:
                        pct = int(round((i + 1) * 100.0 / max(total, 1)))
                        self.root.after(0, lambda p=pct, a=i + 1, t=total: self.status_var.set(
                            f"Precomputing cache ({p}%)... {a}/{t}"
                        ))
                        self.root.after(0, lambda p=pct: self.cache_progress_var.set(float(p)))
            finally:
                if token == self.cache_token:
                    self.cache_building = False
                    self.root.after(0, lambda: self.cache_progress_var.set(100.0))
                    self.root.after(0, self.render_current)

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
                if roi_decision.center is not None:
                    cache_state["last_good_pupil"] = (roi_decision.center[0], roi_decision.center[1], pr)

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

        overrides = self.overrides_by_image.get(fp.name, {})
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

        tracked_xy = None
        if args.temporal:
            if cache_state["temporal_tracker"] is None:
                cache_state["temporal_tracker"] = MultiGlintTracker(
                    n_tracks=4,
                    gate_px=args.temporal_gate_px,
                    max_missed=args.temporal_max_missed,
                )
            meas_labeled = np.full((4, 2), np.nan, dtype=float)
            for ti, ci, _ in matches_disp:
                if int(ti) >= 4:
                    continue
                meas_labeled[int(ti)] = cand_xy_disp[int(ci)]
            if not np.isfinite(meas_labeled).any() and len(cand_xy_disp) > 0:
                k = min(4, len(cand_xy_disp))
                meas_labeled[:k] = cand_xy_disp[:k]
            if not np.isfinite(meas_labeled).any() and T_hat is not None and T_hat.shape[0] >= 4:
                meas_labeled[:4] = T_hat[:4]
            tracked_xy, _meta = cache_state["temporal_tracker"].step_labeled(meas_labeled, frame_idx)

        title = f"{fp.name} | matcher={args.matcher} | inliers={inliers} | err={mean_err:.2f}px"
        if args.temporal:
            title += " | temporal"

        if not args.show_overlay:
            overlay = cv2.cvtColor(preview_base, cv2.COLOR_GRAY2BGR)
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            return {
                "overlay_rgb": overlay_rgb,
                "matches": matches_disp,
                "cand_xy": cand_xy_disp,
            }

        matches_draw = matches_disp
        T_hat_draw = T_hat
        if args.temporal and tracked_xy is not None:
            if np.isfinite(tracked_xy).any():
                T_hat_draw = tracked_xy
                T_hat_draw = np.where(np.isfinite(T_hat_draw), T_hat_draw, -1e6)
            matches_draw = []
        overlay = g.draw_overlay(preview_base, cand_xy_disp, T_hat_draw, matches_draw, title_text=title, gt_xy=None, match_tol=args.match_tol)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        return {
            "overlay_rgb": overlay_rgb,
            "matches": matches_draw,
            "cand_xy": cand_xy_disp,
        }

    def _invalidate_cache_for_frame(self, name: str) -> None:
        with self.cache_lock:
            if name in self.frame_cache:
                self.frame_cache.pop(name, None)

    def load_templates_json(self) -> None:
        path = filedialog.askopenfilename(
            title="Select templates JSON",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        self.templates_path = path
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
        self.cfg_var.set(f"image_config: {Path(path).name}")
        self.render_current()
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
                # Avoid UI "freeze" if filenames don't match: fall back to full frame.
                if "pupil_roi_fail_policy" in self.vars:
                    self.vars["pupil_roi_fail_policy"].set("full_frame")
                self.cfg_var.set(f"pupil_npz: {Path(path).name}")
                self.render_current()
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
        if self.pupil_npz_path:
            try:
                self.pupil_npz_map = g.load_pupil_npz(self.pupil_npz_path)
            except Exception as exc:
                messagebox.showerror("Error", f"Failed to load pupil NPZ: {exc}")
        self.cfg_var.set(f"settings: {Path(in_path).name}")
        self.render_current()
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
            self.root.after(0, self.render_current)
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

        # apply manual corrections for display
        overrides = self.overrides_by_image.get(fp.name, {})
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

        tracked_xy = None
        if update_tracker and args.temporal:
            if self.temporal_tracker is None:
                self.temporal_tracker = MultiGlintTracker(
                    n_tracks=4,
                    gate_px=args.temporal_gate_px,
                    max_missed=args.temporal_max_missed,
                )
            meas_labeled = np.full((4, 2), np.nan, dtype=float)
            for ti, ci, _ in matches_disp:
                if int(ti) >= 4:
                    continue
                meas_labeled[int(ti)] = cand_xy_disp[int(ci)]
            # If no labeled matches, fall back to top candidates (unordered)
            if not np.isfinite(meas_labeled).any() and len(cand_xy_disp) > 0:
                k = min(4, len(cand_xy_disp))
                meas_labeled[:k] = cand_xy_disp[:k]
            # If still empty but we have a hypothesis, seed from T_hat
            if not np.isfinite(meas_labeled).any() and T_hat is not None and T_hat.shape[0] >= 4:
                meas_labeled[:4] = T_hat[:4]
            tracked_xy, _meta = self.temporal_tracker.step_labeled(meas_labeled, frame_idx)
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

        if not draw:
            return

        title = f"{fp.name} | matcher={args.matcher} | inliers={inliers} | err={mean_err:.2f}px"
        if args.temporal:
            title += " | temporal"
        if self.zoom != 1.0:
            z = float(self.zoom)
            gray_disp = cv2.resize(preview_base, (int(preview_base.shape[1] * z), int(preview_base.shape[0] * z)), interpolation=cv2.INTER_LINEAR)
            if not args.show_overlay:
                overlay = cv2.cvtColor(gray_disp, cv2.COLOR_GRAY2BGR)
                overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(overlay_rgb)
                self.photo = ImageTk.PhotoImage(img)
                self.canvas.delete("all")
                self.canvas.config(width=img.width, height=img.height, scrollregion=(0, 0, img.width, img.height))
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
                self.status_var.set(f"{self.idx+1}/{len(self.files)}: {fp.name}")
                self.current_fp = fp
                self.current_matches = matches_disp
                self.current_cand_xy = cand_xy_disp
                self._refresh_detection_list()
                return
            cand_xy_s = cand_xy_disp * z
            T_hat_s = None if T_hat is None else (T_hat * z)
            matches_draw = matches_disp
            T_hat_draw = T_hat_s
            if args.temporal and tracked_xy is not None:
                if np.isfinite(tracked_xy).any():
                    T_hat_draw = tracked_xy * z
                    T_hat_draw = np.where(np.isfinite(T_hat_draw), T_hat_draw, -1e6)
                matches_draw = []
            overlay = g.draw_overlay(gray_disp, cand_xy_s, T_hat_draw, matches_draw, title_text=title, gt_xy=None, match_tol=args.match_tol * z)
            if args.pupil_roi_debug and roi_info is not None:
                rx0 = float(roi_info.offset_x) * z
                ry0 = float(roi_info.offset_y) * z
                rx1 = float(roi_info.offset_x + int(args.pupil_roi_size)) * z
                ry1 = float(roi_info.offset_y + int(args.pupil_roi_size)) * z
                cv2.rectangle(
                    overlay,
                    (int(round(rx0)), int(round(ry0))),
                    (int(round(rx1)), int(round(ry1))),
                    (0, 200, 255),
                    2,
                )
        else:
            if not args.show_overlay:
                overlay = cv2.cvtColor(preview_base, cv2.COLOR_GRAY2BGR)
                overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(overlay_rgb)
                self.photo = ImageTk.PhotoImage(img)
                self.canvas.delete("all")
                self.canvas.config(width=img.width, height=img.height, scrollregion=(0, 0, img.width, img.height))
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
                self.status_var.set(f"{self.idx+1}/{len(self.files)}: {fp.name}")
                self.current_fp = fp
                self.current_matches = matches_disp
                self.current_cand_xy = cand_xy_disp
                self._refresh_detection_list()
                return
            matches_draw = matches_disp
            T_hat_draw = T_hat
            if args.temporal and tracked_xy is not None:
                if np.isfinite(tracked_xy).any():
                    T_hat_draw = tracked_xy
                    T_hat_draw = np.where(np.isfinite(T_hat_draw), T_hat_draw, -1e6)
                matches_draw = []
            overlay = g.draw_overlay(preview_base, cand_xy_disp, T_hat_draw, matches_draw, title_text=title, gt_xy=None, match_tol=args.match_tol)
            if args.pupil_roi_debug and roi_info is not None:
                rx0 = roi_info.offset_x
                ry0 = roi_info.offset_y
                rx1 = roi_info.offset_x + int(args.pupil_roi_size)
                ry1 = roi_info.offset_y + int(args.pupil_roi_size)
                cv2.rectangle(
                    overlay,
                    (int(round(rx0)), int(round(ry0))),
                    (int(round(rx1)), int(round(ry1))),
                    (0, 200, 255),
                    2,
                )

        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        try:
            cache_key = self._settings_cache_key(args)
            if cache_key == self.cache_key:
                with self.cache_lock:
                    self.frame_cache[fp.name] = {
                        "overlay_rgb": overlay_rgb,
                        "matches": matches_disp,
                        "cand_xy": cand_xy_disp,
                    }
        except Exception:
            pass
        img = Image.fromarray(overlay_rgb)
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.config(width=img.width, height=img.height, scrollregion=(0, 0, img.width, img.height))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        self.status_var.set(f"{self.idx+1}/{len(self.files)}: {fp.name}")
        self.current_fp = fp
        self.current_matches = matches_disp
        self.current_cand_xy = cand_xy_disp
        self._refresh_detection_list()

    def render_current(self) -> None:
        if self.bulk_processing:
            return
        if not self.files:
            return
        args = self._get_args()
        cache_key = self._settings_cache_key(args)
        if args.temporal and self.last_idx is not None and self.idx < self.last_idx:
            self._reset_temporal()
            for i in range(self.idx):
                self._process_frame(self.files[i], i, update_tracker=True, draw=False)
        fp = self.files[self.idx]
        if cache_key == self.cache_key:
            with self.cache_lock:
                entry = self.frame_cache.get(fp.name)
            if entry is not None:
                overlay_rgb = entry["overlay_rgb"]
                if self.zoom != 1.0:
                    z = float(self.zoom)
                    overlay_rgb = cv2.resize(
                        overlay_rgb,
                        (int(overlay_rgb.shape[1] * z), int(overlay_rgb.shape[0] * z)),
                        interpolation=cv2.INTER_LINEAR,
                    )
                img = Image.fromarray(overlay_rgb)
                self.photo = ImageTk.PhotoImage(img)
                self.canvas.delete("all")
                self.canvas.config(width=img.width, height=img.height, scrollregion=(0, 0, img.width, img.height))
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
                self.status_var.set(f"{self.idx+1}/{len(self.files)}: {fp.name} (cached)")
                self.current_fp = fp
                self.current_matches = entry.get("matches", [])
                self.current_cand_xy = entry.get("cand_xy", np.empty((0, 2), dtype=float))
                self._refresh_detection_list()
                self.last_idx = self.idx
                return
        self._process_frame(fp, self.idx, update_tracker=True, draw=True)
        self.last_idx = self.idx

    def next_image(self) -> None:
        if self.bulk_processing:
            return
        if not self.files:
            return
        self.idx = (self.idx + 1) % len(self.files)
        self.render_current()

    def prev_image(self) -> None:
        if self.bulk_processing:
            return
        if not self.files:
            return
        self.idx = (self.idx - 1) % len(self.files)
        self.render_current()

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
        self.render_current()

    def zoom_out(self) -> None:
        self.zoom = max(0.2, self.zoom / 1.25)
        self.render_current()

    def zoom_reset(self) -> None:
        self.zoom = 1.0
        self.render_current()

    # Correction mode --------------------------------------------------
    def _toggle_correction(self) -> None:
        if self.corr_mode.get():
            self.corr_frame.pack(fill=tk.X, pady=(8, 0))
            self._refresh_detection_list()
        else:
            self.corr_frame.pack_forget()

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
        self._invalidate_cache_for_frame(self.current_fp.name)
        self.render_current()

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
            self._invalidate_cache_for_frame(self.current_fp.name)
            self.render_current()
            return
        if self.drag_ti is None:
            return
        x = event.x / float(self.zoom)
        y = event.y / float(self.zoom)
        overrides = self.overrides_by_image.setdefault(self.current_fp.name, {})
        overrides[self.drag_ti] = (float(x), float(y))
        self._invalidate_cache_for_frame(self.current_fp.name)
        self.render_current()

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
        self._invalidate_cache_for_frame(self.current_fp.name)
        self.render_current()

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
                self.root.after(0, self.render_current)

        threading.Thread(target=worker, daemon=True).start()


def main() -> None:
    root = tk.Tk()
    PreviewApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
