#!/usr/bin/env python3
"""
Glint constellation matching pipeline (single images, no pupil).

Input
- A folder of eye images
- (Optional) a JSON label file mapping image filename -> CornealReflectionLocations

Output
- annotated overlays saved to <folder>/annotated
- (if labels are provided) a text report + CSV with matching/localization metrics

Pipeline per image
1) Grayscale preprocessing:
   - median blur (kernel size via --median_ksize; auto-scaled unless --no_auto_scale)
2) Enhancement (small bright spot emphasis):
   - white top-hat (kernel via --kernel)
   - CLAHE (clip/tile via --clahe_clip, --clahe_tiles)
3) Over-detection thresholding:
   - global percentile threshold (--percentile) on enhanced image
4) Candidate extraction:
   - connected components -> per-blob features (area, circularity, max intensity)
   - rule prefilter (area/circularity/max intensity via --min_area/--max_area/--min_circ/--min_maxI)
5) Appearance scoring (score2):
   - heuristic (peakiness + radial symmetry + intensity) [--score2_mode heuristic]
   - contrast (local disk vs ring + DoG glintness) [--score2_mode contrast]
   - contrast + constellation support votes [--score2_mode contrast_support, --support_*]
   - ML on CC features [--score2_mode ml_cc, --ml_model_path]
6) Optional candidate fallback (when raw candidates are scarce):
   - multiple relaxed passes with merged candidates [--cand_fallback, --cand_target_raw, --cand_fallback_*]
7) Candidate pooling:
   - keep top-N by score2 [--max_pool]
8) Optional ROI gating:
   - border ROI [--roi_mode border, --roi_border_*]
   - pupil-centric ROI (auto/labels/naive/swirski) [--pupil_roi, --pupil_*]
9) Template selection:
   - single canonical template (median/procrustes) [--template_mode single]
   - template bank with per-image best selection [--template_mode bank, --bank_select_metric]
10) Constellation matching (per template):
   - ransac: RANSAC similarity fit + assignment [--matcher ransac]
   - star: distance voting + verify [--matcher star]
   - hybrid: ransac + star and choose best [--matcher hybrid]
   - sla: ratio-index seeds + pyramid growth [--matcher sla, --ratio_* / --sla_*]
   - optional layout priors:
     * soft layout prior for ransac/star/hybrid [--layout_prior]
     * soft/hard G0-top2 options for SLA [--sla_layout_prior / --sla_g0_top2]
11) Overlay visualization (if --visualize):
   - candidates (blue), transformed template (green), matches (yellow), GT (magenta)
   - optional debug text (fallback/score2/layout/SLA debug flags)
12) Optional evaluation vs ground truth (if labels provided):
   - per-image/per-subject reports and CSVs
   - identity-aware accuracy (G0..G4) and localization errors (mean/median)
   - identity-free accuracy/loc error (ignores glint labels)
13) Optional diagnostics:
   - candidate recall CSV + plots [--diag_candidate_recall, --diag_make_plots]
"""

from __future__ import annotations

import argparse
import itertools
import json
import bisect
import time
import multiprocessing as mp
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
import warnings
import math
from .pupil_roi import compute_pupil_roi, map_points_to_full, resolve_pupil_roi_center
try:
    from tqdm import tqdm
except ImportError:  # graceful fallback if tqdm is absent
    def tqdm(x, **kwargs):
        return x

_ID_DEBUG_COUNT = 0
_SLA_SEM_DEBUG_COUNT = 0
_TIERB_PLAN_SHOWN = False
_ML_MODEL_CACHE: Dict[str, Any] = {}
_ML_FEATURE_NAMES: Optional[List[str]] = None

# Template index (T0..Tn) -> Label index (G0..Gn)
# Initialized in run_eval() based on template size.
T_TO_G: Optional[List[int]] = None


# ----------------------------
# Template bootstrapping
# ----------------------------
def _load_templates_file(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("templates file must be a JSON object")
    if "templates" not in data or "default_bank" not in data:
        raise ValueError("templates file must contain 'templates' and 'default_bank'")
    return data


def load_default_template_bank() -> List[np.ndarray]:
    """
    Load default templates from templates/default_templates.json.
    JSON schema:
      {
        "templates": {"P1_DEFAULT": [[x,y],...], ...},
        "default_bank": ["P1_DEFAULT", ...]
      }
    """
    path = Path("templates") / "default_templates.json"
    if not path.exists():
        path = Path(__file__).resolve().parent / "templates" / "default_templates.json"
    if not path.exists():
        raise FileNotFoundError(f"Default templates file not found: {path}")
    data = _load_templates_file(path)
    templates = data["templates"]
    bank_names = data["default_bank"]
    bank = []
    for name in bank_names:
        if name not in templates:
            raise ValueError(f"Template name not found in templates file: {name}")
        bank.append(np.asarray(templates[name], dtype=float))
    return bank



def estimate_similarity(A: np.ndarray, B: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Estimate similarity transform mapping A (k,2) -> B (k,2).
    Returns: s, R(2x2), t(2,)
    """
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    if A.shape != B.shape or A.ndim != 2 or A.shape[1] != 2:
        raise ValueError("A and B must have shape (k,2) and match.")

    Ac = A - A.mean(axis=0)
    Bc = B - B.mean(axis=0)

    H = Ac.T @ Bc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # enforce proper rotation (no reflection)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    varA = float((Ac ** 2).sum())
    s = float(S.sum() / varA) if varA > 1e-9 else 1.0
    t = B.mean(axis=0) - s * (A.mean(axis=0) @ R.T)
    return s, R, t


def apply_similarity(P: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    P = np.asarray(P, float)
    return s * (P @ R.T) + t


def normalize(P: np.ndarray) -> np.ndarray:
    """
    Canonicalize constellation: zero-mean and unit RMS scale (identity-preserving).
    """
    P = np.asarray(P, float)
    Pc = P - P.mean(axis=0)
    scale = np.sqrt((Pc ** 2).sum() / len(Pc)) + 1e-9
    return Pc / scale


def best_alignment_by_permutation(P_ref: np.ndarray, P: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Find permutation of P that best aligns to P_ref under similarity transform.
    Returns (P_permuted, mean_alignment_error).
    """
    P_ref = np.asarray(P_ref, float)
    P = np.asarray(P, float)
    if len(P_ref) != len(P):
        raise ValueError("P_ref and P must have same number of points.")

    best_err = float("inf")
    best_Pp = None

    for perm in itertools.permutations(range(len(P))):
        Pp = P[list(perm)]
        s, R, t = estimate_similarity(Pp, P_ref)
        P_hat = apply_similarity(Pp, s, R, t)
        err = float(np.mean(np.linalg.norm(P_hat - P_ref, axis=1)))
        if err < best_err:
            best_err = err
            best_Pp = Pp.copy()

    assert best_Pp is not None
    return best_Pp, best_err


def build_template_median(P_list: List[np.ndarray]) -> np.ndarray:
    """
    Identity-preserving median template over labeled constellations (G0..G4 order).
    """
    if len(P_list) == 0:
        raise ValueError("At least one template constellation is required.")
    normed = [normalize(P) for P in P_list]
    stack = np.stack(normed, axis=0)
    template = np.median(stack, axis=0)
    template = normalize(template)
    return template


def build_template_from_labeled_sets(
    P_list: List[np.ndarray],
    iters: int = 10,
    tol: float = 1e-6,
    verbose: bool = False,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    """
    Generalized Procrustes Analysis without permutations (labels are consistent).
    """
    if len(P_list) == 0:
        raise ValueError("At least one template constellation is required.")
    normed = [normalize(P) for P in P_list]
    ref = normed[0]
    hist: List[Dict[str, float]] = []
    for it in range(iters):
        aligned = []
        errs = []
        for P in normed:
            s, R, t = estimate_similarity(P, ref)
            P_hat = apply_similarity(P, s, R, t)
            aligned.append(P_hat)
            errs.append(float(np.mean(np.linalg.norm(P_hat - ref, axis=1))))
        mean_shape = np.mean(np.stack(aligned, axis=0), axis=0)
        mean_shape = normalize(mean_shape)
        delta = float(np.sqrt(np.mean((mean_shape - ref) ** 2)))
        hist.append(dict(iter=it, mean_align_err=float(np.mean(errs)), ref_delta=delta))
        ref = mean_shape
        if delta < tol:
            break
    if verbose:
        for h in hist:
            print(f"[template] iter={h['iter']} align_err={h['mean_align_err']:.6f} ref_delta={h['ref_delta']:.6f}")
    return ref, hist


def constellation_scale(P: np.ndarray) -> float:
    """
    Robust constellation size: median of all pairwise distances.
    Used to derive an expected image scale for RANSAC gating.
    """
    P = np.asarray(P, float)
    n = len(P)
    if n < 2:
        return 0.0
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(float(np.linalg.norm(P[i] - P[j])))
    return float(np.median(dists)) if dists else 0.0


# ----------------------------
# Template bank utilities
# ----------------------------
def _ensure_template_shape(t: np.ndarray, name: str = "template") -> np.ndarray:
    t = np.asarray(t, float)
    if t.ndim != 2 or t.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N,2), got {t.shape}")
    if t.shape[0] < 3:
        raise ValueError(f"{name} must have at least 3 points, got {t.shape[0]}")
    return t


def load_template_bank(args) -> List[np.ndarray]:
    """
    Load template bank based on args:
      - default: uses templates/default_templates.json
      - custom: loads from .npy (shape BxNx2) or .json (list of Nx2 lists)
    Returns list of (N,2) float arrays.
    """
    if args.template_bank_source == "default":
        bank = [np.array(t, dtype=float) for t in load_default_template_bank()]
    else:
        if not args.template_bank_path:
            raise ValueError("template_bank_path must be provided when template_bank_source=custom")
        path = Path(args.template_bank_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Template bank file not found: {path}")
        if path.suffix.lower() == ".npy":
            arr = np.load(str(path))
            if arr.ndim != 3 or arr.shape[2] != 2:
                raise ValueError(f"Expected .npy shape (B,N,2); got {arr.shape}")
            if arr.shape[1] < 3:
                raise ValueError(f"Template must have at least 3 points; got {arr.shape[1]}")
            bank = [np.asarray(arr[i], float) for i in range(arr.shape[0])]
        elif path.suffix.lower() == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                bank = [np.asarray(t, float) for t in data]
            elif isinstance(data, dict) and "templates" in data and "default_bank" in data:
                templates = data["templates"]
                bank_names = data["default_bank"]
                bank = [np.asarray(templates[name], float) for name in bank_names if name in templates]
            else:
                raise ValueError("JSON template bank must be a list of Nx2 point lists or a templates/default_bank dict")
        else:
            raise ValueError("template_bank_path must be .npy or .json")

    bank_norm = []
    for i, t in enumerate(bank):
        t = _ensure_template_shape(t, name=f"template[{i}]")
        bank_norm.append(normalize(t))
    return bank_norm


def _pair_consistency_from_residuals(matches: Optional[List[Tuple[int, int, float]]]) -> float:
    if not matches or len(matches) < 2:
        return 0.0
    res = np.array([m[2] for m in matches], dtype=float)
    mean = float(np.mean(res))
    if mean <= 1e-6:
        return 1.0
    cv = float(np.std(res) / mean)
    return float(1.0 / (1.0 + cv))


def score_match_result(
    result_tuple,
    cand_score2: Optional[np.ndarray],
    appearance_tiebreak: bool,
    bank_select_metric: str = "strict",
    s_expected: Optional[float] = None,
) -> Tuple:
    """
    Returns a tuple where larger is better (lexicographic).
    Uses conservative scoring to avoid selecting clutter.
    Inputs:
      result_tuple = (s,R,t,matches,T_hat,app_sum)
    """
    if result_tuple is None:
        return (-1, -float("inf"), -float("inf"), -float("inf"))
    s, R, t, matches, T_hat, app_sum = result_tuple
    inliers = len(matches) if matches else 0
    if inliers <= 0:
        return (-1, -float("inf"), -float("inf"), -float("inf"))

    residuals = np.array([m[2] for m in matches], dtype=float)
    median_residual = float(np.median(residuals)) if residuals.size else float("inf")
    mean_residual = float(np.mean(residuals)) if residuals.size else float("inf")
    app_val = float(app_sum) if (appearance_tiebreak and app_sum is not None) else 0.0

    if bank_select_metric == "hybrid":
        pair_consistency = _pair_consistency_from_residuals(matches)
        return (inliers, pair_consistency, -median_residual, app_val)

    return (inliers, -median_residual, -mean_residual, app_val)


def extract_similarity_params(
    template_xy: np.ndarray, result_tuple
) -> Optional[Tuple[float, float, float, float]]:
    """
    Return (scale, theta, tx, ty) for a match result.
    Uses explicit (s,R,t) when available; falls back to estimating from T_hat.
    """
    if result_tuple is None:
        return None
    s, R, t, _matches, T_hat, _app_sum = result_tuple
    if R is None or t is None or not np.isfinite(np.array(R)).all():
        if T_hat is None:
            return None
        try:
            s, R, t = estimate_similarity(template_xy, T_hat)
        except Exception:
            return None
    try:
        theta = float(math.atan2(float(R[1, 0]), float(R[0, 0])))
    except Exception:
        theta = 0.0
    try:
        tx = float(t[0])
        ty = float(t[1])
    except Exception:
        tx, ty = 0.0, 0.0
    try:
        s = float(s)
    except Exception:
        s = 1.0
    return (s, theta, tx, ty)


def _wrap_angle_rad(a: float) -> float:
    return float((a + math.pi) % (2 * math.pi) - math.pi)


def score_match_result_temporal(
    template_xy: np.ndarray,
    result_tuple,
    cand_score2: Optional[np.ndarray],
    appearance_tiebreak: bool,
    bank_select_metric: str = "strict",
    prev_params: Optional[Tuple[float, float, float, float]] = None,
    args=None,
) -> Tuple:
    """
    Temporal wrapper for bank selection. Adds a soft penalty for transform changes.
    Larger is better (lexicographic); penalty is applied to the residual term.
    """
    base = score_match_result(
        result_tuple, cand_score2, appearance_tiebreak, bank_select_metric=bank_select_metric
    )
    if args is None or not getattr(args, "temporal_prior", False) or prev_params is None:
        return base
    cur_params = extract_similarity_params(template_xy, result_tuple)
    if cur_params is None:
        return base
    s0, th0, tx0, ty0 = prev_params
    s1, th1, tx1, ty1 = cur_params
    ds = float(s1 - s0)
    dth = _wrap_angle_rad(float(th1 - th0))
    dtx = float(tx1 - tx0)
    dty = float(ty1 - ty0)
    w_s = float(getattr(args, "temporal_w_scale", 1.0))
    w_r = float(getattr(args, "temporal_w_rot", 1.0))
    w_t = float(getattr(args, "temporal_w_trans", 1.0))
    lam = float(getattr(args, "temporal_lambda", 0.25))
    penalty = lam * (w_s * ds * ds + w_r * dth * dth + w_t * (dtx * dtx + dty * dty))
    # base[1] is -median_residual; subtract penalty to prefer temporal consistency
    return (base[0], base[1] - penalty, base[2], base[3])

# ----------------------------
# Candidate detection
# ----------------------------
def enhance_for_glints(
    gray: np.ndarray,
    kernel_size: int = 11,
    clahe_clip: float = 2.0,
    clahe_tiles: int = 8,
    median_ksize: int = 3,
    denoise: int = 1,
    denoise_k: int = 0,
    clahe_enable: int = 1,
    gamma: float = 1.0,
    unsharp: int = 0,
    unsharp_amount: float = 1.0,
    unsharp_sigma: float = 1.0,
    enhance_mode: str = "tophat",
    dog_sigma1: float = 1.0,
    dog_sigma2: float = 2.2,
    minmax: int = 1,
    enhance_enable: int = 1,
) -> np.ndarray:
    """Return enhanced image where small bright blobs are emphasized."""
    if denoise and (int(denoise_k) > 0 or int(median_ksize) > 0):
        k_med = int(denoise_k) if int(denoise_k) > 0 else int(median_ksize)
        k_med = max(1, int(k_med))
        if k_med % 2 == 0:
            k_med += 1
        gray_dn = cv2.medianBlur(gray, k_med)
    else:
        gray_dn = gray

    work = gray_dn
    if float(gamma) != 1.0:
        gval = max(1e-3, float(gamma))
        work_f = (work.astype(np.float32) / 255.0) ** (1.0 / gval)
        work = np.clip(work_f * 255.0, 0, 255).astype(np.uint8)

    if int(unsharp) != 0 and float(unsharp_amount) > 0:
        sigma = max(0.1, float(unsharp_sigma))
        blur = cv2.GaussianBlur(work, (0, 0), sigma)
        work = cv2.addWeighted(work, 1.0 + float(unsharp_amount), blur, -float(unsharp_amount), 0)

    if int(enhance_enable) != 0:
        enhance_mode = str(enhance_mode).lower().strip()
        if enhance_mode == "dog":
            s1 = max(0.1, float(dog_sigma1))
            s2 = max(0.1, float(dog_sigma2))
            if s2 < s1:
                s1, s2 = s2, s1
            g1 = cv2.GaussianBlur(work, (0, 0), sigmaX=s1)
            g2 = cv2.GaussianBlur(work, (0, 0), sigmaX=s2)
            enh_base = cv2.subtract(g1, g2)
        elif enhance_mode == "highpass":
            k = max(3, int(kernel_size))
            if k % 2 == 0:
                k += 1
            blur = cv2.blur(work, (k, k))
            enh_base = cv2.subtract(work, blur)
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            enh_base = cv2.morphologyEx(work, cv2.MORPH_TOPHAT, kernel)
    else:
        enh_base = work.copy()

    if clahe_enable and float(clahe_clip) > 0 and int(clahe_tiles) > 0:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tiles, clahe_tiles))
        enh = clahe.apply(enh_base)
    else:
        enh = enh_base
    if int(minmax) != 0 and enh.max() > 0:
        enh = cv2.normalize(enh, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return enh


def threshold_candidates(
    enh: np.ndarray,
    percentile: float = 99.7,
    clean_k: int = 3,
    open_iter: int = 1,
    close_iter: int = 0,
) -> np.ndarray:
    """Over-detect bright spots by a global percentile threshold on the enhanced image."""
    thr = np.percentile(enh, percentile)
    mask = (enh >= thr).astype(np.uint8) * 255
    k = max(1, int(clean_k))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    if int(open_iter) > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=int(open_iter))
    if int(close_iter) > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=int(close_iter))
    return mask


def connected_component_features(mask: np.ndarray, gray: np.ndarray) -> List[Dict[str, Any]]:
    """Extract candidate blobs and return a list of dicts with features."""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    rows: List[Dict[str, Any]] = []

    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[i]

        comp = (labels == i).astype(np.uint8)
        ys, xs = np.where(comp > 0)
        vals = gray[ys, xs]
        if vals.size == 0:
            continue
        max_i = int(vals.max())
        mean_i = float(vals.mean())

        # circularity
        comp255 = comp * 255
        contours, _ = cv2.findContours(comp255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        circ = 0.0
        if contours:
            cnt = contours[0]
            perim = cv2.arcLength(cnt, True)
            if perim > 1e-6:
                circ = float(4 * np.pi * area / (perim * perim))

        rows.append(
            dict(
                label=i,
                cx=float(cx),
                cy=float(cy),
                area=area,
                bbox_w=w,
                bbox_h=h,
                max_intensity=max_i,
                mean_intensity=mean_i,
                circularity=float(circ),
            )
        )
    return rows


def extract_cc_candidates_and_features(
    gray: np.ndarray,
    params: Dict[str, float],
    args,
    d_expected: Optional[np.ndarray] = None,
    percentile_override: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], np.ndarray]:
    """
    Return (cand_xy, feature_matrix, rows_debug, enhanced_image) using the same CC pipeline.
    Features are aligned with cand_xy and rows_debug.
    """
    kernel_eff = _make_odd(int(params["kernel_eff"]))
    median_ksize_eff = int(params["median_ksize_eff"])
    enh = enhance_for_glints(
        gray,
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
    percentile = float(args.percentile if percentile_override is None else percentile_override)
    mask = threshold_candidates(
        enh,
        percentile=percentile,
        clean_k=int(getattr(args, "clean_k", 3)),
        open_iter=int(getattr(args, "open_iter", 1)),
        close_iter=int(getattr(args, "close_iter", 0)),
    )

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    rows: List[Dict[str, Any]] = []
    feats: List[List[float]] = []
    comp_ids: List[int] = []
    H, W = gray.shape[:2]

    dog_map = None
    if hasattr(args, "dog_sigma1") and hasattr(args, "dog_sigma2"):
        dog_map = compute_dog(gray, float(args.dog_sigma1), float(args.dog_sigma2))

    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[i]
        comp = (labels == i).astype(np.uint8)
        ys, xs = np.where(comp > 0)
        if ys.size == 0:
            continue
        vals_enh = enh[ys, xs]
        vals_gray = gray[ys, xs]
        max_i = float(vals_gray.max())
        mean_i = float(vals_gray.mean())
        p95_i = float(np.percentile(vals_enh, 95))
        std_i = float(vals_enh.std())

        # circularity + perimeter + solidity
        comp255 = comp * 255
        contours, _ = cv2.findContours(comp255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perim = 0.0
        circ = 0.0
        solidity = 0.0
        if contours:
            cnt = contours[0]
            perim = float(cv2.arcLength(cnt, True))
            if perim > 1e-6:
                circ = float(4 * np.pi * area / (perim * perim))
            hull = cv2.convexHull(cnt)
            hull_area = float(cv2.contourArea(hull))
            if hull_area > 1e-6:
                solidity = float(area / hull_area)

        # eccentricity from coords
        ecc = 0.0
        if ys.size >= 3:
            yy = ys.astype(np.float32)
            xx = xs.astype(np.float32)
            xx -= xx.mean()
            yy -= yy.mean()
            cov = np.array(
                [
                    [float(np.mean(xx * xx)), float(np.mean(xx * yy))],
                    [float(np.mean(xx * yy)), float(np.mean(yy * yy))],
                ],
                dtype=float,
            )
            vals, _vecs = np.linalg.eigh(cov)
            vals = np.sort(vals)
            if vals[1] > 1e-6:
                ecc = float(np.sqrt(1.0 - (vals[0] / vals[1])))

        aspect = float(w) / (float(h) + 1e-6)
        ring_minus_inner = -compute_local_contrast(
            gray, cx, cy,
            int(getattr(args, "contrast_r_inner", 3)),
            int(getattr(args, "contrast_r_outer1", 5)),
            int(getattr(args, "contrast_r_outer2", 8)),
        )
        dog_val = float(dog_map[int(round(cy)), int(round(cx))]) if dog_map is not None else 0.0
        x_norm = float(cx) / max(1.0, float(W))
        y_norm = float(cy) / max(1.0, float(H))
        dist_border = float(min(cx, cy, W - 1 - cx, H - 1 - cy))
        dist_border_norm = dist_border / max(1.0, float(min(W, H)))

        rows.append(
            dict(
                label=i,
                cx=float(cx),
                cy=float(cy),
                area=area,
                bbox_w=w,
                bbox_h=h,
                max_intensity=max_i,
                mean_intensity=mean_i,
                circularity=float(circ),
            )
        )

        feats.append(
            [
                float(area),
                float(perim),
                float(circ),
                float(w),
                float(h),
                aspect,
                solidity,
                ecc,
                float(mean_i),
                float(max_i),
                float(p95_i),
                float(std_i),
                float(np.mean(vals_gray)),
                float(np.max(vals_gray)),
                ring_minus_inner,
                dog_val,
                x_norm,
                y_norm,
                dist_border_norm,
            ]
        )
        comp_ids.append(i)

    rows = rule_prefilter(rows, params["min_area_eff"], params["max_area_eff"], args.min_circ, args.min_maxI)
    # Keep features aligned to filtered rows
    if rows:
        keep_ids = set(r["label"] for r in rows)
        feats = [f for f, cid in zip(feats, comp_ids) if cid in keep_ids]
    else:
        feats = []

    cand_xy = np.array([(r["cx"], r["cy"]) for r in rows], dtype=float)
    feature_matrix = np.array(feats, dtype=float) if feats else np.empty((0, 0), dtype=float)
    return cand_xy, feature_matrix, rows, enh


def cc_feature_names() -> List[str]:
    return [
        "area",
        "perimeter",
        "circularity",
        "bbox_w",
        "bbox_h",
        "aspect",
        "solidity",
        "eccentricity",
        "mean_int",
        "max_int",
        "p95_int",
        "std_int",
        "mean_gray",
        "max_gray",
        "ring_minus_inner",
        "dog",
        "x_norm",
        "y_norm",
        "dist_border_norm",
    ]


def _load_ml_model(path: str):
    global _ML_MODEL_CACHE, _ML_FEATURE_NAMES
    if path in _ML_MODEL_CACHE:
        return _ML_MODEL_CACHE[path], _ML_FEATURE_NAMES
    try:
        import joblib
    except Exception as exc:
        raise RuntimeError("joblib is required to load ML model. Install with: pip install joblib") from exc
    obj = joblib.load(path)
    if isinstance(obj, dict) and "model" in obj:
        model = obj["model"]
        _ML_FEATURE_NAMES = obj.get("feature_names", None)
    else:
        model = obj
        _ML_FEATURE_NAMES = None
    _ML_MODEL_CACHE[path] = model
    return model, _ML_FEATURE_NAMES


def rule_prefilter(rows: List[Dict[str, Any]], min_area: int, max_area: int, min_circ: float, min_maxI: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        if not (min_area <= r["area"] <= max_area):
            continue
        if r["circularity"] < min_circ:
            continue
        if r["max_intensity"] < min_maxI:
            continue
        out.append(r)
    return out


# ----------------------------
# Patch-based appearance features (fast)
# ----------------------------
def peakiness_at(cx: float, cy: float, img: np.ndarray, r_in: int = 1, r_out: int = 4) -> float:
    """
    Core-vs-ring contrast around (cx,cy): max(core) - mean(ring).
    Glints tend to be "spiky": bright center with a darker ring.
    """
    cx, cy = int(round(cx)), int(round(cy))
    h, w = img.shape
    r = r_out
    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)

    patch = img[y0:y1, x0:x1].astype(np.float32)

    yy, xx = np.ogrid[y0:y1, x0:x1]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    core = patch[d <= r_in]
    ring = patch[(d > r_in) & (d <= r_out)]
    if core.size == 0 or ring.size == 0:
        return 0.0
    return float(core.max() - ring.mean())


def radial_symmetry(cx: float, cy: float, img: np.ndarray, r: int = 3, n: int = 16) -> float:
    """
    Std of samples around a small circle.
    Small, round glints should have lower radial std than smeary highlights.
    """
    cx, cy = float(cx), float(cy)
    h, w = img.shape
    vals: List[float] = []
    for k in range(n):
        ang = 2 * np.pi * k / n
        x = int(round(cx + r * np.cos(ang)))
        y = int(round(cy + r * np.sin(ang)))
        if 0 <= x < w and 0 <= y < h:
            vals.append(float(img[y, x]))
    if len(vals) < n // 2:
        return float("inf")
    return float(np.std(vals))


def add_appearance_score(
    rows: List[Dict[str, Any]],
    gray: np.ndarray,
    w_maxI: float = 1.0,
    w_circ: float = 15.0,
    w_peak: float = 2.0,
    w_sym: float = 5.0,
    peak_r_in: int = 1,
    peak_r_out: int = 4,
    sym_r: int = 3,
    sym_n: int = 16,
) -> List[Dict[str, Any]]:
    for r in rows:
        pk = peakiness_at(r["cx"], r["cy"], gray, r_in=peak_r_in, r_out=peak_r_out)
        rs = radial_symmetry(r["cx"], r["cy"], gray, r=sym_r, n=sym_n)
        r["peakiness"] = float(pk)
        r["radial_sym"] = float(rs)
        r["score2"] = float(
            w_maxI * r["max_intensity"]
            + w_circ * r["circularity"]
            + w_peak * r["peakiness"]
            - w_sym * r["radial_sym"]
        )
    return rows


def normalize_scores(vals: np.ndarray) -> np.ndarray:
    vals = np.asarray(vals, float)
    if vals.size == 0:
        return vals
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    if vmax - vmin < 1e-9:
        return np.zeros_like(vals, dtype=float)
    return (vals - vmin) / (vmax - vmin)


def layout_penalty_g0_top2(points_xy: np.ndarray, mode: str = "image") -> float:
    """
    points_xy: shape (5,2) array in the order [G0,G1,G2,G3,G4].
    returns penalty >= 0. 0 means constraint satisfied.
    Constraint: G0 must be among the top-2 highest points.
    """
    if points_xy is None or len(points_xy) != 5:
        return 0.0
    P = np.asarray(points_xy, float)
    if P.shape != (5, 2):
        return 0.0
    if not np.all(np.isfinite(P)):
        return 0.0

    if mode == "image":
        ys = P[:, 1]
        top2 = np.argsort(ys)[:2]
        if 0 in top2:
            return 0.0
        y2 = ys[top2[-1]]
        return max(0.0, float(ys[0] - y2))

    C = P.mean(axis=0, keepdims=True)
    X = P - C
    S = X.T @ X
    vals, vecs = np.linalg.eigh(S)
    order = np.argsort(vals)[::-1]
    vecs = vecs[:, order]
    v = vecs[:, 1]
    vv = X @ v
    top2 = np.argsort(vv)[-2:]
    if 0 in top2:
        return 0.0
    v2 = np.sort(vv)[-2]
    return max(0.0, float(v2 - vv[0]))


def sla_layout_penalty_g0_top2(matched_xy_by_glint: np.ndarray) -> float:
    """
    matched_xy_by_glint: shape (G,2) for G glints in fixed order [G0..G4],
                         with NaNs for unmatched points allowed.
    Returns penalty >= 0 (in pixels). 0 means satisfied.
    Rule: G0 must be among top-2 highest matched points (smallest y in image coords).
    If not enough matched points (fewer than 3 valid points), return 0 (don’t penalize).
    If G0 is unmatched (NaN), return 0 (don’t penalize).
    """
    if matched_xy_by_glint is None or matched_xy_by_glint.shape[0] < 1:
        return 0.0
    P = np.asarray(matched_xy_by_glint, float)
    if P.ndim != 2 or P.shape[1] != 2:
        return 0.0

    if np.any(np.isnan(P[0])):
        return 0.0

    valid = np.isfinite(P[:, 0]) & np.isfinite(P[:, 1])
    if valid.sum() < 3:
        return 0.0

    ys = P[valid, 1]
    idx_valid = np.flatnonzero(valid)
    order = np.argsort(ys)
    top2_glint_idx = set(idx_valid[order[:2]].tolist())
    if 0 in top2_glint_idx:
        return 0.0
    y2 = ys[order[1]]
    y0 = P[0, 1]
    return max(0.0, float(y0 - y2))


def points_g_order_from_matches(
    matches: Optional[List[Tuple[int, int, float]]],
    cand_xy: np.ndarray,
    t_to_g: List[int],
) -> Optional[np.ndarray]:
    if matches is None:
        return None
    P = np.full((len(t_to_g), 2), np.nan, dtype=float)
    for ti, ci, _ in matches:
        if ci < 0 or ci >= len(cand_xy) or ti < 0 or ti >= len(t_to_g):
            continue
        gi = t_to_g[ti]
        if 0 <= gi < len(t_to_g):
            P[gi] = cand_xy[ci]
    if not np.all(np.isfinite(P)):
        return None
    return P


def resolve_identity_permutation(
    template_xy: np.ndarray,
    pred_xy: np.ndarray,
    obs_xy: np.ndarray,
    obs_score2: Optional[np.ndarray],
    eps: float,
    layout_mode: str,
    id_lambda: float,
    id_gamma: float,
    id_eta: float,
    id_tau: Optional[float],
    s_fit: Optional[float] = None,
    R_fit: Optional[np.ndarray] = None,
    t_fit: Optional[np.ndarray] = None,
) -> Tuple[Optional[Tuple[int, ...]], Dict[str, float]]:
    """
    Resolve identity permutation for 5-point constellation.
    Returns (perm, cost_dict). perm maps template index -> obs index (0..4).
    If resolution not possible, perm is None.
    """
    T = np.asarray(template_xy, float)
    P = np.asarray(pred_xy, float)
    O = np.asarray(obs_xy, float)
    if O.shape != (5, 2) or T.shape != (5, 2) or P.shape != (5, 2):
        return None, {"geom": 0.0, "layout": 0.0, "pair": 0.0, "app": 0.0, "total": 0.0}

    tau = float(id_tau) if id_tau is not None else float(2.0 * eps)
    tau2 = tau * tau

    if obs_score2 is None:
        obs_score2 = np.zeros(5, dtype=float)
    obs_score2 = np.asarray(obs_score2, float)

    # optional template-space layout: map obs into template coordinates
    O_layout = O
    if layout_mode == "template" and s_fit is not None and R_fit is not None and t_fit is not None:
        if np.isfinite(s_fit) and abs(float(s_fit)) > 1e-9:
            O_layout = (O - t_fit) @ R_fit / float(s_fit)

    # template pair distances
    dT = []
    for i in range(5):
        for j in range(i + 1, 5):
            dT.append(float(np.linalg.norm(T[i] - T[j])))
    dT = np.array(dT, dtype=float)
    mT = float(np.median(dT)) if dT.size > 0 else 1.0

    best_perm = None
    best_cost = float("inf")
    best_geom = float("inf")
    best_layout = float("inf")
    best_pair = float("inf")
    best_app = float("inf")

    for perm in itertools.permutations(range(5)):
        # geometric reprojection cost
        geom = 0.0
        for g in range(5):
            d2 = float(np.sum((O[perm[g]] - P[g]) ** 2))
            geom += min(d2, tau2)

        # layout penalty (image or template coords)
        ys = O_layout[:, 1]
        rank_y = np.argsort(ys)
        rank_pos = np.empty(5, dtype=int)
        for r, idx in enumerate(rank_y):
            rank_pos[idx] = r
        x_med = float(np.median(O_layout[:, 0]))
        # constants
        a, b, c = 2.0, 0.75, 1.0
        p_g0 = 0.0 if rank_pos[perm[0]] <= 1 else 1.0
        p_sides = 0.0
        if O_layout[perm[1], 0] < x_med:
            p_sides += 1.0
        if O_layout[perm[2], 0] < x_med:
            p_sides += 1.0
        if O_layout[perm[3], 0] > x_med:
            p_sides += 1.0
        if O_layout[perm[4], 0] > x_med:
            p_sides += 1.0
        p_base = 0.0
        if rank_pos[perm[2]] < 3:
            p_base += 1.0
        if rank_pos[perm[3]] < 3:
            p_base += 1.0
        layout = a * p_g0 + b * p_sides + c * p_base

        # pairwise distance ratio penalty
        dP = []
        for i in range(5):
            for j in range(i + 1, 5):
                dP.append(float(np.linalg.norm(O[perm[i]] - O[perm[j]])))
        dP = np.array(dP, dtype=float)
        mP = float(np.median(dP)) if dP.size > 0 else 1.0
        pair = float(np.mean(np.abs(dP / mP - dT / mT))) if dP.size > 0 else 0.0

        app = -float(np.sum(obs_score2[list(perm)]))
        total = geom + float(id_lambda) * layout + float(id_gamma) * pair + float(id_eta) * app

        if total < best_cost - 1e-9:
            best_cost = total
            best_perm = perm
            best_geom = geom
            best_layout = layout
            best_pair = pair
            best_app = app
        elif abs(total - best_cost) <= 1e-9:
            if geom < best_geom - 1e-9:
                best_cost = total
                best_perm = perm
                best_geom = geom
                best_layout = layout
                best_pair = pair
                best_app = app
            elif abs(geom - best_geom) <= 1e-9:
                if app < best_app:
                    best_cost = total
                    best_perm = perm
                    best_geom = geom
                    best_layout = layout
                    best_pair = pair
                    best_app = app

    cost_dict = {
        "geom": float(best_geom),
        "layout": float(best_layout),
        "pair": float(best_pair),
        "app": float(best_app),
        "total": float(best_cost),
    }
    return best_perm, cost_dict


def semantic_penalty(
    pred_xy: np.ndarray,
    eps: float,
    top2_margin: float,
    base_ratio_min: float,
    side_margin: float,
    use_top: bool = True,
    use_base: bool = True,
    use_sides: bool = True,
) -> Dict[str, float]:
    """
    Compute semantic penalties on predicted constellation (historically 5-point, G0..G4).
    Returns dict with p_top, p_base, p_sides, p_total (unweighted).
    """
    P = np.asarray(pred_xy, float)
    if P.ndim != 2 or P.shape[1] != 2 or not np.all(np.isfinite(P)):
        return {"p_top": 0.0, "p_base": 0.0, "p_sides": 0.0, "p_total": 0.0}

    K = int(P.shape[0])
    if K <= 0:
        return {"p_top": 0.0, "p_base": 0.0, "p_sides": 0.0, "p_total": 0.0}

    ys = P[:, 1]
    order = np.argsort(ys)

    p_top = 0.0
    if use_top:
        # TOP rule: G0 among top-2 highest (smallest y), with margin
        y2 = float(ys[order[1]]) if len(order) >= 2 else float(ys[order[0]])
        p_top = 0.0 if (ys[0] <= y2 + float(top2_margin)) else 1.0

    p_base = 0.0
    if use_base and K >= 4:
        # BASE rule: G2-G3 is near max edge + G2/G3 not in top-2
        dists = []
        for i in range(K):
            for j in range(i + 1, K):
                dists.append(float(np.linalg.norm(P[i] - P[j])))
        dists = np.array(dists, dtype=float)
        dmax = float(np.max(dists)) if dists.size > 0 else 1.0
        d23 = float(np.linalg.norm(P[2] - P[3]))
        base_short = max(0.0, float(base_ratio_min) * dmax - d23) / max(dmax, 1e-6)
        # bottomness: G2/G3 should not be in top-2
        rank_pos = np.empty(K, dtype=int)
        for rr, idx in enumerate(order):
            rank_pos[idx] = rr
        bottom_viol = 0.0
        if rank_pos[2] <= 1:
            bottom_viol += 1.0
        if rank_pos[3] <= 1:
            bottom_viol += 1.0
        p_base = base_short + 0.5 * bottom_viol

    p_sides = 0.0
    if use_sides and K >= 5:
        # SIDES rule: G1/G2 right, G3/G4 left
        x_med = float(np.median(P[:, 0]))
        v = 0.0
        if P[1, 0] < x_med - float(side_margin):
            v += 1.0
        if P[2, 0] < x_med - float(side_margin):
            v += 1.0
        if P[3, 0] > x_med + float(side_margin):
            v += 1.0
        if P[4, 0] > x_med + float(side_margin):
            v += 1.0
        p_sides = v / 4.0

    return {"p_top": p_top, "p_base": p_base, "p_sides": p_sides, "p_total": p_top + p_base + p_sides}


def _is_reflection(template_xy: np.ndarray, pred_xy: np.ndarray) -> bool:
    """
    Detect reflection by orientation sign of triangle (0,1,2) in template vs predicted.
    """
    T = np.asarray(template_xy, float)
    P = np.asarray(pred_xy, float)
    if T.shape != (5, 2) or P.shape != (5, 2):
        return False
    v1 = T[1] - T[0]
    v2 = T[2] - T[0]
    w1 = P[1] - P[0]
    w2 = P[2] - P[0]
    cross_t = v1[0] * v2[1] - v1[1] * v2[0]
    cross_p = w1[0] * w2[1] - w1[1] * w2[0]
    if not (np.isfinite(cross_t) and np.isfinite(cross_p)):
        return False
    return (cross_t * cross_p) < 0

def compute_local_contrast(gray: np.ndarray, x: float, y: float, r_inner: int, r_outer1: int, r_outer2: int) -> float:
    xi = int(round(x))
    yi = int(round(y))
    h, w = gray.shape
    r2 = int(r_outer2)
    x0 = max(0, xi - r2)
    x1 = min(w, xi + r2 + 1)
    y0 = max(0, yi - r2)
    y1 = min(h, yi + r2 + 1)
    patch = gray[y0:y1, x0:x1].astype(np.float32)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    d = np.sqrt((xx - xi) ** 2 + (yy - yi) ** 2)
    inner = patch[d <= r_inner]
    ring = patch[(d >= r_outer1) & (d <= r_outer2)]
    if inner.size == 0 or ring.size == 0:
        return 0.0
    return float(inner.mean() - ring.mean())


def compute_dog(gray: np.ndarray, sigma1: float, sigma2: float) -> np.ndarray:
    g1 = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma1)
    g2 = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma2)
    return (g1.astype(np.float32) - g2.astype(np.float32))


def compute_expected_pairwise_distances(templates: List[np.ndarray]) -> np.ndarray:
    d_all = []
    for t in templates:
        P = np.asarray(t, float)
        if P.shape != (5, 2):
            continue
        if np.any(P < 0):
            # skip invalid points in this template
            continue
        Pn = normalize(P)
        for i in range(len(Pn)):
            for j in range(i + 1, len(Pn)):
                d_all.append(float(np.linalg.norm(Pn[i] - Pn[j])))
    return np.array(d_all, dtype=float)


def _make_odd(k: int) -> int:
    return k if k % 2 == 1 else k + 1


def _apply_image_config(args, path: str) -> None:
    """
    Load image enhancement config and override args.
    Expected keys (examples):
      denoise, denoise_k, tophat_k, clahe, clahe_clip, clahe_tile,
      thr_pct, clean_k, open_iter, close_iter, min_area, max_area, min_circ
    """
    cfg = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(cfg, dict):
        raise ValueError("image_config must be a JSON object")

    if "denoise" in cfg:
        args.denoise = int(cfg["denoise"])
    if "denoise_k" in cfg:
        args.denoise_k = int(cfg["denoise_k"])
    if "tophat_k" in cfg:
        args.kernel = int(cfg["tophat_k"])
    if "enhance_mode" in cfg:
        args.enhance_mode = str(cfg["enhance_mode"])
    if "clahe" in cfg:
        args.clahe = int(cfg["clahe"])
    if "clahe_clip" in cfg:
        args.clahe_clip = float(cfg["clahe_clip"])
    if "clahe_tile" in cfg:
        args.clahe_tiles = int(cfg["clahe_tile"])
    if "thr_pct" in cfg:
        thr = float(cfg["thr_pct"])
        args.percentile = thr / 10.0 if thr > 100.0 else thr
    if "clean_k" in cfg:
        args.clean_k = int(cfg["clean_k"])
    if "open_iter" in cfg:
        args.open_iter = int(cfg["open_iter"])
    if "close_iter" in cfg:
        args.close_iter = int(cfg["close_iter"])
    if "min_area" in cfg:
        args.min_area = int(cfg["min_area"])
    if "max_area" in cfg:
        args.max_area = int(cfg["max_area"])
    if "min_circ" in cfg:
        args.min_circ = float(cfg["min_circ"])
    if "gamma" in cfg:
        args.gamma = float(cfg["gamma"])
    if "unsharp" in cfg:
        args.unsharp = int(cfg["unsharp"])
    if "unsharp_amount" in cfg:
        args.unsharp_amount = float(cfg["unsharp_amount"])
    if "unsharp_sigma" in cfg:
        args.unsharp_sigma = float(cfg["unsharp_sigma"])
    if "minmax" in cfg:
        args.minmax = int(cfg["minmax"])
    if "enhance_enable" in cfg:
        args.enhance_enable = int(cfg["enhance_enable"])
    if "dog_sigma1" in cfg:
        args.dog_sigma1 = float(cfg["dog_sigma1"])
    if "dog_sigma2" in cfg:
        args.dog_sigma2 = float(cfg["dog_sigma2"])


def _apply_ui_settings(args: argparse.Namespace, path: str) -> None:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"UI settings not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    vars_data = data.get("vars", {}) or {}
    for k, v in vars_data.items():
        if hasattr(args, k):
            setattr(args, k, v)
    templates_path = data.get("templates_path")
    if templates_path:
        tp = Path(str(templates_path))
        if tp.exists():
            setattr(args, "template_bank_source", "custom")
            setattr(args, "template_bank_path", str(tp))
    image_config_path = data.get("image_config_path")
    if image_config_path:
        setattr(args, "image_config", str(image_config_path))
    pupil_npz_path = data.get("pupil_npz_path")
    if pupil_npz_path and not getattr(args, "pupil_npz", None):
        setattr(args, "pupil_npz", str(pupil_npz_path))


def detect_candidates_one_pass(
    gray: np.ndarray,
    params: Dict[str, float],
    args,
    d_expected: Optional[np.ndarray] = None,
    percentile_override: Optional[float] = None,
    kernel_add: int = 0,
):
    params = dict(params)
    params["kernel_eff"] = _make_odd(int(params["kernel_eff"]) + int(kernel_add))
    cand_xy, feature_matrix, rows, enh = extract_cc_candidates_and_features(
        gray, params, args, d_expected=d_expected, percentile_override=percentile_override
    )
    rows = add_appearance_score(rows, gray)
    score2_heur = np.array([r["score2"] for r in rows], dtype=float)

    support = np.zeros(len(cand_xy), dtype=float)
    cand_score2 = score2_heur
    if args.score2_mode == "ml_cc":
        if not getattr(args, "ml_model_path", None):
            raise RuntimeError("score2_mode=ml_cc requires --ml_model_path")
        model, feat_names = _load_ml_model(args.ml_model_path)
        if feature_matrix.size == 0:
            cand_score2 = np.zeros((0,), dtype=float)
        else:
            if feat_names is not None:
                # feature alignment (assume same order)
                pass
            try:
                proba = model.predict_proba(feature_matrix)[:, 1]
            except Exception:
                proba = model.predict(feature_matrix)
            cand_score2 = np.asarray(proba, dtype=float)
    elif args.score2_mode in ("contrast", "contrast_support") and cand_xy.size > 0:
        contrast_vals = []
        for x, y in cand_xy:
            contrast_vals.append(
                compute_local_contrast(
                    gray, x, y,
                    int(args.contrast_r_inner),
                    int(args.contrast_r_outer1),
                    int(args.contrast_r_outer2),
                )
            )
        contrast_vals = np.array(contrast_vals, dtype=float)
        dog_map = compute_dog(gray, float(args.dog_sigma1), float(args.dog_sigma2))
        dog_vals = []
        for x, y in cand_xy:
            xi = int(round(x))
            yi = int(round(y))
            if 0 <= yi < dog_map.shape[0] and 0 <= xi < dog_map.shape[1]:
                dog_vals.append(float(max(0.0, dog_map[yi, xi])))
            else:
                dog_vals.append(0.0)
        dog_vals = np.array(dog_vals, dtype=float)

        base_score = (
            normalize_scores(score2_heur)
            + 0.6 * normalize_scores(contrast_vals)
            + 0.4 * normalize_scores(dog_vals)
        )

        if args.score2_mode == "contrast_support" and d_expected is not None and d_expected.size > 0:
            M = max(1, int(args.support_M))
            order = np.argsort(-base_score)[:M]
            C = cand_xy[order]
            if len(C) >= 2:
                Cn = normalize(C)
                sup = np.zeros(len(Cn), dtype=float)
                tol = float(args.support_tol)
                for i in range(len(Cn)):
                    for j in range(i + 1, len(Cn)):
                        d = float(np.linalg.norm(Cn[i] - Cn[j]))
                        if np.any(np.abs(d_expected - d) <= tol * d_expected):
                            sup[i] += 1.0
                            sup[j] += 1.0
                support[order] = sup
                cand_score2 = base_score * (1.0 + float(args.support_w) * support)
            else:
                cand_score2 = base_score
        else:
            cand_score2 = base_score

    cand_raw_count = int(len(cand_xy))
    return cand_xy, rows, cand_score2, cand_raw_count, support


def merge_candidates(
    cand_xy_list: List[np.ndarray],
    cand_score2_list: List[np.ndarray],
    cand_support_list: List[np.ndarray],
    merge_eps: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not cand_xy_list:
        return np.empty((0, 2), dtype=float), np.empty((0,), dtype=float), np.empty((0,), dtype=float)
    xy = np.vstack([c for c in cand_xy_list if c.size > 0]) if any(c.size > 0 for c in cand_xy_list) else np.empty((0, 2), dtype=float)
    sc = np.hstack([s for s in cand_score2_list if s.size > 0]) if any(s.size > 0 for s in cand_score2_list) else np.empty((0,), dtype=float)
    sp = np.hstack([s for s in cand_support_list if s.size > 0]) if any(s.size > 0 for s in cand_support_list) else np.empty((0,), dtype=float)
    if xy.size == 0:
        return xy.reshape(0, 2), sc.reshape(0), sp.reshape(0)

    order = np.lexsort((xy[:, 1], xy[:, 0], -sc))
    xy = xy[order]
    sc = sc[order]
    sp = sp[order]
    keep_xy = []
    keep_sc = []
    keep_sp = []
    eps2 = float(merge_eps) ** 2
    for p, s, u in zip(xy, sc, sp):
        if not keep_xy:
            keep_xy.append(p)
            keep_sc.append(float(s))
            keep_sp.append(float(u))
            continue
        d2 = np.sum((np.array(keep_xy) - p) ** 2, axis=1)
        if np.any(d2 <= eps2):
            continue
        keep_xy.append(p)
        keep_sc.append(float(s))
        keep_sp.append(float(u))
    return np.array(keep_xy, dtype=float), np.array(keep_sc, dtype=float), np.array(keep_sp, dtype=float)


# ----------------------------
# Constellation matching (RANSAC)
# ----------------------------
def _match_hungarian(T_hat: np.ndarray, cand_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return row_ind, col_ind for min-cost assignment; uses scipy if available, else small exact solver."""
    D = np.sqrt(((T_hat[:, None, :] - cand_xy[None, :, :]) ** 2).sum(axis=2))
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore

        row_ind, col_ind = linear_sum_assignment(D)
        return row_ind, col_ind
    except Exception:
        # fallback to existing tiny solver used elsewhere
        pairs, _ = _best_bipartite_match_small(D)
        if not pairs:
            return np.array([], dtype=int), np.array([], dtype=int)
        rows, cols = zip(*pairs)
        return np.array(rows, dtype=int), np.array(cols, dtype=int)


def match_template_to_candidates_hungarian(T_hat: np.ndarray, cand_xy: np.ndarray, eps: float) -> List[Tuple[int, int, float]]:
    """
    Global 1-to-1 assignment (Hungarian) minimizes total distance, avoiding greedy local traps.
    Returns list of (template_idx, cand_idx, distance_px) within eps.
    """
    T_hat = np.asarray(T_hat, float)
    C = np.asarray(cand_xy, float)
    if T_hat.size == 0 or C.size == 0:
        return []
    row_ind, col_ind = _match_hungarian(T_hat, C)
    matches: List[Tuple[int, int, float]] = []
    for r, c in zip(row_ind, col_ind):
        d = float(np.linalg.norm(T_hat[r] - C[c]))
        if d <= eps:
            matches.append((int(r), int(c), d))
    return matches


def match_template_to_candidates_greedy(T_hat: np.ndarray, cand_xy: np.ndarray, eps: float) -> List[Tuple[int, int, float]]:
    """
    Greedy one-to-one: each template takes nearest unused candidate within eps.
    Kept for A/B testing.
    """
    C = np.asarray(cand_xy, float)
    used = set()
    matches: List[Tuple[int, int, float]] = []
    for ti, p in enumerate(T_hat):
        d = np.sqrt(((C - p) ** 2).sum(axis=1))
        ci = int(np.argmin(d))
        if d[ci] <= eps and ci not in used:
            used.add(ci)
            matches.append((ti, ci, float(d[ci])))
    return matches


def build_ratio_index(template_xy: np.ndarray) -> Dict[str, Any]:
    """
    Build per-pivot ratio lists for a template.
    Each pivot i has list of (log_ratio, ratio, j, k) sorted by log_ratio.
    """
    T = np.asarray(template_xy, float)
    K = len(T)
    pivot_lists: List[List[Tuple[float, float, int, int]]] = []
    for i in range(K):
        items = []
        for j in range(K):
            if j == i:
                continue
            for k in range(K):
                if k == i or k == j:
                    continue
                dij = float(np.linalg.norm(T[i] - T[j]))
                dik = float(np.linalg.norm(T[i] - T[k]))
                if dij <= 1e-9 or dik <= 1e-9:
                    continue
                r = min(dij / dik, dik / dij)
                if r <= 0:
                    continue
                items.append((math.log(r), r, j, k))
        items.sort(key=lambda x: x[0])
        pivot_lists.append(items)
    return {"pivot_lists": pivot_lists, "template_xy": T}


def _ratio_range_query(pivot_list: List[Tuple[float, float, int, int]], log_r: float, tol: float):
    lo = log_r - tol
    hi = log_r + tol
    logs = [t[0] for t in pivot_list]
    a = bisect.bisect_left(logs, lo)
    b = bisect.bisect_right(logs, hi)
    return pivot_list[a:b]


def sla_pyramid_constellation(
    template_xy: np.ndarray,
    cand_xy: np.ndarray,
    cand_score2: Optional[np.ndarray] = None,
    eps: float = 6.0,
    min_inliers: int = 3,
    matching: str = "greedy",
    appearance_tiebreak: bool = True,
    ratio_index: Optional[Dict[str, Any]] = None,
    pivot_P: int = 8,
    ratio_tol: float = 0.12,
    max_seeds: int = 500,
    grow_resid_max: Optional[float] = None,
    sla_w_seed_score2: float = 1.0,
    sla_w_seed_geom: float = 1.0,
    max_seeds_per_pivot: int = 80,
    sla_adaptive_ratio_tol: bool = True,
    sla_ratio_tol_min: float = 0.06,
    sla_ratio_tol_refN: int = 12,
    sla_scale_min: float = 0.2,
    sla_scale_max: float = 5.0,
    sla_g0_top2: bool = False,
    sla_semantic_prior: bool = False,
    sla_semantic_lambda: float = 1.5,
    sla_semantic_mode: str = "full",
    sla_semantic_hard: bool = False,
    sla_mirror_reject: bool = True,
    sla_top2_margin: float = 0.0,
    sla_base_ratio_min: float = 0.80,
    sla_side_margin: float = 0.0,
    sla_semantic_debug: bool = False,
    sla_layout_prior: bool = False,
    sla_layout_lambda: float = 0.25,
    sla_layout_mode: str = "image",
    sla_layout_debug: bool = False,
    sla_raw_count: Optional[int] = None,
):
    """
    SLA/Pyramid-style constellation matcher.
    Returns (s, R, t, matches, T_hat, app_sum) or None.
    """
    T = np.asarray(template_xy, float)
    C = np.asarray(cand_xy, float)
    if cand_score2 is None:
        cand_score2 = np.zeros(len(C), float)
    cand_score2 = np.asarray(cand_score2, float)
    K, N = len(T), len(C)
    if N < min_inliers or K < min_inliers:
        return None

    if ratio_index is None:
        ratio_index = build_ratio_index(T)
    pivot_lists = ratio_index["pivot_lists"]

    grow_resid_max = float(eps if grow_resid_max is None else grow_resid_max)

    sem_use_top = True
    sem_use_base = True
    sem_use_sides = True
    if str(sla_semantic_mode).lower().strip() == "top_only":
        sem_use_base = False
        sem_use_sides = False

    # adaptive ratio tolerance based on candidate pool size
    eff_ratio_tol = float(ratio_tol)
    if sla_adaptive_ratio_tol:
        refN = max(1, int(sla_ratio_tol_refN))
        eff_ratio_tol = float(ratio_tol) * float(np.sqrt(refN / max(N, 1)))
        eff_ratio_tol = float(np.clip(eff_ratio_tol, float(sla_ratio_tol_min), float(ratio_tol)))

    # pick top pivots by score2
    pivot_P = max(1, int(pivot_P))
    piv_idx = np.argsort(-cand_score2)[: min(pivot_P, N)]

    seeds = []
    seed_set = set()
    num_seeds_generated = 0
    seeds_per_pivot: Dict[int, List[Tuple[float, Tuple[int, int, int], Tuple[int, int, int]]]] = {}
    sem_veto_seed = 0
    sem_veto_final = 0
    best_sem = None
    best_mirror = False
    for cp in piv_idx:
        per_pivot = []
        for a in range(N):
            if a == cp:
                continue
            for b in range(a + 1, N):
                if b == cp:
                    continue
                da = float(np.linalg.norm(C[cp] - C[a]))
                db = float(np.linalg.norm(C[cp] - C[b]))
                if da <= 1e-9 or db <= 1e-9:
                    continue
                r_obs = min(da / db, db / da)
                log_r = math.log(r_obs)
                for ti in range(K):
                    plist = pivot_lists[ti]
                    if not plist:
                        continue
                    for log_rt, r_t, tj, tk in _ratio_range_query(plist, log_r, eff_ratio_tol):
                        # two possible assignments for (a,b) -> (tj,tk)
                        seed1 = ((ti, tj, tk), (cp, a, b))
                        seed2 = ((ti, tj, tk), (cp, b, a))
                        for sd in (seed1, seed2):
                            if sd in seed_set:
                                continue
                            seed_set.add(sd)
                            num_seeds_generated += 1
                            (ti1, tj1, tk1), (cp1, ca1, cb1) = sd
                            t_idx = np.array([ti1, tj1, tk1])
                            c_idx = np.array([cp1, ca1, cb1])
                            A = T[t_idx]
                            B = C[c_idx]
                            try:
                                s0, R0, t0 = estimate_similarity(A, B)
                            except np.linalg.LinAlgError:
                                continue
                            B_hat = apply_similarity(A, s0, R0, t0)
                            seed_resid = float(np.median(np.sqrt(((B_hat - B) ** 2).sum(axis=1))))
                            if sla_semantic_prior:
                                sem = semantic_penalty(
                                    B_hat,
                                    eps,
                                    sla_top2_margin,
                                    sla_base_ratio_min,
                                    sla_side_margin,
                                    use_top=sem_use_top,
                                    use_base=sem_use_base,
                                    use_sides=sem_use_sides,
                                )
                                mirror = _is_reflection(T, B_hat)
                                if sla_mirror_reject and mirror and sla_semantic_hard:
                                    sem_veto_seed += 1
                                    continue
                                if sla_semantic_hard:
                                    if sem_use_top and sem["p_top"] >= 1.0:
                                        sem_veto_seed += 1
                                        continue
                                    if sem_use_sides and sem["p_sides"] > 0.5:
                                        sem_veto_seed += 1
                                        continue
                                    if sem_use_base and B_hat.shape[0] >= 4:
                                        d23 = float(np.linalg.norm(B_hat[2] - B_hat[3]))
                                        dmax = max(
                                            float(np.max([np.linalg.norm(B_hat[i] - B_hat[j]) for i in range(5) for j in range(i + 1, 5)])),
                                            1e-6,
                                        )
                                        if d23 < float(sla_base_ratio_min) * dmax:
                                            sem_veto_seed += 1
                                            continue
                            seed_score = float(sla_w_seed_score2) * float(
                                cand_score2[cp1] + cand_score2[ca1] + cand_score2[cb1]
                            ) - float(sla_w_seed_geom) * seed_resid
                            per_pivot.append((seed_score, (ti1, tj1, tk1), (cp1, ca1, cb1)))
        if per_pivot:
            per_pivot.sort(key=lambda x: (-x[0], x[1], x[2]))
            per_pivot = per_pivot[: max(1, int(max_seeds_per_pivot))]
            seeds_per_pivot[int(cp)] = per_pivot

    for cp in sorted(seeds_per_pivot.keys()):
        seeds.extend(seeds_per_pivot[cp])

    if seeds:
        seeds.sort(key=lambda x: (-x[0], x[1], x[2]))
        seeds = seeds[: max_seeds]
    num_seeds_kept = len(seeds)

    best = None
    best_inl = -1
    best_err = float("inf")
    best_cost = float("inf")
    best_app = -float("inf")
    best_med = float("inf")
    best_max = float("inf")

    for seed_score, (ti, tj, tk), (cp, ca, cb) in seeds:
        t_idx = np.array([ti, tj, tk])
        c_idx = np.array([cp, ca, cb])
        A = T[t_idx]
        B = C[c_idx]
        try:
            s, R, t = estimate_similarity(A, B)
        except np.linalg.LinAlgError:
            continue
        T_hat = apply_similarity(T, s, R, t)

        matched_t = set(t_idx.tolist())
        matched_c = set(c_idx.tolist())
        matches = [(int(ti), int(cp), float(np.linalg.norm(T_hat[ti] - C[cp]))),
                   (int(tj), int(ca), float(np.linalg.norm(T_hat[tj] - C[ca]))),
                   (int(tk), int(cb), float(np.linalg.norm(T_hat[tk] - C[cb])))]

        # grow hypotheses
        changed = True
        while changed:
            changed = False
            for ti2 in range(K):
                if ti2 in matched_t:
                    continue
                # candidates within eps
                d = np.sqrt(((C - T_hat[ti2]) ** 2).sum(axis=1))
                idxs = np.where(d <= eps)[0]
                if idxs.size == 0:
                    continue
                # choose best by distance then score2
                idxs = sorted(idxs, key=lambda c: (d[c], -cand_score2[c]))
                added = False
                for ci2 in idxs:
                    if ci2 in matched_c:
                        continue
                    tmp_t = [m[0] for m in matches] + [ti2]
                    tmp_c = [m[1] for m in matches] + [ci2]
                    A2 = T[tmp_t]
                    B2 = C[tmp_c]
                    try:
                        s2, R2, t2 = estimate_similarity(A2, B2)
                    except np.linalg.LinAlgError:
                        continue
                    T_hat2 = apply_similarity(T, s2, R2, t2)
                    res = [float(np.linalg.norm(T_hat2[ti3] - C[ci3])) for ti3, ci3 in zip(tmp_t, tmp_c)]
                    if np.median(res) <= grow_resid_max:
                        # accept
                        matches.append((int(ti2), int(ci2), float(np.linalg.norm(T_hat2[ti2] - C[ci2]))))
                        matched_t.add(ti2)
                        matched_c.add(ci2)
                        s, R, t = s2, R2, t2
                        T_hat = T_hat2
                        changed = True
                        added = True
                        break
                if added:
                    continue

        # final assignment polishing
        if matching == "hungarian":
            matches = match_template_to_candidates_hungarian(T_hat, C, eps=eps)
        else:
            matches = match_template_to_candidates_greedy(T_hat, C, eps=eps)

        inl = len(matches)
        if inl < min_inliers:
            continue
        if matches:
            t_idx_all = [m[0] for m in matches]
            c_idx_all = [m[1] for m in matches]
            A_all = T[t_idx_all]
            B_all = C[c_idx_all]
            try:
                s_fit, R_fit, t_fit = estimate_similarity(A_all, B_all)
            except np.linalg.LinAlgError:
                continue
            T_hat = apply_similarity(T, s_fit, R_fit, t_fit)
            res = np.sqrt(((T_hat[t_idx_all] - B_all) ** 2).sum(axis=1))
            med_res = float(np.median(res)) if res.size else float("inf")
            max_res = float(np.max(res)) if res.size else float("inf")
        else:
            s_fit, R_fit, t_fit = s, R, t
            med_res = float("inf")
            max_res = float("inf")

        if not np.isfinite(s_fit) or s_fit < float(sla_scale_min) or s_fit > float(sla_scale_max):
            continue
        if med_res > eps or max_res > 2.0 * eps:
            continue

        sem = {"p_top": 0.0, "p_base": 0.0, "p_sides": 0.0, "p_total": 0.0}
        mirror = False
        if sla_semantic_prior:
            sem = semantic_penalty(
                T_hat,
                eps,
                sla_top2_margin,
                sla_base_ratio_min,
                sla_side_margin,
                use_top=sem_use_top,
                use_base=sem_use_base,
                use_sides=sem_use_sides,
            )
            mirror = _is_reflection(T, T_hat)
            if sla_mirror_reject and mirror:
                if sla_semantic_hard:
                    sem_veto_final += 1
                    continue
            if sla_semantic_hard:
                if sem_use_top and sem["p_top"] >= 1.0:
                    sem_veto_final += 1
                    continue
                if sem_use_sides and sem["p_sides"] > 0.5:
                    sem_veto_final += 1
                    continue
                if sem_use_base and T_hat.shape[0] >= 4:
                    d23 = float(np.linalg.norm(T_hat[2] - T_hat[3]))
                    dmax = max(
                        float(np.max([np.linalg.norm(T_hat[i] - T_hat[j]) for i in range(5) for j in range(i + 1, 5)])),
                        1e-6,
                    )
                    if d23 < float(sla_base_ratio_min) * dmax:
                        sem_veto_final += 1
                        continue

        if sla_g0_top2:
            g0_match = None
            for ti2, ci2, _ in matches:
                if ti2 == 0:
                    g0_match = ci2
                    break
            if g0_match is not None and len(matches) >= 2:
                ys = np.array([C[ci, 1] for (_, ci, _) in matches], dtype=float)
                y0 = float(C[g0_match, 1])
                order = np.argsort(ys)
                y2 = float(ys[order[1]]) if len(order) >= 2 else float(ys[order[0]])
                if y0 > y2:
                    continue

        err = med_res
        app = float(sum(cand_score2[m[1]] for m in matches)) if cand_score2 is not None else 0.0
        lp = 0.0
        if sla_layout_prior and sla_layout_mode == "image":
            glint_xy_hat = np.full((5, 2), np.nan, dtype=float)
            for ti2, ci2, _ in matches:
                if 0 <= ti2 < 5 and 0 <= ci2 < len(C):
                    glint_xy_hat[ti2] = C[ci2]
            lp = sla_layout_penalty_g0_top2(glint_xy_hat)
        sem_pen = float(sem["p_total"]) if sla_semantic_prior else 0.0
        mirror_pen = 0.0
        if sla_mirror_reject and mirror and not sla_semantic_hard:
            mirror_pen = 10.0
        total_cost = float(err + float(sla_layout_lambda) * lp + float(sla_semantic_lambda) * sem_pen + mirror_pen) if (sla_layout_prior or sla_semantic_prior or mirror_pen > 0.0) else float(err)

        better = False
        if inl > best_inl:
            better = True
        elif inl == best_inl:
            if total_cost < best_cost - 1e-9:
                better = True
            elif abs(total_cost - best_cost) <= 1e-9:
                if app > best_app:
                    better = True
                elif app == best_app and err < best_err:
                    better = True
        if better:
            best_inl, best_err, best_app, best_cost = inl, err, app, total_cost
            best_med, best_max = med_res, max_res
            best = (s_fit, R_fit, t_fit, matches, T_hat, app)
            best_sem = sem
            best_mirror = mirror

    n_raw = "na" if sla_raw_count is None else str(int(sla_raw_count))
    if sla_layout_debug:
        print(f"[sla] N_raw={n_raw} N_pool={N} seeds_gen={num_seeds_generated} seeds_kept={num_seeds_kept} "
              f"best_inl={best_inl} best_med={best_med:.2f} best_max={best_max:.2f}")
    if sla_semantic_debug:
        global _SLA_SEM_DEBUG_COUNT
        if _SLA_SEM_DEBUG_COUNT < 10:
            sem_txt = "na" if best_sem is None else f"top={best_sem['p_top']:.2f} base={best_sem['p_base']:.2f} sides={best_sem['p_sides']:.2f}"
            print(f"[sla_sem] N_raw={n_raw} N_pool={N} seeds_gen={num_seeds_generated} seeds_kept={num_seeds_kept} "
                  f"veto_seed={sem_veto_seed} veto_final={sem_veto_final} "
                  f"best_inl={best_inl} best_med={best_med:.2f} best_max={best_max:.2f} "
                  f"mirror={int(best_mirror)} {sem_txt}")
            _SLA_SEM_DEBUG_COUNT += 1
    return best if best_inl >= min_inliers else None


def startracker_constellation(
    template_xy: np.ndarray,
    cand_xy: np.ndarray,
    cand_score2: Optional[np.ndarray] = None,
    eps: float = 6.0,
    min_inliers: int = 3,
    seed: int = 0,
    s_expected: Optional[float] = None,
    scale_min: float = 0.6,
    scale_max: float = 1.6,
    disable_scale_gate: bool = False,
    matching: str = "greedy",
    appearance_tiebreak: bool = True,
    vote_M: int = 8,
    vote_ratio_tol: float = 0.12,
    vote_max_hyp: int = 2000,
    vote_w_score2: float = 0.0,
    layout_prior: bool = False,
    layout_lambda: float = 0.25,
    layout_mode: str = "image",
):
    """
    Star-tracker style matcher: pairwise distance voting + seed-and-verify.
    Returns (s, R, t, matches, T_hat, app_sum) or None.
    """
    rng = np.random.default_rng(seed)
    T = np.asarray(template_xy, float)
    C = np.asarray(cand_xy, float)
    if cand_score2 is None:
        cand_score2 = np.zeros(len(C), float)
    cand_score2 = np.asarray(cand_score2, float)
    K, N = len(T), len(C)
    if N < min_inliers or K < min_inliers:
        return None

    # pairwise distances
    dT = np.full((K, K), np.nan)
    for i in range(K):
        for j in range(i + 1, K):
            dT[i, j] = dT[j, i] = float(np.linalg.norm(T[i] - T[j]))
    dC = np.full((N, N), np.nan)
    for a in range(N):
        for b in range(a + 1, N):
            dC[a, b] = dC[b, a] = float(np.linalg.norm(C[a] - C[b]))

    # voting matrix
    V = np.zeros((K, N), dtype=float)
    for i in range(K):
        for j in range(i + 1, K):
            dt = dT[i, j]
            if dt < 1e-6:
                continue
            scales_all = []
            for a in range(N):
                for b in range(a + 1, N):
                    dc = dC[a, b]
                    if dc < 1e-6:
                        continue
                    s_hat = dc / dt
                    if (not disable_scale_gate) and (s_expected is not None):
                        if not (scale_min * s_expected <= s_hat <= scale_max * s_expected):
                            continue
                    scales_all.append(s_hat)
            if not scales_all:
                continue
            s_ref = s_expected if (s_expected is not None and not disable_scale_gate) else float(np.median(scales_all))
            for a in range(N):
                for b in range(a + 1, N):
                    dc = dC[a, b]
                    if dc < 1e-6:
                        continue
                    s_hat = dc / dt
                    if abs(math.log(s_hat / s_ref)) <= vote_ratio_tol:
                        V[i, a] += 1
                        V[j, b] += 1
                        V[i, b] += 1
                        V[j, a] += 1

    # shortlist per template
    shortlist = []
    for i in range(K):
        idx = list(range(N))
        idx.sort(key=lambda c: (V[i, c], cand_score2[c]), reverse=True)
        shortlist.append(idx[:vote_M])

    # generate seeds (2-point)
    seeds = []
    for i1 in range(K):
        for i2 in range(i1 + 1, K):
            for c1 in shortlist[i1]:
                for c2 in shortlist[i2]:
                    if c1 == c2:
                        continue
                    score = V[i1, c1] + V[i2, c2] + vote_w_score2 * (cand_score2[c1] + cand_score2[c2])
                    seeds.append((score, (i1, i2, c1, c2)))
    seeds.sort(key=lambda x: x[0], reverse=True)
    seeds = seeds[:vote_max_hyp]

    best = None
    best_inl = -1
    best_err = float("inf")
    best_app = -float("inf")

    for _, (i1, i2, c1, c2) in seeds:
        A = T[[i1, i2]]
        B = C[[c1, c2]]
        if np.linalg.norm(A[0] - A[1]) < 1e-6 or np.linalg.norm(B[0] - B[1]) < 1e-6:
            continue
        try:
            s, R, t = estimate_similarity(A, B)
        except np.linalg.LinAlgError:
            continue
        if (not disable_scale_gate) and (s_expected is not None):
            if not (scale_min * s_expected <= s <= scale_max * s_expected):
                continue

        T_hat = apply_similarity(T, s, R, t)
        if matching == "greedy":
            matches = match_template_to_candidates_greedy(T_hat, C, eps=eps)
        else:
            matches = match_template_to_candidates_hungarian(T_hat, C, eps=eps)
        inl = len(matches)
        if inl < min_inliers:
            continue
        err = float(np.mean([m[2] for m in matches])) if matches else float("inf")
        err_eff = err
        if layout_prior and matches is not None and len(matches) >= 5:
            pts_g = points_g_order_from_matches(matches, C, T_TO_G)
            if pts_g is not None:
                lp = layout_penalty_g0_top2(pts_g, mode=layout_mode)
                err_eff = err + float(layout_lambda) * lp
        app = float(sum(cand_score2[m[1]] for m in matches)) if cand_score2 is not None else 0.0

        better = (inl > best_inl) or \
                 (appearance_tiebreak and inl == best_inl and app > best_app) or \
                 (not appearance_tiebreak and inl == best_inl and err_eff < best_err) or \
                 (appearance_tiebreak and inl == best_inl and app == best_app and err_eff < best_err)
        if better:
            best_inl, best_err, best_app = inl, err_eff, app
            best = (s, R, t, matches, T_hat, app)

    return best if best_inl >= min_inliers else None

def ransac_constellation(
    template_xy: np.ndarray,
    cand_xy: np.ndarray,
    cand_score2: Optional[np.ndarray] = None,
    min_k: int = 3,
    n_iters: int = 4000,
    eps: float = 6.0,
    seed: int = 0,
    s_expected: Optional[float] = None,
    scale_min: float = 0.6,
    scale_max: float = 1.6,
    disable_scale_gate: bool = False,
    matching: str = "greedy",
    appearance_tiebreak: bool = True,
    layout_prior: bool = False,
    layout_lambda: float = 0.25,
    layout_mode: str = "image",
):
    """
    RANSAC over similarity transforms:
    - Sample min_k template points and min_k candidates
    - Fit similarity
    - Count inliers by nearest-neighbor match within eps
    """
    rng = np.random.default_rng(seed)
    T = np.asarray(template_xy, float)
    C = np.asarray(cand_xy, float)
    M, N = len(T), len(C)

    if N < min_k:
        return None

    best = None
    best_inliers = -1
    best_err = float("inf")
    best_app = -float("inf")
    T_subs = list(itertools.combinations(range(M), min_k))

    for _ in range(n_iters):
        t_sub = np.array(T_subs[rng.integers(len(T_subs))])
        c_sub = rng.choice(N, size=min_k, replace=False)

        A = T[t_sub]
        B = C[c_sub]

        try:
            s, R, t = estimate_similarity(A, B)
        except np.linalg.LinAlgError:
            continue

        # Reject hypotheses whose scale is implausible relative to labeled examples
        if (not disable_scale_gate) and (s_expected is not None):
            if not (scale_min * s_expected <= s <= scale_max * s_expected):
                continue

        T_hat = apply_similarity(T, s, R, t)
        if matching == "greedy":
            matches = match_template_to_candidates_greedy(T_hat, C, eps=eps)
        else:
            matches = match_template_to_candidates_hungarian(T_hat, C, eps=eps)
        inliers = len(matches)
        if inliers == 0:
            continue
        err = float(np.mean([m[2] for m in matches]))
        err_eff = err
        if layout_prior and matches is not None and len(matches) >= 5:
            pts_g = points_g_order_from_matches(matches, C, T_TO_G)
            if pts_g is not None:
                lp = layout_penalty_g0_top2(pts_g, mode=layout_mode)
                err_eff = err + float(layout_lambda) * lp
        app_sum = float(sum(cand_score2[m[1]] for m in matches)) if cand_score2 is not None else 0.0

        if appearance_tiebreak:
            better = (
                (inliers > best_inliers) or
                (inliers == best_inliers and app_sum > best_app) or
                (inliers == best_inliers and app_sum == best_app and err_eff < best_err)
            )
        else:
            better = (inliers > best_inliers) or (inliers == best_inliers and err_eff < best_err)

        if better:
            best_inliers = inliers
            best_err = err_eff
            best_app = app_sum
            best = (s, R, t, matches, T_hat, app_sum)

    return best


# ----------------------------
# Evaluation (optional)
# ----------------------------
def load_labels_json(label_path: Path) -> Dict[str, Any]:
    """Load the label JSON mapping filename -> annotation dict."""
    with open(label_path, "r", encoding="utf-8") as f:
        return json.load(f)


def gt_glints_for_image(labels: Dict[str, Any], filename: str) -> Optional[np.ndarray]:
    """
    Returns GT glints as (5,2) float array with NaNs where missing, or None if filename not in labels.
    label.txt uses -1 for missing.
    """
    if filename not in labels:
        return None
    info = labels[filename]
    cr = info.get("CornealReflectionLocations", {})
    xs = cr.get("CornealX", [-1, -1, -1, -1, -1])
    ys = cr.get("CornealY", [-1, -1, -1, -1, -1])
    P = np.full((5, 2), np.nan, dtype=float)
    for i in range(5):
        x, y = float(xs[i]), float(ys[i])
        if x >= 0 and y >= 0:
            P[i] = (x, y)
    return P


def evaluate_matches(
    gt_xy: np.ndarray,
    cand_xy: np.ndarray,
    matches: Optional[List[Tuple[int, int, float]]],
    match_tol: float,
) -> Dict[str, Any]:
    """
    Compute per-glint and per-image metrics.

    Definitions:
    - A GT glint i is "present" if gt_xy[i] is not NaN.
    - A predicted glint i exists if template idx i appears in matches.
      Its predicted location is the matched candidate location.
    - A glint i is "correct" if present and predicted and distance(pred, gt) <= match_tol.

    Returns a dict with:
      present_count, pred_count, correct_count,
      match_accuracy = correct/present,
      precision = correct/max(pred_count,1),
      loc_err_mean/median over present & predicted (NaNs ignored),
      per_glint: list of dicts for i=0..4
    """
    out: Dict[str, Any] = {}
    gt = np.asarray(gt_xy, float)
    pred = np.full((5, 2), np.nan, dtype=float)

    if matches is not None:
        for ti, ci, _ in matches:
            if 0 <= ti < 5 and 0 <= ci < len(cand_xy):
                pred[ti] = cand_xy[ci]

    present = ~np.isnan(gt[:, 0])
    predicted = ~np.isnan(pred[:, 0])

    present_count = int(present.sum())
    pred_count = int(predicted.sum())

    dists = np.full(5, np.nan, dtype=float)
    for i in range(5):
        if present[i] and predicted[i]:
            dists[i] = float(np.linalg.norm(pred[i] - gt[i]))

    correct = (present & predicted & (dists <= match_tol))
    correct_count = int(correct.sum())

    out["present_count"] = present_count
    out["pred_count"] = pred_count
    out["correct_count"] = correct_count
    out["match_accuracy"] = float(correct_count / present_count) if present_count > 0 else float("nan")
    out["precision"] = float(correct_count / pred_count) if pred_count > 0 else float("nan")

    # localization error over all present+predicted glints (not only "correct")
    valid = present & predicted
    if valid.any():
        out["loc_err_mean"] = float(np.nanmean(dists[valid]))
        out["loc_err_median"] = float(np.nanmedian(dists[valid]))
    else:
        out["loc_err_mean"] = float("nan")
        out["loc_err_median"] = float("nan")

    per_glint = []
    for i in range(5):
        per_glint.append(
            dict(
                glint=i,
                present=bool(present[i]),
                predicted=bool(predicted[i]),
                correct=bool(correct[i]),
                err_px=(None if np.isnan(dists[i]) else float(dists[i])),
                gt_x=(None if np.isnan(gt[i, 0]) else float(gt[i, 0])),
                gt_y=(None if np.isnan(gt[i, 1]) else float(gt[i, 1])),
                pred_x=(None if np.isnan(pred[i, 0]) else float(pred[i, 0])),
                pred_y=(None if np.isnan(pred[i, 1]) else float(pred[i, 1])),
            )
        )
    out["per_glint"] = per_glint
    return out


def _best_bipartite_match_small(cost: np.ndarray) -> Tuple[List[Tuple[int, int]], float]:
    """Exact tiny assignment via brute force.

    Parameters
    ----------
    cost : (m,n) array
        Pairwise costs.

    Returns
    -------
    pairs : list of (row_i, col_j)
        One-to-one pairs with k=min(m,n).
    total_cost : float
        Sum of costs over pairs.

    Notes
    -----
    We do this brute-force because m,n <= 5 here; it's fast and avoids scipy.
    """
    cost = np.asarray(cost, float)
    m, n = cost.shape
    if m == 0 or n == 0:
        return [], 0.0

    k = min(m, n)

    best_pairs: List[Tuple[int, int]] = []
    best_total = float("inf")

    if n >= m:
        # assign each row to a unique col (choose m cols, permute)
        cols = range(n)
        for col_subset in itertools.combinations(cols, m):
            for perm in itertools.permutations(col_subset):
                total = 0.0
                for i, j in enumerate(perm):
                    total += float(cost[i, j])
                if total < best_total:
                    best_total = total
                    best_pairs = [(i, int(perm[i])) for i in range(m)]
    else:
        # more rows than cols: choose k rows to match all cols (permute cols)
        rows = range(m)
        cols = list(range(n))
        for row_subset in itertools.combinations(rows, k):
            for perm in itertools.permutations(cols):
                total = 0.0
                for ii, r in enumerate(row_subset):
                    total += float(cost[r, perm[ii]])
                if total < best_total:
                    best_total = total
                    best_pairs = [(int(row_subset[ii]), int(perm[ii])) for ii in range(k)]

    return best_pairs, float(best_total)


def evaluate_identity_free(
    gt_xy: np.ndarray,
    pred_xy: np.ndarray,
    match_tol: float,
) -> Dict[str, Any]:
    """Identity-free evaluation (template index ignored).

    This answers: "Did we find the 5 glints somewhere near the GT glints?" without
    caring which glint is which.

    - GT glints: (5,2) with NaNs for missing.
    - Pred glints: (K,2) (typically K=len(matches)).
    - We solve a tiny assignment between present GT points and predictions.

    Outputs
    -------
    present_count, pred_count, correct_count
    match_accuracy = correct/present
    loc_err_mean/median over assigned pairs
    loc_err_mean_correct/median_correct over correct pairs only
    """
    gt = np.asarray(gt_xy, float)
    pred = np.asarray(pred_xy, float)

    present_mask = np.isfinite(gt[:, 0])
    gt_present = gt[present_mask]
    present_count = int(gt_present.shape[0])
    pred_count = int(pred.shape[0])

    out: Dict[str, Any] = {
        "present_count": present_count,
        "pred_count": pred_count,
        "correct_count": 0,
        "match_accuracy": float("nan"),
        "loc_err_mean": float("nan"),
        "loc_err_median": float("nan"),
        "loc_err_mean_correct": float("nan"),
        "loc_err_median_correct": float("nan"),
    }

    if present_count == 0:
        out["match_accuracy"] = float("nan")
        return out
    if pred_count == 0:
        out["match_accuracy"] = 0.0
        return out

    # cost matrix (present_count, pred_count)
    cost = np.sqrt(((gt_present[:, None, :] - pred[None, :, :]) ** 2).sum(axis=2))
    pairs, _ = _best_bipartite_match_small(cost)

    if not pairs:
        out["match_accuracy"] = 0.0
        return out

    dists = np.array([float(cost[i, j]) for (i, j) in pairs], dtype=float)
    correct_mask = dists <= match_tol
    correct_count = int(correct_mask.sum())

    out["correct_count"] = correct_count
    out["match_accuracy"] = float(correct_count / present_count)
    out["loc_err_mean"] = float(np.mean(dists))
    out["loc_err_median"] = float(np.median(dists))

    if correct_count > 0:
        out["loc_err_mean_correct"] = float(np.mean(dists[correct_mask]))
        out["loc_err_median_correct"] = float(np.median(dists[correct_mask]))

    return out


def _safe_import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
        return plt
    except Exception as e:
        warnings.warn(f"Skipping metric visualizations (matplotlib import failed): {e}")
        return None


def generate_metric_visuals(
    out_dir: Path,
    per_glint_counts: List[Dict[str, Any]],
    all_errs: List[float],
    idf_assigned_errs: List[float],
    per_image_rows: List[Dict[str, Any]],
) -> None:
    """
    Create quick-look plots for metrics:
    - Per-glint accuracy/precision bars
    - Per-glint localization error boxplots
    - Histograms of localization errors (overall and identity-free)
    - Per-image accuracy vs error scatter
    """
    plt = _safe_import_matplotlib()
    if plt is None:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- per-glint accuracy / precision
    glints = list(range(len(per_glint_counts)))
    acc = []
    prec = []
    for pc in per_glint_counts:
        p = pc["present"]
        pr = pc["pred"]
        c = pc["correct"]
        acc.append(c / p if p > 0 else np.nan)
        prec.append(c / pr if pr > 0 else np.nan)
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(np.array(glints) - width / 2, acc, width, label="accuracy", color="#4e79a7")
    ax.bar(np.array(glints) + width / 2, prec, width, label="precision", color="#f28e2b")
    ax.set_xticks(glints)
    ax.set_xlabel("Glint index (G0–G4)")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Per-glint accuracy / precision")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "metrics_per_glint_bar.png", dpi=200)
    plt.close(fig)

    # ---- per-glint localization error boxplot
    err_data = [pc["errs"] for pc in per_glint_counts]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(err_data, positions=glints, widths=0.6, showfliers=False)
    ax.set_xticks(glints)
    ax.set_xlabel("Glint index (G0–G4)")
    ax.set_ylabel("Localization error (px)")
    ax.set_title("Per-glint localization error (matched pairs)")
    fig.tight_layout()
    fig.savefig(out_dir / "metrics_per_glint_loc_error_box.png", dpi=200)
    plt.close(fig)

    # ---- histograms of errors
    if all_errs:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(all_errs, bins=30, color="#59a14f", alpha=0.8)
        ax.set_xlabel("Localization error (px)")
        ax.set_ylabel("Count")
        ax.set_title("Localization error histogram (matched glints)")
        fig.tight_layout()
        fig.savefig(out_dir / "metrics_loc_error_hist.png", dpi=200)
        plt.close(fig)

    if idf_assigned_errs:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(idf_assigned_errs, bins=30, color="#e15759", alpha=0.8)
        ax.set_xlabel("Identity-free error (px)")
        ax.set_ylabel("Count")
        ax.set_title("Identity-free localization error histogram")
        fig.tight_layout()
        fig.savefig(out_dir / "metrics_loc_error_idf_hist.png", dpi=200)
        plt.close(fig)

    # ---- per-image accuracy vs error scatter (quick leaderboard view)
    if per_image_rows:
        xs = []
        ys = []
        labels = []
        for r in per_image_rows:
            acc_val = r.get("match_accuracy")
            err_val = r.get("loc_err_mean")
            if acc_val is None or err_val is None:
                continue
            xs.append(acc_val)
            ys.append(err_val)
            labels.append(r["filename"])
        if xs:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(xs, ys, c="#af7aa1", alpha=0.8)
            ax.set_xlabel("Match accuracy")
            ax.set_ylabel("Mean localization error (px)")
            ax.set_title("Per-image accuracy vs error")
            for x, y, lab in zip(xs, ys, labels):
                if len(labels) <= 20:  # avoid clutter on large sets
                    ax.annotate(lab, (x, y), fontsize=7, alpha=0.7)
            fig.tight_layout()
            fig.savefig(out_dir / "metrics_per_image_scatter.png", dpi=200)
            plt.close(fig)


def generate_subject_bubble(per_subject_rows: List[Dict[str, Any]], out_dir: Path) -> None:
    """
    Bubble chart: x=match_accuracy, y=mean localization error, bubble size=#images.
    """
    plt = _safe_import_matplotlib()
    if plt is None or not per_subject_rows:
        return

    rows = [r for r in per_subject_rows if r.get("match_accuracy") is not None]
    if not rows:
        return

    # Prepare data
    acc = []
    err = []
    imgs = []
    labels = []
    for r in rows:
        try:
            a = float(r["match_accuracy"])
        except Exception:
            continue
        e = r.get("loc_err_mean")
        try:
            e = float(e) if e not in (None, "") else None
        except Exception:
            e = None
        acc.append(a)
        err.append(e)
        imgs.append(max(1, int(float(r.get("images", 1)))))
        labels.append(r["subject"])

    if not acc:
        return

    max_imgs = max(imgs) if imgs else 1
    sizes = [80 + 320 * (n / max_imgs) for n in imgs]  # bubble size

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(acc, err, s=sizes, alpha=0.7, color="#4e79a7", edgecolor="k")
    ax.set_xlabel("Match accuracy")
    ax.set_ylabel("Mean localization error (px)")
    ax.set_title("Per-subject bubble chart (size = images)")
    for x, y, lab in zip(acc, err, labels):
        if y is None or not np.isfinite(y):
            continue
        ax.annotate(lab, (x, y), fontsize=8, alpha=0.8, ha="center", va="center")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / "metrics_subject_bubble.png", dpi=200)
    plt.close(fig)


def write_subject_report(per_image_rows: List[Dict[str, Any]], out_dir: Path) -> None:
    """
    Aggregate per-image metrics into per-subject metrics and write report_per_subject.csv.
    Subject ID is parsed as the prefix before the first underscore in filename (e.g., PX from PX_YY_pos.jpg).
    """
    if not per_image_rows:
        return

    def subj_from_fname(fname: str) -> str:
        return fname.split("_")[0]

    buckets: Dict[str, Dict[str, Any]] = {}
    for r in per_image_rows:
        sid = subj_from_fname(r["filename"])
        b = buckets.setdefault(
            sid,
            dict(
                images=0,
                candidates_sum=0.0,
                inliers_sum=0.0,
                ransac_errs=[],
                app_sums=[],
                present_total=0,
                predicted_total=0,
                correct_total=0,
                loc_err_mean_vals=[],
                loc_err_median_vals=[],
                idf_present_total=0,
                idf_pred_total=0,
                idf_correct_total=0,
                idf_loc_err_mean_vals=[],
                idf_loc_err_median_vals=[],
            ),
        )
        b["images"] += 1
        b["candidates_sum"] += r.get("candidates", 0) or 0
        b["inliers_sum"] += r.get("inliers", 0) or 0
        if r.get("ransac_err_px") is not None:
            b["ransac_errs"].append(float(r["ransac_err_px"]))
        if r.get("app_sum") is not None:
            b["app_sums"].append(float(r["app_sum"]))

        b["present_total"] += r.get("present", 0) or 0
        b["predicted_total"] += r.get("predicted", 0) or 0
        b["correct_total"] += r.get("correct", 0) or 0
        if r.get("loc_err_mean") is not None:
            b["loc_err_mean_vals"].append(float(r["loc_err_mean"]))
        if r.get("loc_err_median") is not None:
            b["loc_err_median_vals"].append(float(r["loc_err_median"]))

        b["idf_present_total"] += r.get("idf_present", 0) or 0
        b["idf_pred_total"] += r.get("idf_predicted", 0) or 0
        b["idf_correct_total"] += r.get("idf_correct", 0) or 0
        if r.get("idf_loc_err_mean") is not None:
            b["idf_loc_err_mean_vals"].append(float(r["idf_loc_err_mean"]))
        if r.get("idf_loc_err_median") is not None:
            b["idf_loc_err_median_vals"].append(float(r["idf_loc_err_median"]))

    rows_out = []
    for sid, b in buckets.items():
        present_total = b["present_total"]
        predicted_total = b["predicted_total"]
        correct_total = b["correct_total"]
        idf_present_total = b["idf_present_total"]
        idf_pred_total = b["idf_pred_total"]
        idf_correct_total = b["idf_correct_total"]

        rows_out.append(
            dict(
                subject=sid,
                images=b["images"],
                candidates_mean=b["candidates_sum"] / b["images"],
                inliers_mean=b["inliers_sum"] / b["images"],
                ransac_err_mean=(np.mean(b["ransac_errs"]) if b["ransac_errs"] else None),
                app_sum_mean=(np.mean(b["app_sums"]) if b["app_sums"] else None),
                present_total=present_total,
                predicted_total=predicted_total,
                correct_total=correct_total,
                match_accuracy=(correct_total / present_total if present_total > 0 else None),
                precision=(correct_total / predicted_total if predicted_total > 0 else None),
                loc_err_mean=(float(np.mean(b["loc_err_mean_vals"])) if b["loc_err_mean_vals"] else None),
                loc_err_median=(float(np.median(b["loc_err_median_vals"])) if b["loc_err_median_vals"] else None),
                idf_present_total=idf_present_total,
                idf_predicted_total=idf_pred_total,
                idf_correct_total=idf_correct_total,
                idf_match_accuracy=(idf_correct_total / idf_present_total if idf_present_total > 0 else None),
                idf_loc_err_mean=(float(np.mean(b["idf_loc_err_mean_vals"])) if b["idf_loc_err_mean_vals"] else None),
                idf_loc_err_median=(float(np.median(b["idf_loc_err_median_vals"])) if b["idf_loc_err_median_vals"] else None),
            )
        )

    csv_path = out_dir / "report_per_subject.csv"
    if rows_out:
        import csv

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = list(rows_out[0].keys())
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows_out)
        generate_subject_bubble(rows_out, out_dir)


def subject_from_filename(filename: str) -> str:
    if "_" in filename:
        return filename.split("_")[0]
    return "unknown"


def filter_candidates_roi(
    cand_xy: np.ndarray,
    cand_score2: np.ndarray,
    img_shape: Tuple[int, int],
    mode: str,
    border_frac: float,
    border_px: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Lightweight ROI gating to drop implausible candidates (e.g., eyelid/skin glare).
    mode == "border": keep only points inside an inner rectangle with given margin.
    Returns filtered cand_xy, cand_score2, and the margin (px) used.
    """
    if mode == "none":
        return cand_xy, cand_score2, 0

    cand_xy = np.asarray(cand_xy, float)
    cand_score2 = np.asarray(cand_score2, float)
    if cand_xy.size == 0:
        return cand_xy.reshape(0, 2), cand_score2.reshape(0), 0
    if cand_xy.ndim != 2 or cand_xy.shape[1] != 2:
        cand_xy = cand_xy.reshape(-1, 2)

    H, W = img_shape
    margin = border_px if border_px is not None else int(border_frac * min(H, W))
    if margin <= 0:
        return cand_xy, cand_score2, 0

    x = cand_xy[:, 0]
    y = cand_xy[:, 1]
    mask = (x >= margin) & (x < W - margin) & (y >= margin) & (y < H - margin)
    return cand_xy[mask], cand_score2[mask], margin


def integral_image(gray: np.ndarray) -> np.ndarray:
    """Integral image with zero padding (shape H+1 x W+1)."""
    return cv2.integral(gray)


def integral_box_sum(ii: np.ndarray, size: int) -> np.ndarray:
    """Sum over all size x size boxes using integral image; returns map of shape (H-size+1, W-size+1)."""
    s = int(size)
    A = ii[s:, s:]
    B = ii[:-s, s:]
    C = ii[s:, :-s]
    D = ii[:-s, :-s]
    return A - B - C + D


def estimate_pupil_center(gray: np.ndarray, dark_thresh: int, min_area: int):
    """
    Estimate pupil center and radius from a grayscale eye image.
    Returns (cx, cy, radius) or None if detection fails.
    """
    mask = (gray < dark_thresh).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return None

    best_idx = None
    best_area = float(min_area - 1)
    for i in range(1, num_labels):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area and area > best_area:
            best_idx = i
            best_area = area
    if best_idx is None:
        return None

    cx, cy = centroids[best_idx]
    radius = math.sqrt(best_area / math.pi)
    return float(cx), float(cy), float(radius)


def load_pupil_npz(path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load pupil NPZ and return mapping: filename -> {"center": (cx, cy), "radius": r}.
    Expected NPZ keys: filenames, ellipses (dict with centroid, major_axis_length, minor_axis_length).
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Pupil NPZ not found: {p}")
    data = np.load(p, allow_pickle=True)
    filenames = data.get("filenames", None)
    ellipses = data.get("ellipses", None)
    if filenames is None or ellipses is None:
        raise ValueError("Pupil NPZ must contain 'filenames' and 'ellipses'")
    if len(filenames) != len(ellipses):
        raise ValueError("Pupil NPZ 'filenames' and 'ellipses' length mismatch")

    out: Dict[str, Dict[str, Any]] = {}
    for name, ell in zip(filenames, ellipses):
        if name is None:
            continue
        fname = Path(str(name)).name
        if not isinstance(ell, dict):
            continue
        centroid = ell.get("centroid", None)
        if not isinstance(centroid, (list, tuple)) or len(centroid) < 2:
            continue
        try:
            cx = float(centroid[0])
            cy = float(centroid[1])
        except (TypeError, ValueError):
            continue
        maj = ell.get("major_axis_length", None)
        minr = ell.get("minor_axis_length", None)
        r = None
        try:
            if maj is not None and minr is not None:
                r = 0.25 * (float(maj) + float(minr))
        except (TypeError, ValueError):
            r = None
        out[fname] = {"center": (cx, cy), "radius": r, "ellipse": ell}
    return out


def pupil_from_labels(
    labels: Dict[str, Any],
    filename: str,
    W: int,
    H: int,
    axis_mode: str,
) -> Optional[Tuple[float, float, float]]:
    entry = labels.get(filename, None)
    if entry is None:
        return None
    min_dim = float(min(W, H)) if W > 0 and H > 0 else 0.0

    def _safe_float(val, default=-1.0):
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    def _scale_center(cx: float, cy: float) -> Tuple[float, float]:
        if 0 <= cx <= 1.5 and 0 <= cy <= 1.5 and W > 0 and H > 0:
            return cx * W, cy * H
        return cx, cy

    def _scale_axis(ax: float, ay: float) -> Tuple[float, float]:
        if 0 < ax <= 1.5 and 0 < ay <= 1.5 and W > 0 and H > 0:
            return ax * W, ay * H
        return ax, ay

    def _axis_to_radius(ax: float, ay: float) -> Optional[float]:
        if ax <= 0 or ay <= 0:
            return None
        if axis_mode == "radius":
            pr = 0.5 * (ax + ay)
        elif axis_mode == "diameter":
            pr = 0.25 * (ax + ay)
        else:
            if min_dim > 0:
                if ax > 0.75 * min_dim or ay > 0.75 * min_dim:
                    pr = 0.25 * (ax + ay)
                elif ax < 0.5 * min_dim and ay < 0.5 * min_dim:
                    pr = 0.5 * (ax + ay)
                else:
                    pr = 0.5 * (ax + ay)
            else:
                pr = 0.5 * (ax + ay)
        if min_dim > 0:
            pr = min(pr, 0.5 * min_dim)
        return pr

    center = entry.get("PupilCenter", None)
    axis = entry.get("PupilEllipseAxis", None)
    cx = cy = -1.0
    ax = ay = -1.0

    if isinstance(center, dict):
        cx = _safe_float(center.get("PupilX", center.get("X", -1)))
        cy = _safe_float(center.get("PupilY", center.get("Y", -1)))
    elif isinstance(center, (list, tuple)) and len(center) >= 2:
        cx = _safe_float(center[0])
        cy = _safe_float(center[1])
    if cx < 0 or cy < 0:
        cx = _safe_float(entry.get("PupilX", cx))
        cy = _safe_float(entry.get("PupilY", cy))
    if cx >= 0 and cy >= 0:
        cx, cy = _scale_center(cx, cy)

    if isinstance(axis, dict):
        ax = _safe_float(axis.get("X", axis.get("Width", -1)))
        ay = _safe_float(axis.get("Y", axis.get("Height", -1)))
    elif isinstance(axis, (list, tuple)) and len(axis) >= 2:
        ax = _safe_float(axis[0])
        ay = _safe_float(axis[1])
    if ax > 0 and ay > 0:
        ax, ay = _scale_axis(ax, ay)
        if min_dim > 0:
            ax = min(ax, min_dim)
            ay = min(ay, min_dim)

    pts = entry.get("PupilBoundaryPoints", None)
    pts_arr = None
    if isinstance(pts, list) and len(pts) >= 3:
        if all(isinstance(p, dict) for p in pts):
            arr = []
            for p in pts:
                x = _safe_float(p.get("X", p.get("x", p.get("PupilX", None))), None)
                y = _safe_float(p.get("Y", p.get("y", p.get("PupilY", None))), None)
                if x is None or y is None:
                    continue
                arr.append((x, y))
            if len(arr) >= 3:
                pts_arr = np.asarray(arr, dtype=float)
        else:
            try:
                pts_arr = np.asarray(pts, dtype=float)
                if pts_arr.ndim != 2 or pts_arr.shape[1] != 2:
                    pts_arr = None
            except (TypeError, ValueError):
                pts_arr = None
    if pts_arr is not None and pts_arr.size > 0:
        if np.all(pts_arr >= 0) and np.max(pts_arr[:, 0]) <= 1.5 and np.max(pts_arr[:, 1]) <= 1.5:
            pts_arr[:, 0] *= W
            pts_arr[:, 1] *= H

    if cx >= 0 and cy >= 0 and ax > 0 and ay > 0:
        pr = _axis_to_radius(ax, ay)
        if pr is not None and pr > 0:
            return float(cx), float(cy), float(pr)
    if ax > 0 and ay > 0 and pts_arr is not None and (cx < 0 or cy < 0):
        cx = float(np.mean(pts_arr[:, 0]))
        cy = float(np.mean(pts_arr[:, 1]))
        pr = _axis_to_radius(ax, ay)
        if pr is not None and pr > 0:
            return float(cx), float(cy), float(pr)
    if pts_arr is not None and pts_arr.size > 0:
        if cx < 0 or cy < 0:
            cx = float(np.mean(pts_arr[:, 0]))
            cy = float(np.mean(pts_arr[:, 1]))
        d = np.sqrt((pts_arr[:, 0] - cx) ** 2 + (pts_arr[:, 1] - cy) ** 2)
        if d.size > 0:
            pr = float(np.median(d))
            if pr > 0:
                if min_dim > 0:
                    pr = min(pr, 0.5 * min_dim)
                return float(cx), float(cy), float(pr)
    return None


def swirski_coarse_center(gray: np.ndarray, radii: List[int]) -> Optional[Tuple[float, float, float]]:
    """Stage 1: center-surround search over radii using integral image (Swirski et al. 2012)."""
    if gray.size == 0:
        return None
    ii = integral_image(gray.astype(np.float32))
    H, W = gray.shape
    best = None
    for r in radii:
        if r <= 0:
            continue
        s_inner = int(r)
        s_outer = int(3 * r)
        if s_outer >= min(H, W):
            continue
        # sums for outer and inner squares
        outer_sum = integral_box_sum(ii, s_outer)
        inner_sum = integral_box_sum(ii, s_inner)
        # align inner to outer centers
        offset = (s_outer - s_inner) // 2
        oh, ow = outer_sum.shape
        inner_aligned = inner_sum[offset:offset + oh, offset:offset + ow]
        if inner_aligned.shape != outer_sum.shape:
            continue
        area_outer = float(s_outer * s_outer)
        area_inner = float(s_inner * s_inner)
        ring_mean = (outer_sum - inner_aligned) / max(area_outer - area_inner, 1e-6)
        inner_mean = inner_aligned / max(area_inner, 1e-6)
        resp = ring_mean - inner_mean
        max_loc = np.unravel_index(np.argmax(resp), resp.shape)
        y0, x0 = max_loc
        cx = x0 + (s_outer - 1) * 0.5
        cy = y0 + (s_outer - 1) * 0.5
        val = resp[max_loc]
        if best is None or val > best[0]:
            best = (float(val), float(cx), float(cy), float(r))
    if best is None:
        return None
    _, cx, cy, r = best
    return cx, cy, r


def _kmeans_hist_2cluster(hist: np.ndarray) -> Tuple[float, float, int]:
    """Weighted 1D k-means (k=2) on a 256-bin histogram. Returns (c0, c1, thr_bin_dark_cluster_max)."""
    c0, c1 = 64.0, 192.0
    hist = hist.astype(np.float64)
    for _ in range(20):
        sum0 = sum1 = cnt0 = cnt1 = 1e-6
        for v, w in enumerate(hist):
            if w <= 0:
                continue
            if abs(v - c0) <= abs(v - c1):
                sum0 += v * w
                cnt0 += w
            else:
                sum1 += v * w
                cnt1 += w
        new0, new1 = sum0 / cnt0, sum1 / cnt1
        if max(abs(new0 - c0), abs(new1 - c1)) < 1e-3:
            break
        c0, c1 = new0, new1
    # determine dark cluster threshold (max bin assigned to darker center)
    thr = 0
    dark_is_0 = c0 <= c1
    dark_cent = c0 if dark_is_0 else c1
    for v, w in enumerate(hist):
        if w <= 0:
            continue
        d0 = abs(v - c0)
        d1 = abs(v - c1)
        assign0 = d0 <= d1
        if (assign0 and dark_is_0) or ((not assign0) and (not dark_is_0)):
            thr = v
    return c0, c1, thr


def swirski_kmeans_refine(gray: np.ndarray, cx: float, cy: float, r0: float) -> Optional[Tuple[float, float, float]]:
    """Stage 2: histogram k-means refinement around coarse center."""
    H, W = gray.shape
    half = int(max(4, round(1.5 * 3 * r0)))  # cover ~3r region with margin
    x0 = max(0, int(round(cx)) - half)
    y0 = max(0, int(round(cy)) - half)
    x1 = min(W, int(round(cx)) + half)
    y1 = min(H, int(round(cy)) + half)
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    hist = cv2.calcHist([roi], [0], None, [256], [0, 256]).flatten()
    _, _, thr_bin = _kmeans_hist_2cluster(hist)
    pupil_mask = (roi <= thr_bin).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pupil_mask, connectivity=8)
    if num_labels <= 1:
        return None
    best_idx = None
    best_area = 0
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area > best_area:
            best_area = area
            best_idx = i
    if best_idx is None or best_area <= 0:
        return None
    cx_roi, cy_roi = centroids[best_idx]
    pr = math.sqrt(best_area / math.pi)
    return float(x0 + cx_roi), float(y0 + cy_roi), float(pr)


def pupil_is_plausible(cx: float, cy: float, pr: float, H: int, W: int) -> bool:
    if pr <= 0 or pr > 0.5 * min(H, W):
        return False
    if not (0 <= cx < W and 0 <= cy < H):
        return False
    if (cx - pr) < 0 or (cx + pr) >= W or (cy - pr) < 0 or (cy + pr) >= H:
        return False
    min_area_frac = 1e-5
    if (math.pi * pr * pr) < min_area_frac * (H * W):
        return False
    return True


def detect_pupil_center_for_frame(
    gray: np.ndarray,
    labels: Optional[Dict[str, Any]],
    filename: str,
    W: int,
    H: int,
    args,
    params: Dict[str, float],
    pupil_radii_raw: List[int],
    pupil_npz_map: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[Optional[float], Optional[float], Optional[float], bool, bool, str]:
    pupil_detected = False
    pupil_ok = False
    pupil_source_used = "none"

    if args.pupil_source != "none":
        if args.pupil_source == "labels" or (args.pupil_source == "auto" and labels is not None):
            pupil_source_used = "labels"
        elif args.pupil_source == "npz":
            pupil_source_used = "npz"
        else:
            pupil_source_used = args.pupil_method if args.pupil_source == "auto" else args.pupil_source

    pcx = pcy = pr = None
    if pupil_source_used == "npz":
        if pupil_npz_map is not None:
            entry = pupil_npz_map.get(filename, None)
            if entry is None:
                # try basename match
                entry = pupil_npz_map.get(Path(filename).name, None)
            if entry is not None:
                center = entry.get("center", None)
                if isinstance(center, (list, tuple)) and len(center) >= 2:
                    pcx = float(center[0])
                    pcy = float(center[1])
                    pr = entry.get("radius", None)
                    pr = float(pr) if pr is not None else None
                    pupil_detected = True
    elif pupil_source_used == "labels":
        if labels is not None:
            pupil_lab = pupil_from_labels(labels, filename, W, H, args.pupil_axis_mode)
            if pupil_lab is not None:
                pcx, pcy, pr = pupil_lab
                pupil_detected = True
    else:
        if pupil_source_used == "naive":
            pupil = estimate_pupil_center(gray, args.pupil_dark_thresh, args.pupil_min_area)
            if pupil is not None:
                pcx, pcy, pr = pupil
                pupil_detected = True
        elif pupil_source_used == "swirski":
            radii_scaled = [max(1, int(round(r * params["s"]))) for r in pupil_radii_raw]
            coarse = swirski_coarse_center(gray, radii_scaled)
            if coarse is not None:
                pcx_c, pcy_c, pr_c = coarse
                fine = swirski_kmeans_refine(gray, pcx_c, pcy_c, pr_c)
                if fine is not None:
                    pcx, pcy, pr = fine
                    pupil_detected = True

    if pupil_detected and pcx is not None and pcy is not None and pr is not None:
        pupil_ok = pupil_is_plausible(pcx, pcy, pr, H, W)

    return pcx, pcy, pr, pupil_detected, pupil_ok, pupil_source_used


def gate_candidates_by_pupil(
    cand_xy: np.ndarray,
    pupil_center: Tuple[float, float],
    pupil_radius: float,
    rmin: float,
    rmax: float,
) -> np.ndarray:
    """
    Returns boolean mask keeping candidates within [rmin*r, rmax*r] of pupil_center.
    """
    cand_xy = np.asarray(cand_xy, float)
    if cand_xy.size == 0:
        return np.zeros(len(cand_xy), dtype=bool)
    if cand_xy.ndim != 2 or cand_xy.shape[1] != 2:
        cand_xy = cand_xy.reshape(-1, 2)
    dx = cand_xy[:, 0] - float(pupil_center[0])
    dy = cand_xy[:, 1] - float(pupil_center[1])
    dist = np.sqrt(dx * dx + dy * dy)
    return (dist >= rmin * pupil_radius) & (dist <= rmax * pupil_radius)


def hybrid_constellation_match(
    template_xy: np.ndarray,
    cand_xy: np.ndarray,
    cand_score2: Optional[np.ndarray] = None,
    min_k: int = 3,
    n_iters: int = 4000,
    eps: float = 6.0,
    seed: int = 0,
    s_expected: Optional[float] = None,
    scale_min: float = 0.6,
    scale_max: float = 1.6,
    disable_scale_gate: bool = False,
    matching: str = "greedy",
    appearance_tiebreak: bool = True,
    min_inliers: int = 3,
    vote_M: int = 8,
    vote_ratio_tol: float = 0.12,
    vote_max_hyp: int = 2000,
    vote_w_score2: float = 0.0,
    layout_prior: bool = False,
    layout_lambda: float = 0.25,
    layout_mode: str = "image",
):
    """
    Hybrid matcher: run RANSAC and star-tracker, then select the stronger result.
    Returns (s, R, t, matches, T_hat, app_sum) or None.
    """
    r_best = ransac_constellation(
        template_xy,
        cand_xy,
        cand_score2=cand_score2,
        min_k=min_k,
        n_iters=n_iters,
        eps=eps,
        seed=seed,
        s_expected=s_expected,
        scale_min=scale_min,
        scale_max=scale_max,
        disable_scale_gate=disable_scale_gate,
        matching=matching,
        appearance_tiebreak=appearance_tiebreak,
        layout_prior=layout_prior,
        layout_lambda=layout_lambda,
        layout_mode=layout_mode,
    )
    s_best = startracker_constellation(
        template_xy,
        cand_xy,
        cand_score2=cand_score2,
        eps=eps,
        min_inliers=min_inliers,
        seed=seed,
        s_expected=s_expected,
        scale_min=scale_min,
        scale_max=scale_max,
        disable_scale_gate=disable_scale_gate,
        matching=matching,
        appearance_tiebreak=appearance_tiebreak,
        vote_M=vote_M,
        vote_ratio_tol=vote_ratio_tol,
        vote_max_hyp=vote_max_hyp,
        vote_w_score2=vote_w_score2,
        layout_prior=layout_prior,
        layout_lambda=layout_lambda,
        layout_mode=layout_mode,
    )

    if r_best is None:
        return s_best
    if s_best is None:
        return r_best

    r_key = score_match_result(r_best, cand_score2, appearance_tiebreak, bank_select_metric="strict")
    s_key = score_match_result(s_best, cand_score2, appearance_tiebreak, bank_select_metric="strict")
    return r_best if r_key >= s_key else s_best


def run_matcher_for_template(
    template_xy: np.ndarray,
    cand_xy: np.ndarray,
    cand_score2: Optional[np.ndarray],
    args,
    s_expected: Optional[float],
    eps_eff: float,
    ratio_index: Optional[Dict[str, Any]] = None,
    cand_raw_count: Optional[int] = None,
):
    res = None
    if args.matcher == "ransac":
        res = ransac_constellation(
            template_xy,
            cand_xy,
            cand_score2=cand_score2,
            min_k=args.min_k,
            n_iters=args.iters,
            eps=eps_eff,
            seed=args.seed,
            s_expected=s_expected,
            scale_min=args.scale_min,
            scale_max=args.scale_max,
            disable_scale_gate=args.disable_scale_gate,
            matching=args.matching,
            appearance_tiebreak=args.appearance_tiebreak,
            layout_prior=args.layout_prior,
            layout_lambda=args.layout_lambda,
            layout_mode=args.layout_mode,
        )
    elif args.matcher == "star":
        res = startracker_constellation(
            template_xy,
            cand_xy,
            cand_score2=cand_score2,
            eps=eps_eff,
            min_inliers=args.min_inliers,
            seed=args.seed,
            s_expected=s_expected,
            scale_min=args.scale_min,
            scale_max=args.scale_max,
            disable_scale_gate=args.disable_scale_gate,
            matching=args.matching,
            appearance_tiebreak=args.appearance_tiebreak,
            vote_M=args.vote_M,
            vote_ratio_tol=args.vote_ratio_tol,
            vote_max_hyp=args.vote_max_hyp,
            vote_w_score2=args.vote_w_score2,
            layout_prior=args.layout_prior,
            layout_lambda=args.layout_lambda,
            layout_mode=args.layout_mode,
        )
    elif args.matcher == "sla":
        res = sla_pyramid_constellation(
            template_xy,
            cand_xy,
            cand_score2=cand_score2,
            eps=eps_eff,
            min_inliers=args.min_inliers,
            matching=args.matching,
            appearance_tiebreak=args.appearance_tiebreak,
            ratio_index=ratio_index,
            pivot_P=args.pivot_P,
            ratio_tol=args.ratio_tol,
            max_seeds=args.max_seeds,
            grow_resid_max=args.grow_resid_max,
            sla_w_seed_score2=args.sla_w_seed_score2,
            sla_w_seed_geom=args.sla_w_seed_geom,
            max_seeds_per_pivot=args.max_seeds_per_pivot,
            sla_adaptive_ratio_tol=args.sla_adaptive_ratio_tol,
            sla_ratio_tol_min=args.sla_ratio_tol_min,
            sla_ratio_tol_refN=args.sla_ratio_tol_refN,
            sla_scale_min=args.sla_scale_min,
            sla_scale_max=args.sla_scale_max,
            sla_g0_top2=args.sla_g0_top2,
            sla_semantic_prior=args.sla_semantic_prior,
            sla_semantic_lambda=args.sla_semantic_lambda,
            sla_semantic_mode=getattr(args, "sla_semantic_mode", "full"),
            sla_semantic_hard=args.sla_semantic_hard,
            sla_mirror_reject=args.sla_mirror_reject,
            sla_top2_margin=args.sla_top2_margin,
            sla_base_ratio_min=args.sla_base_ratio_min,
            sla_side_margin=args.sla_side_margin,
            sla_semantic_debug=args.sla_semantic_debug,
            sla_layout_prior=args.sla_layout_prior,
            sla_layout_lambda=args.sla_layout_lambda,
            sla_layout_mode=args.sla_layout_mode,
            sla_layout_debug=args.sla_layout_debug,
            sla_raw_count=cand_raw_count,
        )
    else:
        res = hybrid_constellation_match(
            template_xy,
            cand_xy,
            cand_score2=cand_score2,
            min_k=args.min_k,
            n_iters=args.iters,
            eps=eps_eff,
            seed=args.seed,
            s_expected=s_expected,
            scale_min=args.scale_min,
            scale_max=args.scale_max,
            disable_scale_gate=args.disable_scale_gate,
            matching=args.matching,
            appearance_tiebreak=args.appearance_tiebreak,
            min_inliers=args.min_inliers,
            vote_M=args.vote_M,
            vote_ratio_tol=args.vote_ratio_tol,
            vote_max_hyp=args.vote_max_hyp,
            vote_w_score2=args.vote_w_score2,
            layout_prior=args.layout_prior,
            layout_lambda=args.layout_lambda,
            layout_mode=args.layout_mode,
        )
    if not args.post_id_resolve or res is None:
        return res
    if template_xy.shape[0] != 5:
        return res
    # post-match identity resolution (only when full 5-point match is present)
    s, R, t, matches, T_hat, app_sum = res
    if matches is None or len(matches) != 5:
        return res
    t_idx = [m[0] for m in matches]
    if sorted(t_idx) != [0, 1, 2, 3, 4]:
        return res
    obs_xy = np.zeros((5, 2), dtype=float)
    obs_cand_idx = np.zeros(5, dtype=int)
    for ti, ci, _ in matches:
        obs_xy[int(ti)] = cand_xy[int(ci)]
        obs_cand_idx[int(ti)] = int(ci)
    pred_xy = apply_similarity(template_xy, s, R, t)
    obs_score2 = None
    if cand_score2 is not None and len(cand_score2) >= 5:
        obs_score2 = cand_score2[obs_cand_idx]
    perm, cost = resolve_identity_permutation(
        template_xy,
        pred_xy,
        obs_xy,
        obs_score2,
        eps_eff,
        args.id_layout_mode,
        args.id_lambda,
        args.id_gamma,
        args.id_eta,
        args.id_tau,
        s_fit=s,
        R_fit=R,
        t_fit=t,
    )
    if perm is None:
        return res
    perm = list(perm)
    if perm == [0, 1, 2, 3, 4]:
        return res
    new_matches = []
    for g in range(5):
        oi = perm[g]
        ci = int(obs_cand_idx[oi])
        d = float(np.linalg.norm(pred_xy[g] - cand_xy[ci]))
        new_matches.append((int(g), ci, d))
    if cand_score2 is not None:
        app_sum = float(np.sum(cand_score2[[m[1] for m in new_matches]]))
    res = (s, R, t, new_matches, pred_xy, app_sum)

    global _ID_DEBUG_COUNT
    if args.id_debug and _ID_DEBUG_COUNT < 10:
        orig_map = [int(obs_cand_idx[i]) for i in range(5)]
        new_map = [int(obs_cand_idx[perm[g]]) for g in range(5)]
        print(f"[id_resolve] img#{_ID_DEBUG_COUNT} orig={orig_map} new={new_map} "
              f"geom={cost['geom']:.2f} layout={cost['layout']:.2f} "
              f"pair={cost['pair']:.4f} app={cost['app']:.2f} total={cost['total']:.2f}")
        _ID_DEBUG_COUNT += 1
    return res


def _self_test_star_vs_ransac(args):
    rng = np.random.default_rng(0)
    T = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.4, 0.3]], float)
    s_true = 2.5
    ang = 0.6
    R_true = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
    t_true = np.array([10.0, -4.0])
    C_true = s_true * (T @ R_true.T) + t_true
    keep = rng.choice(len(T), size=4, replace=False)
    C = C_true[keep] + rng.normal(0, 0.5, (4, 2))
    clutter = rng.normal(0, 15, (12, 2))
    cand_xy = np.vstack([C, clutter])
    cand_score2 = rng.normal(100, 10, len(cand_xy))
    eps = 3.0
    r_best = ransac_constellation(T, cand_xy, cand_score2=cand_score2, eps=eps,
                                  min_k=3, n_iters=800, seed=0,
                                  s_expected=s_true, scale_min=0.5, scale_max=2.0,
                                  disable_scale_gate=False, matching="hungarian",
                                  appearance_tiebreak=True)
    s_best = startracker_constellation(T, cand_xy, cand_score2=cand_score2, eps=eps,
                                       min_inliers=3, seed=0, s_expected=s_true,
                                       scale_min=0.5, scale_max=2.0, disable_scale_gate=False,
                                       matching="hungarian", appearance_tiebreak=True,
                                       vote_M=8, vote_ratio_tol=0.15, vote_max_hyp=500,
                                       vote_w_score2=0.0)

    def desc(name, res):
        if res is None:
            return f"{name}: None"
        s, R, t, m, _, app = res
        err = np.mean([x[2] for x in m]) if m else float("nan")
        return f"{name}: inl={len(m)} err={err:.2f} s={s:.2f} app={app:.1f}"

    print(desc("ransac", r_best))
    print(desc("star", s_best))


def scale_params_for_image(args, w: int, h: int) -> Dict[str, float]:
    """
    Compute per-image effective parameters scaled by image width vs ref_width.
    """
    if not args.auto_scale:
        return dict(
            s=1.0,
            kernel_eff=int(args.kernel),
            median_ksize_eff=int(args.median_ksize),
            eps_eff=float(args.eps),
            match_tol_eff=float(args.match_tol),
            min_area_eff=int(args.min_area),
            max_area_eff=int(args.max_area),
        )

    s = float(w) / float(args.ref_width)

    def make_odd(k: int) -> int:
        return k if k % 2 == 1 else k + 1

    kernel_raw = max(args.min_kernel, int(round(args.kernel * s)))
    kernel_eff = make_odd(kernel_raw)
    med_raw = max(args.min_kernel, int(round(args.median_ksize * s)))
    median_ksize_eff = make_odd(med_raw)

    eps_eff = float(args.eps * s)
    match_tol_eff = float(args.match_tol * s)
    min_area_eff = max(1, int(round(args.min_area * s * s)))
    max_area_eff = max(min_area_eff, int(round(args.max_area * s * s)))

    return dict(
        s=s,
        kernel_eff=kernel_eff,
        median_ksize_eff=median_ksize_eff,
        eps_eff=eps_eff,
        match_tol_eff=match_tol_eff,
        min_area_eff=min_area_eff,
        max_area_eff=max_area_eff,
    )

def remap_matches_template_to_gt(matches, t_to_g):
    """
    matches: list of (template_idx, cand_idx, dist)
    Returns matches remapped into GT index space:
        (gt_idx, cand_idx, dist)
    """
    if matches is None:
        return None
    out = []
    for t_idx, c_idx, d in matches:
        if t_idx is None:
            continue
        if 0 <= int(t_idx) < len(t_to_g):
            g_idx = int(t_to_g[int(t_idx)])
            out.append((g_idx, c_idx, d))
    return out



# ----------------------------
# Visualization
# ----------------------------
def draw_overlay(
    gray: np.ndarray,
    cand_xy: np.ndarray,
    T_hat: Optional[np.ndarray],
    matches: Optional[List[Tuple[int, int, float]]],
    title_text: str = "",
    gt_xy: Optional[np.ndarray] = None,
    match_tol: float = 10.0,
) -> np.ndarray:
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # candidates in blue
    for i, (x, y) in enumerate(cand_xy):
        cv2.circle(overlay, (int(round(x)), int(round(y))), 4, (255, 0, 0), 1)

    if T_hat is not None:
        # transformed template in green (what the model thinks the constellation should be)
        for ti, (x, y) in enumerate(T_hat):
            cv2.circle(overlay, (int(round(x)), int(round(y))), 8, (0, 255, 0), 2)
            cv2.putText(
                overlay,
                f"T{ti}",
                (int(round(x)) + 6, int(round(y)) + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

    if matches is not None and T_hat is not None:
        # match lines in yellow
        for ti, ci, d in matches:
            p = tuple(np.round(T_hat[ti]).astype(int))
            q = tuple(np.round(cand_xy[ci]).astype(int))
            cv2.line(overlay, p, q, (0, 255, 255), 2)
            mid = ((p[0] + q[0]) // 2, (p[1] + q[1]) // 2)
            cv2.putText(overlay, f"{d:.1f}", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

    # ground truth in magenta (if provided)
    if gt_xy is not None:
        gt_xy = np.asarray(gt_xy, float)
        for gi in range(5):
            if np.isfinite(gt_xy[gi, 0]):
                gx, gy = int(round(gt_xy[gi, 0])), int(round(gt_xy[gi, 1]))
                cv2.circle(overlay, (gx, gy), 7, (255, 0, 255), 2)
                cv2.putText(overlay, f"G{gi}", (gx + 6, gy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1, cv2.LINE_AA)

    if title_text:
        cv2.putText(overlay, title_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(overlay, title_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1, cv2.LINE_AA)

    return overlay


# ----------------------------
# Main
# ----------------------------
_MP_STATE: Dict[str, Any] = {}


def _mp_init(
    args_dict: Dict[str, Any],
    template: np.ndarray,
    bank_templates: Optional[List[np.ndarray]],
    ratio_index_bank: Optional[List[Dict[str, Any]]],
    ratio_index_single: Optional[Dict[str, Any]],
    d_expected: Optional[np.ndarray],
    s_expected: float,
    n_glints: int,
) -> None:
    args = argparse.Namespace(**args_dict)
    pupil_npz_map = None
    if getattr(args, "pupil_npz", None):
        pupil_npz_map = load_pupil_npz(args.pupil_npz)
    pupil_radii_raw = [int(x) for x in args.pupil_radii.split(",") if x.strip().isdigit()]
    if not pupil_radii_raw:
        pupil_radii_raw = [12, 16, 20, 24, 28, 32]
    _MP_STATE.clear()
    _MP_STATE.update(
        dict(
            args=args,
            template=template,
            bank_templates=bank_templates,
            ratio_index_bank=ratio_index_bank,
            ratio_index_single=ratio_index_single,
            d_expected=d_expected,
            s_expected=s_expected,
            n_glints=n_glints,
            pupil_npz_map=pupil_npz_map,
            pupil_radii_raw=pupil_radii_raw,
        )
    )


def _process_one_image_mp(fp_str: str) -> Optional[Dict[str, Any]]:
    state = _MP_STATE
    args = state["args"]
    fp = Path(fp_str)
    bgr = cv2.imread(str(fp), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    gray_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray_full.shape[:2]
    image_center = (0.5 * W, 0.5 * H)

    params = scale_params_for_image(args, w=gray_full.shape[1], h=gray_full.shape[0])
    pcx = pcy = pr = None
    pupil_detected = False
    pupil_ok = False
    pupil_source_used = "none"
    if args.pupil_roi and args.pupil_source != "none":
        pcx, pcy, pr, pupil_detected, pupil_ok, pupil_source_used = detect_pupil_center_for_frame(
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

    cand_xy_pass0, rows_pass0, cand_score2_pass0, cand_raw_pass0, cand_support_pass0 = detect_candidates_one_pass(
        gray, params, args, d_expected=state["d_expected"]
    )
    cand_xy_raw = cand_xy_pass0
    cand_score2_raw = cand_score2_pass0
    cand_support_raw = cand_support_pass0
    cand_raw_merged = int(cand_raw_pass0)

    cand_passes_used = 1
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
            cand_xy_i, rows_i, cand_score2_i, cand_raw_i, cand_support_i = detect_candidates_one_pass(
                gray, params, args, d_expected=state["d_expected"], percentile_override=perc, kernel_add=kernel_add
            )
            cand_xy_list.append(cand_xy_i)
            cand_score2_list.append(cand_score2_i)
            cand_support_list.append(cand_support_i)
            cand_xy_raw, cand_score2_raw, cand_support_raw = merge_candidates(
                cand_xy_list, cand_score2_list, cand_support_list, merge_eps=float(args.cand_merge_eps)
            )
            cand_raw_merged = int(len(cand_xy_raw))
            cand_passes_used = i + 1
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
    cand_xy, cand_score2, roi_margin = filter_candidates_roi(
        cand_xy, cand_score2, roi_shape, mode=args.roi_mode,
        border_frac=args.roi_border_frac, border_px=args.roi_border_px)
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
            pupil_mask = gate_candidates_by_pupil(
                cand_xy, (pcx_gate, pcy_gate), pr_gate, args.pupil_rmin, args.pupil_rmax)
            mask_sum = int(pupil_mask.sum())
            if mask_sum >= args.min_k or args.pupil_force_gate:
                cand_xy = cand_xy[pupil_mask]
                cand_score2 = cand_score2[pupil_mask]

    best = None
    best_key = None
    chosen_template_idx = None
    T_hat = None
    matches = None

    if len(cand_xy) >= args.min_k:
        if args.template_mode == "single":
            best = run_matcher_for_template(
                state["template"], cand_xy, cand_score2, args, state["s_expected"], params["eps_eff"], state["ratio_index_single"],
                cand_raw_count=int(cand_raw_merged)
            )
        else:
            bank = state["bank_templates"] if state["bank_templates"] is not None else load_template_bank(args)
            for bi, template_xy in enumerate(bank):
                ratio_idx = None
                if state["ratio_index_bank"] is not None and bi < len(state["ratio_index_bank"]):
                    ratio_idx = state["ratio_index_bank"][bi]
                res = run_matcher_for_template(
                    template_xy, cand_xy, cand_score2, args, state["s_expected"], params["eps_eff"], ratio_idx,
                    cand_raw_count=int(cand_raw_merged)
                )
                if res is None:
                    continue
                key = score_match_result(
                    res, cand_score2, args.appearance_tiebreak, args.bank_select_metric, s_expected=state["s_expected"]
                )
                if best is None or key > best_key:
                    best = res
                    best_key = key
                    chosen_template_idx = bi
    if best is not None:
        s_fit, _, _, matches, T_hat, app_sum = best

    n_glints = int(state["n_glints"])
    glint_xy = np.full((n_glints, 2), np.nan, dtype=float)
    if matches is not None:
        for ti, ci, _ in matches:
            if 0 <= int(ti) < n_glints and 0 <= int(ci) < len(cand_xy):
                glint_xy[int(ti)] = cand_xy[int(ci)]
    if T_hat is not None:
        template_xy = np.array(T_hat, dtype=float)
    else:
        template_xy = np.full((n_glints, 2), np.nan, dtype=float)

    return {"name": fp.name, "glint_xy": glint_xy, "template_xy": template_xy}


def _run_eval_multiproc(
    files: List[Path],
    args: argparse.Namespace,
    template: np.ndarray,
    bank_templates: Optional[List[np.ndarray]],
    ratio_index_bank: Optional[List[Dict[str, Any]]],
    ratio_index_single: Optional[Dict[str, Any]],
    d_expected: Optional[np.ndarray],
    s_expected: float,
    n_glints: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    glints_by_image: Dict[str, np.ndarray] = {}
    template_by_image: Dict[str, np.ndarray] = {}
    mp_ctx = mp.get_context("spawn")
    args_dict = vars(args)
    with mp_ctx.Pool(
        processes=int(args.workers),
        initializer=_mp_init,
        initargs=(args_dict, template, bank_templates, ratio_index_bank, ratio_index_single, d_expected, s_expected, n_glints),
    ) as pool:
        file_list = [str(p) for p in files]
        for res in tqdm(pool.imap_unordered(_process_one_image_mp, file_list), total=len(file_list), desc="Processing images (mp)"):
            if res is None:
                continue
            glints_by_image[res["name"]] = res["glint_xy"]
            template_by_image[res["name"]] = res["template_xy"]
    return glints_by_image, template_by_image

def iter_images(folder: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(files)


def safe_float_str(x: float) -> str:
    if x is None or (isinstance(x, float) and (not np.isfinite(x))):
        return "nan"
    return f"{x:.3f}"


def run_eval(args: argparse.Namespace) -> Dict[str, Any]:
    """Run one evaluation with current args; returns summary metrics."""
    t0 = time.time()
    in_dir = Path(args.folder).expanduser().resolve()
    if not in_dir.exists():
        raise FileNotFoundError(f"Folder not found: {in_dir}")

    sweep_mode = bool(getattr(args, "_sweep_mode", False))
    sweep_keep = bool(getattr(args, "sweep_keep_reports", False))
    if sweep_mode and getattr(args, "save_glints_npz", None):
        raise ValueError("--save_glints_npz is not supported in sweep mode")
    out_dir_override = getattr(args, "_sweep_out_dir", None)
    out_dir = Path(out_dir_override) if out_dir_override is not None else (in_dir / "annotated")
    out_dir.mkdir(parents=True, exist_ok=True)
    roi_debug_dir = None
    if getattr(args, "pupil_roi_debug", False):
        roi_debug_dir = out_dir / "roi_debug"
        roi_debug_dir.mkdir(parents=True, exist_ok=True)
    
    labels = None
    if args.labels is not None:
        label_path = Path(args.labels).expanduser().resolve()
        if not label_path.exists():
            raise FileNotFoundError(f"Label file not found: {label_path}")
        labels = load_labels_json(label_path)

    if getattr(args, "image_config", None):
        _apply_image_config(args, args.image_config)

    pupil_npz_map = None
    if getattr(args, "pupil_npz", None):
        args.pupil_roi = True
        args.pupil_source = "npz"
        args.pupil_roi_fail_policy = "full_frame"
        pupil_npz_map = load_pupil_npz(args.pupil_npz)
    else:
        args.pupil_roi = False

    ml_log_done = False
    if args.score2_mode == "ml_cc":
        if not args.ml_model_path:
            raise RuntimeError("score2_mode=ml_cc requires --ml_model_path")
        model, feat_names = _load_ml_model(args.ml_model_path)
        feat_dim = len(feat_names) if feat_names is not None else len(cc_feature_names())
        print(f"[ml_cc] model={args.ml_model_path} feature_dim={feat_dim}")
    
    if args.template_bank_source == "custom" and args.template_bank_path:
        P_list = load_template_bank(args)
        if not P_list:
            raise FileNotFoundError(f"Custom template bank empty or not found: {args.template_bank_path}")
    else:
        P_list = load_default_template_bank()
    if args.template_build_mode == "median":
        template = build_template_median(P_list)
        hist: List[Dict[str, float]] = []
        if args.verbose_template:
            print("[template] using median template from labeled sets")
    else:
        template, hist = build_template_from_labeled_sets(P_list, iters=10, tol=1e-6, verbose=args.verbose_template)
    # Expected scale prior derived from labeled constellations vs canonical template
    L_ref = float(np.median([constellation_scale(P) for P in P_list]))
    L_template = constellation_scale(template)
    s_expected = L_ref / L_template if L_template > 1e-9 else 1.0
    # Set template->glint mapping size
    global T_TO_G
    n_glints = int(template.shape[0])
    T_TO_G = list(range(n_glints))

    # If template size != 5, disable 5-glint-specific priors/eval to avoid errors
    if n_glints != 5:
        if args.labels is not None:
            print(f"[gen] Template has {n_glints} points; skipping label-based evaluation (expects 5).")
            labels = None
        args.layout_prior = False
        args.layout_debug = False
        args.sla_layout_prior = False
        args.sla_layout_debug = False
        if getattr(args, "sla_semantic_prior", False) and getattr(args, "sla_semantic_mode", "full") == "full":
            print(f"[gen] Template has {n_glints} points; sla_semantic_mode=full assumes K=5. Switching to top_only.")
            args.sla_semantic_mode = "top_only"
        args.post_id_resolve = False
    bank_templates = None
    if args.template_mode == "bank":
        bank_templates = load_template_bank(args)
    ratio_index_bank = None
    ratio_index_single = None
    if args.matcher == "sla":
        if bank_templates is not None:
            ratio_index_bank = [build_ratio_index(t) for t in bank_templates]
        else:
            ratio_index_single = build_ratio_index(template)
    d_expected = None
    if args.score2_mode == "contrast_support":
        bank_for_support = bank_templates if bank_templates is not None else [template]
        d_expected = compute_expected_pairwise_distances(bank_for_support)
    
    files = iter_images(in_dir)
    if not files:
        print("No images found.")
        return {
            "overall_acc": float("nan"),
            "overall_precision": float("nan"),
            "idf_acc": float("nan"),
            "loc_mean": float("nan"),
            "loc_median": float("nan"),
            "images": 0,
            "gt_present": 0,
            "predicted": 0,
            "correct": 0,
            "runtime_s": float(time.time() - t0),
        }

    # Accumulators for report
    per_image_rows: List[Dict[str, Any]] = []
    glints_by_image: Dict[str, np.ndarray] = {}
    template_by_image: Dict[str, np.ndarray] = {}
    per_glint_counts = [{"present": 0, "pred": 0, "correct": 0, "errs": []} for _ in range(5)]
    # Identity-free accumulators
    idf_present_total = 0
    idf_correct_total = 0
    idf_assigned_errs: List[float] = []
    diag_rows: List[Dict[str, Any]] = []
    
    pupil_radii_raw = [int(x) for x in args.pupil_radii.split(",") if x.strip().isdigit()]
    if not pupil_radii_raw:
        pupil_radii_raw = [12, 16, 20, 24, 28, 32]
    last_good_pupil = None

    workers = int(getattr(args, "workers", 0) or 0)
    use_mp = workers > 1
    if use_mp:
        if labels is not None or args.visualize or args.diag_candidate_recall or args.viz_metrics or args.pupil_roi_debug:
            print("[mp] Multiprocessing disabled for labels/visualize/diagnostics.")
            use_mp = False
        elif args.pupil_roi_fail_policy == "last_good":
            print("[mp] pupil_roi_fail_policy=last_good not supported in multiprocessing; using full_frame.")
            args.pupil_roi_fail_policy = "full_frame"

    if use_mp:
        glints_by_image, template_by_image = _run_eval_multiproc(
            files,
            args,
            template,
            bank_templates,
            ratio_index_bank,
            ratio_index_single,
            d_expected,
            s_expected,
            n_glints,
        )
    else:
        for fp in tqdm(files, desc="Processing images"):
            bgr = cv2.imread(str(fp), cv2.IMREAD_COLOR)
            if bgr is None:
                print(f"Skip (unreadable): {fp.name}")
                continue

            gray_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            H, W = gray_full.shape[:2]
            image_center = (0.5 * W, 0.5 * H)

            params = scale_params_for_image(args, w=gray_full.shape[1], h=gray_full.shape[0])
            pcx = pcy = pr = None
            pupil_detected = False
            pupil_ok = False
            pupil_source_used = "none"
            if args.pupil_roi and args.pupil_source != "none":
                pcx, pcy, pr, pupil_detected, pupil_ok, pupil_source_used = detect_pupil_center_for_frame(
                    gray_full, labels, fp.name, W, H, args, params, pupil_radii_raw, pupil_npz_map
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
                    last_good_pupil,
                )
                if roi_decision.action == "skip":
                    continue
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
                        last_good_pupil = (roi_decision.center[0], roi_decision.center[1], roi_decision.radius)
                    else:
                        last_good_pupil = (roi_decision.center[0], roi_decision.center[1], float("nan"))

                    if roi_debug_dir is not None:
                        dbg = bgr.copy()
                        rx0 = roi_info.offset_x
                        ry0 = roi_info.offset_y
                        rx1 = rx0 + int(args.pupil_roi_size)
                        ry1 = ry0 + int(args.pupil_roi_size)
                        x0 = max(0, rx0)
                        y0 = max(0, ry0)
                        x1 = min(W - 1, rx1)
                        y1 = min(H - 1, ry1)
                        cv2.rectangle(dbg, (int(x0), int(y0)), (int(x1), int(y1)), (0, 200, 255), 2)
                        cv2.imwrite(str(roi_debug_dir / f"{fp.stem}_roi_full.png"), dbg)
                        cv2.imwrite(str(roi_debug_dir / f"{fp.stem}_roi_crop.png"), roi_info.roi_img)
                        if roi_info.roi_mask is not None:
                            mask = roi_info.roi_mask
                            if mask.ndim == 2:
                                mask_vis = cv2.cvtColor(mask.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
                            else:
                                mask_vis = mask.copy()
                            cv2.imwrite(str(roi_debug_dir / f"{fp.stem}_roi_mask.png"), mask_vis)
    
            cand_xy_pass0, rows_pass0, cand_score2_pass0, cand_raw_pass0, cand_support_pass0 = detect_candidates_one_pass(
                gray, params, args, d_expected=d_expected
            )
            cand_xy_raw = cand_xy_pass0
            cand_score2_raw = cand_score2_pass0
            cand_support_raw = cand_support_pass0
            cand_raw_merged = int(cand_raw_pass0)
            cand_passes_used = 1
    
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
                    cand_xy_i, rows_i, cand_score2_i, cand_raw_i, cand_support_i = detect_candidates_one_pass(
                        gray, params, args, d_expected=d_expected, percentile_override=perc, kernel_add=kernel_add
                    )
                    cand_xy_list.append(cand_xy_i)
                    cand_score2_list.append(cand_score2_i)
                    cand_support_list.append(cand_support_i)
                    cand_xy_raw, cand_score2_raw, cand_support_raw = merge_candidates(
                        cand_xy_list, cand_score2_list, cand_support_list, merge_eps=float(args.cand_merge_eps)
                    )
                    cand_raw_merged = int(len(cand_xy_raw))
                    cand_passes_used = i + 1
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

            cand_raw_count = int(cand_raw_merged)
            support_mean = float(np.mean(cand_support_raw)) if cand_support_raw.size > 0 else 0.0
    
            gt_xy_diag = None
            gt_present = 0
            gt_recalled = 0
            if args.diag_candidate_recall and labels is not None:
                gt_xy_diag = gt_glints_for_image(labels, fp.name)
                if gt_xy_diag is not None:
                    gt = np.asarray(gt_xy_diag, float)
                    present_mask = np.isfinite(gt[:, 0]) & np.isfinite(gt[:, 1]) & (gt[:, 0] >= 0) & (gt[:, 1] >= 0)
                    gt_points = gt[present_mask]
                    gt_present = int(len(gt_points))
                    if gt_present > 0 and cand_xy_raw.size > 0:
                        diag_eps = float(args.diag_recall_eps) if args.diag_recall_eps is not None else float(args.eps)
                        d = np.sqrt(((gt_points[:, None, :] - cand_xy_raw[None, :, :]) ** 2).sum(axis=2))
                        gt_recalled = int(np.sum(np.min(d, axis=1) <= diag_eps))
    
            # Candidates for RANSAC from top score2
            rows_sorted = sorted(
                zip(cand_xy_raw, cand_score2_raw),
                key=lambda p: p[1],
                reverse=True,
            )[: args.max_pool]
            cand_xy = np.array([p[0] for p in rows_sorted], dtype=float) if rows_sorted else np.empty((0, 2), dtype=float)
            cand_score2 = np.array([p[1] for p in rows_sorted], dtype=float) if rows_sorted else np.empty((0,), dtype=float)
            if args.score2_mode == "ml_cc" and not ml_log_done:
                print(f"[ml_cc] candidates raw={cand_raw_count} pooled={len(cand_xy)}")
                ml_log_done = True
            cand_raw = int(cand_raw_merged)
            cand_pool_count = int(len(cand_xy))
            roi_shape = gray_full.shape if roi_active else gray.shape
            cand_xy, cand_score2, roi_margin = filter_candidates_roi(
                cand_xy, cand_score2, roi_shape, mode=args.roi_mode,
                border_frac=args.roi_border_frac, border_px=args.roi_border_px)
            cand_roi = len(cand_xy)
            roi_center = image_center
            roi_radius_estimate = max(1.0, 0.5 * (min(W, H) - 2 * roi_margin))
            pupil_center_vis: Optional[Tuple[float, float]] = None
            pupil_radius_vis: Optional[float] = None
            pupil_gate_applied = False
            cand_before_pupil = len(cand_xy)
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
                    pupil_center_vis = (pcx_gate, pcy_gate)
                    pupil_radius_vis = pr_gate
                    pupil_mask = gate_candidates_by_pupil(
                        cand_xy, (pcx_gate, pcy_gate), pr_gate, args.pupil_rmin, args.pupil_rmax)
                    mask_sum = int(pupil_mask.sum())
                    if mask_sum < args.min_k and not args.pupil_force_gate:
                        pupil_gate_applied = False
                    else:
                        cand_xy = cand_xy[pupil_mask]
                        cand_score2 = cand_score2[pupil_mask]
                        pupil_gate_applied = True
            cand_after_pupil = len(cand_xy)
            if args.debug_pupil:
                if pupil_center_vis is not None and pupil_radius_vis is not None:
                    cx_dbg, cy_dbg, pr_dbg = pupil_center_vis[0], pupil_center_vis[1], pupil_radius_vis
                    center_str = f"\"({cx_dbg:.2f},{cy_dbg:.2f},{pr_dbg:.2f})\""
                else:
                    center_str = "\"(na,na,na)\""
                print(
                    f"{fp.name},{pupil_source_used},{int(pupil_detected)},{int(pupil_ok)},"
                    f"{center_str},{cand_raw},{cand_roi},{cand_after_pupil}"
                )
    
            best = None
            best_key = None
            chosen_template_idx = None
            T_hat = None
            matches = None
            inliers = 0
            mean_err = float("nan")
            app_sum = float("nan")
            s_fit = float("nan")
    
            if len(cand_xy) >= args.min_k:
                if args.template_mode == "single":
                    best = run_matcher_for_template(
                        template, cand_xy, cand_score2, args, s_expected, params["eps_eff"], ratio_index_single,
                        cand_raw_count=cand_raw_count
                    )
                else:
                    bank = bank_templates if bank_templates is not None else load_template_bank(args)
                    for bi, template_xy in enumerate(bank):
                        ratio_idx = None
                        if ratio_index_bank is not None and bi < len(ratio_index_bank):
                            ratio_idx = ratio_index_bank[bi]
                        res = run_matcher_for_template(
                            template_xy, cand_xy, cand_score2, args, s_expected, params["eps_eff"], ratio_idx,
                            cand_raw_count=cand_raw_count
                        )
                        if res is None:
                            continue
                        key = score_match_result(
                            res, cand_score2, args.appearance_tiebreak, args.bank_select_metric, s_expected=s_expected
                        )
                        if best is None or key > best_key:
                            best = res
                            best_key = key
                            chosen_template_idx = bi
            if best is not None:
                s_fit, _, _, matches, T_hat, app_sum = best
                inliers = len(matches) if matches else 0
                mean_err = float(np.mean([m[2] for m in matches])) if matches else float("nan")
            else:
                app_sum = float("nan")

            if getattr(args, "save_glints_npz", None):
                glint_xy = np.full((n_glints, 2), np.nan, dtype=float)
                if matches is not None:
                    for ti, ci, _ in matches:
                        if 0 <= int(ti) < n_glints and 0 <= int(ci) < len(cand_xy):
                            glint_xy[int(ti)] = cand_xy[int(ci)]
                glints_by_image[fp.name] = glint_xy
                if T_hat is not None:
                    template_by_image[fp.name] = np.array(T_hat, dtype=float)
                else:
                    template_by_image[fp.name] = np.full((n_glints, 2), np.nan, dtype=float)
    
            if args.diag_candidate_recall and labels is not None:
                diag_rows.append(
                    dict(
                        image=fp.name,
                        subject=subject_from_filename(fp.name),
                        gt_present=int(gt_present),
                        gt_recalled=int(gt_recalled),
                        cand_raw_count=int(cand_raw_count),
                        cand_pool_count=int(cand_pool_count),
                        inliers=int(inliers),
                        matched_ge3=int(inliers >= 3),
                    )
                )
    
       
    
        
    
            # ---- optional metrics ----
            gt_xy = gt_xy_diag if gt_xy_diag is not None else None
            eval_out = None
            idf_out = None
            if labels is not None:
                if gt_xy is None:
                    gt_xy = gt_glints_for_image(labels, fp.name)
                if gt_xy is not None:
                    matches_gt = remap_matches_template_to_gt(matches, T_TO_G)
                    eval_out = evaluate_matches(gt_xy, cand_xy, matches_gt, match_tol=args.match_tol)
    
    
                    # identity-free evaluation uses the predicted glint set (matched candidates), ignoring template indices
                    if matches is not None:
                        pred_xy_idf = np.array([cand_xy[ci] for (_, ci, _) in matches], dtype=float)
                    else:
                        pred_xy_idf = np.empty((0, 2), dtype=float)
                    idf_out = evaluate_identity_free(gt_xy, pred_xy_idf, match_tol=args.match_tol)
    
                    # accumulate identity-free totals
                    idf_present_total += int(idf_out["present_count"])
                    idf_correct_total += int(idf_out["correct_count"])
                    if np.isfinite(idf_out["loc_err_mean"]):
                        # store per-pair errors implicitly by expanding from mean? no.
                        # Instead, recompute assigned pair errors here for global mean/median.
                        gt = np.asarray(gt_xy, float)
                        pmask = np.isfinite(gt[:, 0])
                        gtP = gt[pmask]
                        if pred_xy_idf.shape[0] > 0 and gtP.shape[0] > 0:
                            cost = np.sqrt(((gtP[:, None, :] - pred_xy_idf[None, :, :]) ** 2).sum(axis=2))
                            pairs, _ = _best_bipartite_match_small(cost)
                            idf_assigned_errs.extend([float(cost[i, j]) for (i, j) in pairs])
    
                    # accumulate per-glint stats
                    for g in eval_out["per_glint"]:
                        gi = int(g["glint"])
                        if g["present"]:
                            per_glint_counts[gi]["present"] += 1
                        if g["predicted"]:
                            per_glint_counts[gi]["pred"] += 1
                        if g["correct"]:
                            per_glint_counts[gi]["correct"] += 1
                        if g["err_px"] is not None:
                            per_glint_counts[gi]["errs"].append(float(g["err_px"]))
    
            # ---- build title + filename suffix ----
            # include fitted vs expected scale, appearance tie-break sum, and scaled params for debugging
            tmpl_txt = ""
            if args.template_mode == "bank":
                tmpl_id = chosen_template_idx if chosen_template_idx is not None else "na"
                tmpl_txt = f" | tmpl={tmpl_id}"
            fb_txt = ""
            if args.cand_fallback_debug and args.cand_fallback:
                fb_txt = f" | fallback=on passes={cand_passes_used} raw0={cand_raw_pass0} rawM={cand_raw_merged}"
            score2_txt = f" | score2={args.score2_mode} supp_mean={support_mean:.2f}"
            lp_txt = ""
            if args.layout_debug and args.layout_prior:
                pts_g = points_g_order_from_matches(matches, cand_xy, T_TO_G)
                if pts_g is not None:
                    lp_val = layout_penalty_g0_top2(pts_g, mode=args.layout_mode)
                    lp_txt = f" | lp={lp_val:.2f}"
            sla_lp_txt = ""
            if args.sla_layout_debug and args.sla_layout_prior and args.matcher == "sla":
                glint_xy_hat = np.full((5, 2), np.nan, dtype=float)
                if matches is not None:
                    for ti2, ci2, _ in matches:
                        if 0 <= ti2 < 5 and 0 <= ci2 < len(cand_xy):
                            glint_xy_hat[ti2] = cand_xy[ci2]
                lp_val = sla_layout_penalty_g0_top2(glint_xy_hat)
                g0_top2 = None
                valid = np.isfinite(glint_xy_hat[:, 0]) & np.isfinite(glint_xy_hat[:, 1])
                if valid.sum() >= 3 and np.all(np.isfinite(glint_xy_hat[0])):
                    ys = glint_xy_hat[valid, 1]
                    idx_valid = np.flatnonzero(valid)
                    order = np.argsort(ys)
                    top2 = set(idx_valid[order[:2]].tolist())
                    g0_top2 = (0 in top2)
                if g0_top2 is None:
                    sla_lp_txt = f" | sla_lp={lp_val:.2f}"
                else:
                    sla_lp_txt = f" | sla_lp={lp_val:.2f} g0top2={'yes' if g0_top2 else 'no'}"
            title = (f"{fp.name} | matcher={args.matcher}{tmpl_txt} | cand_raw={cand_raw} cand_roi={cand_roi} | roi={args.roi_mode}"
                     f"(m={roi_margin}) | s={params['s']:.2f} eps={params['eps_eff']:.2f} "
                     f"tol={params['match_tol_eff']:.2f} k={params['kernel_eff']} "
                     f"area=[{params['min_area_eff']},{params['max_area_eff']}] | "
                     f"inliers={inliers} | err={mean_err:.2f}px | "
                     f"scale={s_fit:.2f} (exp {s_expected:.2f}) | app_sum={app_sum:.1f}{fb_txt}{score2_txt}{lp_txt}{sla_lp_txt}")
    
            suffix = args.matcher
            if args.pupil_roi and args.pupil_source != "none":
                suffix += "_pupilROI"
            suffix += f"_inl{inliers}_err{mean_err:.2f}"
            if args.template_mode == "bank":
                tmpl_id = chosen_template_idx if chosen_template_idx is not None else "na"
                suffix += f"_bankT{tmpl_id}"
            if eval_out is not None:
                acc = eval_out["match_accuracy"]
                loc = eval_out["loc_err_mean"]
                title += f" | acc={acc:.2f} | loc={loc:.2f}px"
                suffix += f"_acc{acc:.2f}_loc{loc:.2f}"
    
                # Append identity-free metrics (if available)
                if idf_out is not None:
                    idfa = idf_out["match_accuracy"]
                    idfl = idf_out["loc_err_mean"]
                    title += f" | idf_acc={idfa:.2f} | idf_loc={idfl:.2f}px | tol={args.match_tol:.0f}"
                    suffix += f"_idfacc{idfa:.2f}_idfloc{idfl:.2f}"
                else:
                    title += f" | tol={args.match_tol:.0f}"
    
            if args.visualize:
                overlay = draw_overlay(gray_full, cand_xy, T_hat, matches, title_text=title, gt_xy=gt_xy, match_tol=args.match_tol)
                if pupil_center_vis is not None and pupil_radius_vis is not None:
                    color = (0, 165, 255) if pupil_detected else (128, 128, 0)
                    pcx, pcy = pupil_center_vis
                    cv2.circle(overlay, (int(round(pcx)), int(round(pcy))), int(round(pupil_radius_vis)), color, 1, lineType=cv2.LINE_AA)
                    cv2.circle(overlay, (int(round(pcx)), int(round(pcy))), 2, color, -1, lineType=cv2.LINE_AA)
                if args.pupil_roi:
                    color = (0, 165, 255) if pupil_detected else (128, 128, 0)
                    r_txt = f"{pupil_radius_vis:.1f}" if pupil_radius_vis is not None else "na"
                    txt = (f"pupil={pupil_source_used} "
                           f"ok={'yes' if pupil_ok else 'no'} "
                           f"gate={'on' if pupil_gate_applied else 'off'} "
                           f"r={r_txt} "
                           f"cand={cand_before_pupil}->{len(cand_xy)}")
                    cv2.putText(overlay, txt, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, lineType=cv2.LINE_AA)
                out_path = out_dir / f"{fp.stem}_{suffix}.png"
                cv2.imwrite(str(out_path), overlay)
    
            # record per-image line for report (if eval)
            if eval_out is not None:
                per_image_rows.append(
                    dict(
                        filename=fp.name,
                        candidates=int(len(cand_xy)),
                        cand_raw=int(cand_raw),
                        cand_roi=int(cand_roi),
                        roi_margin_px=int(roi_margin),
                        kernel_eff=int(params["kernel_eff"]),
                        median_ksize_eff=int(params["median_ksize_eff"]),
                        cand_passes_used=int(cand_passes_used),
                        cand_raw_pass0=int(cand_raw_pass0),
                        cand_raw_merged=int(cand_raw_merged),
                        eps_eff=float(params["eps_eff"]),
                        match_tol_eff=float(params["match_tol_eff"]),
                        min_area_eff=int(params["min_area_eff"]),
                        max_area_eff=int(params["max_area_eff"]),
                        inliers=int(inliers),
                        ransac_err_px=(None if not np.isfinite(mean_err) else float(mean_err)),
                        app_sum=(None if not np.isfinite(app_sum) else float(app_sum)),
                        scale_factor=float(params["s"]),
                        present=int(eval_out["present_count"]),
                        predicted=int(eval_out["pred_count"]),
                        correct=int(eval_out["correct_count"]),
                        match_accuracy=float(eval_out["match_accuracy"]),
                        precision=float(eval_out["precision"]),
                        loc_err_mean=float(eval_out["loc_err_mean"]) if np.isfinite(eval_out["loc_err_mean"]) else None,
                        loc_err_median=float(eval_out["loc_err_median"]) if np.isfinite(eval_out["loc_err_median"]) else None,
    
                        idf_present=(None if idf_out is None else int(idf_out["present_count"])),
                        idf_predicted=(None if idf_out is None else int(idf_out["pred_count"])),
                        idf_correct=(None if idf_out is None else int(idf_out["correct_count"])),
                        idf_match_accuracy=(None if idf_out is None else float(idf_out["match_accuracy"])),
                        idf_loc_err_mean=(None if (idf_out is None or not np.isfinite(idf_out["loc_err_mean"])) else float(idf_out["loc_err_mean"])),
                        idf_loc_err_median=(None if (idf_out is None or not np.isfinite(idf_out["loc_err_median"])) else float(idf_out["loc_err_median"])),
                    )
                )
    
    # ---- compute metrics ----
    overall_acc = float("nan")
    overall_prec = float("nan")
    overall_loc_mean = float("nan")
    overall_loc_median = float("nan")
    overall_idf_acc = float("nan")
    overall_idf_loc_mean = float("nan")
    overall_idf_loc_median = float("nan")
    present_total = 0
    correct_total = 0
    pred_total = 0
    all_errs = []

    if labels is not None and per_image_rows:
        # Overall (micro-averaged)
        present_total = sum(r["present"] for r in per_image_rows)
        correct_total = sum(r["correct"] for r in per_image_rows)
        pred_total = sum(r["predicted"] for r in per_image_rows)

        overall_acc = correct_total / present_total if present_total > 0 else float("nan")
        overall_prec = correct_total / pred_total if pred_total > 0 else float("nan")

        # Localization across all matched GT glints
        for gi in range(5):
            all_errs.extend(per_glint_counts[gi]["errs"])
        overall_loc_mean = float(np.mean(all_errs)) if all_errs else float("nan")
        overall_loc_median = float(np.median(all_errs)) if all_errs else float("nan")

        # Identity-free (micro-averaged)
        overall_idf_acc = idf_correct_total / idf_present_total if idf_present_total > 0 else float("nan")
        overall_idf_loc_mean = float(np.mean(idf_assigned_errs)) if idf_assigned_errs else float("nan")
        overall_idf_loc_median = float(np.median(idf_assigned_errs)) if idf_assigned_errs else float("nan")

        if (not sweep_mode) or sweep_keep:
            # Optional metric visualizations
            if args.viz_metrics:
                generate_metric_visuals(out_dir, per_glint_counts, all_errs, idf_assigned_errs, per_image_rows)

            # Save CSV per image
            import csv

            csv_path = out_dir / "report_per_image.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(per_image_rows[0].keys()))
                w.writeheader()
                w.writerows(per_image_rows)

            # Save per-subject aggregates
            write_subject_report(per_image_rows, out_dir)

            # Save text report
            report_path = out_dir / "report.txt"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write("Glint constellation matching report\n")
                f.write("=" * 40 + "\n\n")
                f.write(f"Images evaluated: {len(per_image_rows)}\n")
                f.write(f"GT glints (present): {present_total}\n")
                f.write(f"Predicted glints: {pred_total}\n")
                f.write(f"Correct matches (<= {args.match_tol:.0f}px): {correct_total}\n")
                f.write(f"Overall matching accuracy (correct/present): {overall_acc:.4f}\n")
                f.write(f"Overall precision (correct/predicted): {overall_prec:.4f}\n")
                f.write(f"Overall localization error mean (px): {safe_float_str(overall_loc_mean)}\n")
                f.write(f"Overall localization error median (px): {safe_float_str(overall_loc_median)}\n\n")

                f.write("Identity-free metrics (glint identity ignored)\n")
                f.write("-" * 40 + "\n")
                f.write(f"Identity-free matching accuracy (correct/present): {safe_float_str(overall_idf_acc)}\n")
                f.write(f"Identity-free localization error mean (px): {safe_float_str(overall_idf_loc_mean)}\n")
                f.write(f"Identity-free localization error median (px): {safe_float_str(overall_idf_loc_median)}\n\n")

                f.write("Per-glint breakdown (label index G0..G4)\n")

                f.write("-" * 40 + "\n")
                for gi in range(5):
                    pc = per_glint_counts[gi]
                    p = pc["present"]
                    c = pc["correct"]
                    pr = pc["pred"]
                    acc_g = (c / p) if p > 0 else float("nan")
                    prec_g = (c / pr) if pr > 0 else float("nan")
                    mean_g = float(np.mean(pc["errs"])) if pc["errs"] else float("nan")
                    med_g = float(np.median(pc["errs"])) if pc["errs"] else float("nan")
                    f.write(
                        f"Glint {gi}: present={p} predicted={pr} correct={c} | "
                        f"acc={safe_float_str(acc_g)} prec={safe_float_str(prec_g)} | "
                        f"err_mean={safe_float_str(mean_g)} err_median={safe_float_str(med_g)}\n"
                    )

                f.write("\nPer-image summary (see report_per_image.csv for full table)\n")
                f.write("-" * 40 + "\n")
                for r in per_image_rows:
                    f.write(
                        f"{r['filename']}: acc={safe_float_str(r['match_accuracy'])} "
                        f"loc_mean={safe_float_str(r['loc_err_mean'] if r['loc_err_mean'] is not None else float('nan'))} "
                        f"idf_acc={safe_float_str(r['idf_match_accuracy'] if r.get('idf_match_accuracy') is not None else float('nan'))} "
                        f"inliers={r['inliers']}\n"
                    )

            print(f"Done. Saved overlays to: {out_dir}")
            print(f"Saved report: {report_path}")
            print(f"Saved CSV: {csv_path}")
    else:
        if not sweep_mode:
            print(f"Done. Saved overlays to: {out_dir}")
            if labels is not None:
                print("Note: labels were provided, but no images matched keys in the label file.")

    if args.diag_candidate_recall and diag_rows and ((not sweep_mode) or sweep_keep):
        out_csv = Path(args.diag_out_csv)
        if not out_csv.is_absolute():
            out_csv = out_dir / out_csv
        import csv
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "image",
                    "subject",
                    "gt_present",
                    "gt_recalled",
                    "cand_raw_count",
                    "cand_pool_count",
                    "inliers",
                    "matched_ge3",
                ],
            )
            w.writeheader()
            w.writerows(diag_rows)

        if args.diag_make_plots:
            try:
                import matplotlib.pyplot as plt
                rows = diag_rows
                gt_present_arr = np.array([r["gt_present"] for r in rows], dtype=int)
                gt_recalled_arr = np.array([r["gt_recalled"] for r in rows], dtype=int)
                cand_raw_arr = np.array([r["cand_raw_count"] for r in rows], dtype=int)
                matched_ge3_arr = np.array([r["matched_ge3"] for r in rows], dtype=int)

                fig, axes = plt.subplots(1, 3, figsize=(12, 4))

                ks = np.arange(1, 6)
                recall_at_k = []
                for k in ks:
                    mask = gt_present_arr >= k
                    if np.any(mask):
                        recall_at_k.append(float(np.mean(gt_recalled_arr[mask] >= k)))
                    else:
                        recall_at_k.append(float("nan"))
                axes[0].plot(ks, recall_at_k, marker="o")
                axes[0].set_title("Recall@K")
                axes[0].set_xlabel("K")
                axes[0].set_ylabel("Recall")
                axes[0].set_ylim(0, 1.0)

                max_c = int(cand_raw_arr.max()) if cand_raw_arr.size > 0 else 0
                bins = np.arange(0, max_c + 11, 10)
                if len(bins) < 2:
                    bins = np.array([0, 10], dtype=int)
                centers = 0.5 * (bins[:-1] + bins[1:])
                recall_bins = []
                for lo, hi in zip(bins[:-1], bins[1:]):
                    mask_bin = (cand_raw_arr >= lo) & (cand_raw_arr < hi) & (gt_present_arr >= 3)
                    if np.any(mask_bin):
                        recall_bins.append(float(np.mean(gt_recalled_arr[mask_bin] >= 3)))
                    else:
                        recall_bins.append(float("nan"))
                axes[1].plot(centers, recall_bins, marker="o")
                axes[1].set_title("Recall>=3 vs cand_raw")
                axes[1].set_xlabel("cand_raw bin center")
                axes[1].set_ylabel("Recall")
                axes[1].set_ylim(0, 1.0)

                recall_bins_success = []
                recall_bins_failure = []
                for lo, hi in zip(bins[:-1], bins[1:]):
                    mask_bin = (cand_raw_arr >= lo) & (cand_raw_arr < hi) & (gt_present_arr >= 3)
                    mask_s = mask_bin & (matched_ge3_arr == 1)
                    mask_f = mask_bin & (matched_ge3_arr == 0)
                    if np.any(mask_s):
                        recall_bins_success.append(float(np.mean(gt_recalled_arr[mask_s] >= 3)))
                    else:
                        recall_bins_success.append(float("nan"))
                    if np.any(mask_f):
                        recall_bins_failure.append(float(np.mean(gt_recalled_arr[mask_f] >= 3)))
                    else:
                        recall_bins_failure.append(float("nan"))
                axes[2].plot(centers, recall_bins_success, marker="o", label="matched>=3")
                axes[2].plot(centers, recall_bins_failure, marker="x", label="matched<3")
                axes[2].set_title("Recall>=3 by matcher success")
                axes[2].set_xlabel("cand_raw bin center")
                axes[2].set_ylabel("Recall")
                axes[2].set_ylim(0, 1.0)
                axes[2].legend()

                plt.tight_layout()
                plot_path = Path(args.diag_plot_path)
                if not plot_path.is_absolute():
                    plot_path = out_dir / plot_path
                plt.savefig(plot_path)
                plt.close(fig)
            except Exception:
                pass

    if getattr(args, "save_glints_npz", None):
        out_path = Path(str(args.save_glints_npz)).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        config = {"args": vars(args), "folder": str(in_dir)}
        np.savez(out_path, glints=glints_by_image, template_xy=template_by_image, config=config)
        print(f"Wrote glints NPZ: {out_path}")

    return {
        "overall_acc": float(overall_acc),
        "overall_precision": float(overall_prec),
        "idf_acc": float(overall_idf_acc),
        "loc_mean": float(overall_loc_mean),
        "loc_median": float(overall_loc_median),
        "images": int(len(files)),
        "gt_present": int(present_total),
        "predicted": int(pred_total),
        "correct": int(correct_total),
        "runtime_s": float(time.time() - t0),
    }


def _load_sweep_grid(sweep_grid: str) -> Dict[str, Any]:
    """
    Load sweep grid from JSON string or .json file.
    Schema: { "base":{...}, "vary":{k:[...]}, "zip":[[k1,k2,...], [[v1,v2,...], ...]] }
    """
    p = Path(sweep_grid)
    if p.exists() and p.is_file():
        data = json.loads(p.read_text(encoding="utf-8-sig"))
    else:
        data = json.loads(sweep_grid)
    if not isinstance(data, dict):
        raise ValueError("sweep_grid must be a JSON object")
    return data


def _expand_sweep_runs(grid: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
    base = grid.get("base", {}) or {}
    vary = grid.get("vary", {}) or {}
    zip_spec = grid.get("zip", None)
    zip_only = bool(grid.get("zip_only", False))
    if not isinstance(base, dict) or not isinstance(vary, dict):
        raise ValueError("sweep grid must contain dicts for 'base' and 'vary'")

    runs: List[Dict[str, Any]] = []
    varied_keys = set(vary.keys())

    # Cartesian product of vary
    if vary:
        keys = list(vary.keys())
        values_lists = [vary[k] for k in keys]
        for vals in itertools.product(*values_lists):
            cfg = dict(base)
            for k, v in zip(keys, vals):
                cfg[k] = v
            runs.append(cfg)
    else:
        if not zip_only:
            runs.append(dict(base))

    # Zipped runs appended to cartesian
    if zip_spec is not None:
        if (not isinstance(zip_spec, list)) or len(zip_spec) != 2:
            raise ValueError("zip must be [ [argA,argB,...], [ [a1,b1,...], ... ] ]")
        zip_keys, zip_rows = zip_spec
        if not isinstance(zip_keys, list) or not isinstance(zip_rows, list):
            raise ValueError("zip keys/rows must be lists")
        varied_keys.update(zip_keys)
        for row in zip_rows:
            if not isinstance(row, list) or len(row) != len(zip_keys):
                raise ValueError("zip row length must match zip keys length")
            cfg = dict(base)
            for k, v in zip(zip_keys, row):
                cfg[k] = v
            runs.append(cfg)

    return runs, sorted(varied_keys)


def _parse_comma_list(s: Optional[str]) -> Optional[List[str]]:
    if s is None:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def _get_overridden_keys(cfg: Dict[str, Any]) -> List[str]:
    return sorted([k for k, v in cfg.items() if v is not None])


def canonicalize_run_args_for_matcher(
    run_args: argparse.Namespace,
    overridden_keys: List[str],
    strict: bool = True,
) -> None:
    """
    Enforce matcher-specific constraints and map shared layout knobs for SLA.
    """
    all_keys = set(vars(run_args).keys())
    invalid = [k for k in overridden_keys if k not in all_keys]
    if invalid:
        raise ValueError(f"Invalid sweep keys: {invalid}")

    sla_keys = [k for k in overridden_keys if k.startswith("sla_")]
    if run_args.matcher != "sla":
        if sla_keys and strict:
            allow = {"sla_scale_min", "sla_scale_max"}
            bad = [k for k in sla_keys if k not in allow]
            if bad:
                raise ValueError(f"SLA-only keys not allowed for matcher={run_args.matcher}: {bad}")
        # ignore sla_* for non-SLA
        return

    # matcher == "sla": map generic layout knobs to SLA equivalents if not explicitly set
    if "layout_prior" in all_keys and "sla_layout_prior" in all_keys:
        if "sla_layout_prior" not in overridden_keys and "layout_prior" in overridden_keys:
            run_args.sla_layout_prior = run_args.layout_prior
    if "layout_lambda" in all_keys and "sla_layout_lambda" in all_keys:
        if "sla_layout_lambda" not in overridden_keys and "layout_lambda" in overridden_keys:
            run_args.sla_layout_lambda = run_args.layout_lambda


def make_tierb_grid(tier: str, args: argparse.Namespace) -> Dict[str, Any]:
    """
    Generate Tier B sweep grids:
    - B1: shared knobs only (fair head-to-head)
    - B2: matcher-specific tuning (separated by matcher)
    - B3: SLA ablation (one factor at a time)
    """
    tier = tier.lower()
    matchers_default = ["hybrid", "sla"]
    matchers = _parse_comma_list(getattr(args, "tierb_matchers", None)) or matchers_default

    base_common = dict(
        template_mode="bank",
        matching="greedy",
        score2_mode="contrast_support",
        cand_fallback=True,
        sla_scale_min=0.01,
        sla_scale_max=500,
    )

    if tier == "b1":
        rows = []
        shared_rows = [
            [8, 2, "99,98,97", 20, 0.08, 0.10, 0.10, 6, 100, False, 0.15],
            [8, 4, "99,98,97", 30, 0.10, 0.15, 0.12, 8, 200, True, 0.25],
            [12, 2, "99.5,99,98.5,98", 20, 0.10, 0.15, 0.10, 8, 200, False, 0.15],
            [12, 4, "99.5,99,98.5,98", 30, 0.08, 0.10, 0.12, 6, 100, True, 0.25],
        ]
        for m in matchers:
            for r in shared_rows:
                rows.append([m] + r)
        grid = {
            "base": base_common,
            "vary": {},
            "zip_only": True,
            "zip": [
                [
                    "matcher",
                    "cand_target_raw",
                    "cand_fallback_passes",
                    "cand_fallback_percentiles",
                    "support_M",
                    "support_tol",
                    "support_w",
                    "ratio_tol",
                    "pivot_P",
                    "max_seeds",
                    "layout_prior",
                    "layout_lambda",
                ],
                rows,
            ],
        }
        return grid

    if tier == "b2":
        # matcher-specific tuning: explicit rows
        rows = []
        # Hybrid (non-SLA) rows
        for cand_target_raw, cand_fallback_passes, cand_fallback_percentiles in [
            (8, 2, "99,98,97"),
            (8, 4, "99,98,97"),
            (12, 2, "99.5,99,98.5,98"),
            (12, 4, "99.5,99,98.5,98"),
        ]:
            for support_M in [20, 30]:
                for support_tol in [0.08, 0.10]:
                    for support_w in [0.10, 0.15]:
                        rows.append([
                            "hybrid",
                            cand_target_raw,
                            cand_fallback_passes,
                            cand_fallback_percentiles,
                            support_M,
                            support_tol,
                            support_w,
                            0.10,
                            6,
                            100,
                            False,
                            0.15,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                        ])
                        rows.append([
                            "hybrid",
                            cand_target_raw,
                            cand_fallback_passes,
                            cand_fallback_percentiles,
                            support_M,
                            support_tol,
                            support_w,
                            0.12,
                            8,
                            200,
                            True,
                            0.25,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                        ])
        # SLA rows
        for cand_target_raw, cand_fallback_passes, cand_fallback_percentiles in [
            (8, 2, "99,98,97"),
            (8, 4, "99,98,97"),
            (12, 2, "99.5,99,98.5,98"),
            (12, 4, "99.5,99,98.5,98"),
        ]:
            for support_M in [20, 30]:
                for support_tol in [0.08, 0.10]:
                    for support_w in [0.10, 0.15]:
                        for sem in [False, True]:
                            rows.append([
                                "sla",
                                cand_target_raw,
                                cand_fallback_passes,
                                cand_fallback_percentiles,
                                support_M,
                                support_tol,
                                support_w,
                                0.10,
                                6,
                                100,
                                False,
                                0.15,
                                "image",
                                sem,
                                1.5 if sem else 0.0,
                                True,
                                0.80,
                                0.0,
                                0.0,
                            ])
        grid = {
            "base": base_common,
            "vary": {},
            "zip_only": True,
            "zip": [
                [
                    "matcher",
                    "cand_target_raw",
                    "cand_fallback_passes",
                    "cand_fallback_percentiles",
                    "support_M",
                    "support_tol",
                    "support_w",
                    "ratio_tol",
                    "pivot_P",
                    "max_seeds",
                    "layout_prior",
                    "layout_lambda",
                    "sla_layout_mode",
                    "sla_semantic_prior",
                    "sla_semantic_lambda",
                    "sla_mirror_reject",
                    "sla_base_ratio_min",
                    "sla_top2_margin",
                    "sla_side_margin",
                ],
                rows,
            ],
        }
        return grid

    if tier == "b3":
        base = dict(
            template_mode="bank",
            score2_mode="contrast_support",
            cand_fallback=True,
            cand_target_raw=12,
            cand_fallback_passes=4,
            cand_fallback_percentiles="99.5,99,98.5,98",
            support_M=30,
            support_tol=0.10,
            support_w=0.15,
            ratio_tol=0.12,
            pivot_P=8,
            max_seeds=200,
            matching="greedy",
            matcher="sla",
            sla_layout_prior=True,
            sla_layout_lambda=0.25,
            sla_layout_mode="image",
            sla_scale_min=0.01,
            sla_scale_max=500,
            sla_semantic_prior=True,
            sla_semantic_lambda=1.5,
            sla_mirror_reject=True,
            sla_base_ratio_min=0.80,
            sla_side_margin=0.0,
            sla_top2_margin=0.0,
        )
        rows = []
        # A) semantic prior off/on
        for v in [False, True]:
            rows.append(["sla_semantic_prior", v, v, base["sla_mirror_reject"], base["sla_layout_prior"], base["sla_base_ratio_min"]])
        # B) mirror reject off/on
        for v in [False, True]:
            rows.append(["sla_mirror_reject", v, base["sla_semantic_prior"], v, base["sla_layout_prior"], base["sla_base_ratio_min"]])
        # C) layout prior off/on
        for v in [False, True]:
            rows.append(["sla_layout_prior", v, base["sla_semantic_prior"], base["sla_mirror_reject"], v, base["sla_base_ratio_min"]])
        # D) base_ratio_min sweep
        for v in [0.75, 0.80, 0.85]:
            rows.append(["sla_base_ratio_min", v, base["sla_semantic_prior"], base["sla_mirror_reject"], base["sla_layout_prior"], v])

        grid = {
            "base": base,
            "vary": {},
            "zip_only": True,
            "zip": [
                [
                    "ablation_factor",
                    "ablation_value",
                    "sla_semantic_prior",
                    "sla_mirror_reject",
                    "sla_layout_prior",
                    "sla_base_ratio_min",
                ],
                rows,
            ],
        }
        return grid

    raise ValueError(f"Unknown tierb: {tier}")


def run_sweep(args: argparse.Namespace) -> Dict[str, Any]:
    """Run a parameter sweep and write a summary CSV."""
    if not args.sweep_grid:
        raise ValueError("--sweep_grid is required when --sweep is set")
    grid = _load_sweep_grid(args.sweep_grid)
    runs, varied_keys = _expand_sweep_runs(grid)
    if len(runs) > int(args.sweep_max_runs):
        raise ValueError(f"Sweep has {len(runs)} runs > sweep_max_runs={args.sweep_max_runs}")

    in_dir = Path(args.folder).expanduser().resolve()
    sweeps_root = in_dir / "annotated" / "sweeps"
    sweeps_root.mkdir(parents=True, exist_ok=True)

    # deterministic shuffle (optional)
    if getattr(args, "tierb_seed", None) is not None:
        rng = np.random.default_rng(int(args.tierb_seed))
        order = rng.permutation(len(runs))
        runs = [runs[i] for i in order]

    rows = []
    for i, cfg in enumerate(runs):
        run_id = f"{args.sweep_id}_{i:03d}" if args.sweep_id else f"run{i:03d}"
        run_dir = sweeps_root / run_id
        if run_dir.exists() and args.sweep_skip_existing:
            print(f"[sweep] skip existing {run_id}")
            continue
        run_dir.mkdir(parents=True, exist_ok=True)

        run_args = argparse.Namespace(**vars(args))
        run_args._sweep_mode = True
        run_args._sweep_out_dir = str(run_dir)

        # apply overrides
        for k, v in cfg.items():
            setattr(run_args, k, v)

        # default visualize off unless explicitly set in cfg
        if "visualize" not in cfg:
            run_args.visualize = False

        # tier metadata (if using tierb presets)
        if getattr(args, "tierb", None):
            run_args.tierb_subtier = str(args.tierb).lower()
            if run_args.tierb_subtier == "b1":
                run_args.tierb_family = "shared"
            elif run_args.tierb_subtier == "b2":
                run_args.tierb_family = "matcher_tuned"
            else:
                run_args.tierb_family = "ablation"

        # canonicalize and validate for matcher
        canonicalize_run_args_for_matcher(
            run_args,
            _get_overridden_keys(cfg),
            strict=bool(getattr(args, "tierb_strict", True)),
        )

        # write run config (after canonicalization)
        cfg_path = run_dir / "run_config.json"
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(vars(run_args), f, indent=2, default=str)

        metrics = run_eval(run_args)
        row = {"run_id": run_id}
        # tier metadata (if present)
        row["tierb_subtier"] = getattr(run_args, "tierb_subtier", None)
        row["tierb_family"] = getattr(run_args, "tierb_family", None)
        row["ablation_factor"] = getattr(run_args, "ablation_factor", None)
        row["ablation_value"] = getattr(run_args, "ablation_value", None)
        row["matcher"] = getattr(run_args, "matcher", None)
        for k in varied_keys:
            row[k] = getattr(run_args, k, None)
        row.update(metrics)
        rows.append(row)

    # write sweep summary CSV
    out_csv = Path(args.sweep_out_csv)
    if not out_csv.is_absolute():
        # if user provided a relative path with dirs, honor it relative to CWD
        if out_csv.parent != Path("."):
            out_csv = out_csv
        else:
            out_csv = sweeps_root / out_csv
    import csv
    fieldnames = [
        "run_id",
        "tierb_subtier",
        "tierb_family",
        "ablation_factor",
        "ablation_value",
        "matcher",
    ] + [
        "overall_acc",
        "overall_precision",
        "idf_acc",
        "loc_mean",
        "loc_median",
        "images",
        "gt_present",
        "predicted",
        "correct",
        "runtime_s",
    ] + varied_keys
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Sweep complete. Wrote summary CSV: {out_csv}")
    return {"runs": len(rows), "out_csv": str(out_csv)}
    
    

def main():
    ap = argparse.ArgumentParser()
    # Example (layout prior):
    # python glint_pipeline_eval.py data/LabelledImages/dataset --labels data/label.txt \
    #   --matcher hybrid --template_mode bank \
    #   --cand_fallback --cand_target_raw 12 --cand_fallback_passes 4 --cand_fallback_percentiles 99.5,99,98.5,98 \
    #   --score2_mode contrast_support --support_M 30 --support_tol 0.10 --support_w 0.15 \
    #   --layout_prior --layout_lambda 0.25 --layout_mode image \
    #   --diag_candidate_recall --diag_make_plots --diag_recall_eps 10 --visualize
    # Example (SLA layout prior):
    # python glint_pipeline_eval.py data/LabelledImages/dataset --labels data/label.txt \
    #   --matcher sla --template_mode bank \
    #   --score2_mode contrast_support --support_M 30 --support_tol 0.10 --support_w 0.15 \
    #   --cand_fallback --cand_target_raw 12 --cand_fallback_passes 4 --cand_fallback_percentiles 99.5,99,98.5,98 \
    #   --ratio_tol 0.12 --pivot_P 8 --max_seeds 500 --matching greedy \
    #   --sla_layout_prior --sla_layout_lambda 0.25
    ap.add_argument("folder", type=str, help="Folder containing eye images")
    ap.add_argument("--labels", type=str, default=None, help="Path to label JSON (e.g., label.txt). If given, metrics are computed.")
    ap.add_argument("--match_tol", type=float, default=10.0, help="Pixel tolerance for counting a GT glint as correctly matched")

    ap.add_argument("--kernel", type=int, default=11, help="Top-hat kernel size (odd)")
    ap.add_argument("--percentile", type=float, default=99.7, help="Percentile threshold for candidate mask")
    ap.add_argument("--eps", type=float, default=6.0, help="Inlier threshold in pixels for template matching")
    ap.add_argument("--iters", type=int, default=4000, help="RANSAC iterations")
    ap.add_argument("--min_k", type=int, default=3, help="Points used per RANSAC hypothesis (3 recommended)")
    ap.add_argument("--max_pool", type=int, default=30, help="Max candidates (by score2) to pass into RANSAC")

    ap.add_argument("--min_area", type=int, default=8)
    ap.add_argument("--max_area", type=int, default=250)
    ap.add_argument("--min_circ", type=float, default=0.45)
    ap.add_argument("--min_maxI", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--median_ksize", type=int, default=3, help="Median blur kernel size (odd)")
    ap.add_argument("--denoise", type=int, default=1, help="Enable denoising (median blur) for enhancement")
    ap.add_argument("--denoise_k", type=int, default=0, help="Median blur kernel size override (0 uses median_ksize)")
    ap.add_argument("--clahe_clip", type=float, default=2.0, help="CLAHE clip limit")
    ap.add_argument("--clahe_tiles", type=int, default=8, help="CLAHE tile grid size")
    ap.add_argument("--clahe", type=int, default=1, help="Enable CLAHE enhancement")
    ap.add_argument("--enhance_mode", type=str, choices=["tophat", "dog", "highpass"], default="tophat",
                    help="Enhancement mode: tophat (default), dog, or highpass")
    ap.add_argument("--gamma", type=float, default=1.0, help="Gamma correction (>1 darkens, <1 brightens)")
    ap.add_argument("--unsharp", type=int, default=0, help="Enable unsharp mask (1=on, 0=off)")
    ap.add_argument("--unsharp_amount", type=float, default=1.0, help="Unsharp amount (strength)")
    ap.add_argument("--unsharp_sigma", type=float, default=1.0, help="Unsharp blur sigma")
    ap.add_argument("--minmax", type=int, default=1, help="Enable min-max normalization (0/1)")
    ap.add_argument("--enhance_enable", type=int, default=1, help="Enable enhance mode (tophat/dog/highpass) (0/1)")
    ap.add_argument("--clean_k", type=int, default=3, help="Morphology clean kernel size")
    ap.add_argument("--open_iter", type=int, default=1, help="Open iterations for candidate mask cleanup")
    ap.add_argument("--close_iter", type=int, default=0, help="Close iterations for candidate mask cleanup")
    ap.add_argument("--image_config", type=str, default=None, help="Path to image enhancement config JSON")
    ap.add_argument("--ui_settings", type=str, default=None, help="Path to preview UI settings JSON")
    ap.add_argument("--scale_min", type=float, default=0.6,
                    help="Lower bound on fitted scale relative to expected (default 0.6)")
    ap.add_argument("--scale_max", type=float, default=1.6,
                    help="Upper bound on fitted scale relative to expected (default 1.6)")
    ap.add_argument("--disable_scale_gate", action="store_true",
                    help="Disable scale prior gating inside RANSAC")
    ap.add_argument("--visualize", action="store_true", default=False,
                    help="Write annotated overlays (default: off)")
    ap.add_argument("--viz_metrics", action="store_true", default=False,
                    help="If labels provided, save metric visualizations (bar/box/hist)")
    ap.add_argument("--save_glints_npz", type=str, default=None,
                    help="Write glints.npz with per-image glint coordinates")
    ap.add_argument("--workers", type=int, default=0,
                    help="Number of worker processes for multiprocessing (0=off)")
    ap.add_argument("--template_mode", type=str, choices=["single", "bank"], default="single",
                    help="Template usage mode: single canonical template (default) or template bank")
    ap.add_argument("--template_bank_source", type=str, choices=["default", "custom"], default="default",
                    help="Template bank source (default: built-in)")
    ap.add_argument("--template_bank_path", type=str, default=None,
                    help="Path to custom template bank (.npy or .json) when template_bank_source=custom")
    ap.add_argument("--bank_select_metric", type=str, choices=["hybrid", "strict"], default="strict",
                    help="Template bank selection metric (default: strict)")
    ap.add_argument("--template_build_mode", type=str, choices=["median", "procrustes"], default="procrustes",
                    help="Template construction mode from labeled sets (canonical template)")
    ap.add_argument("--verbose_template", action="store_true",
                    help="Print template building diagnostics")
    ap.add_argument("--matcher", type=str, choices=["ransac", "star", "hybrid", "sla"], default="ransac",
                    help="Constellation matcher: ransac (default), star, hybrid (ransac+star), or sla")
    ap.add_argument("--matching", type=str, choices=["greedy", "hungarian"], default="greedy",
                    help="Candidate-template assignment method (default greedy; hungarian available)")
    ap.add_argument("--appearance_tiebreak", action="store_true", default=False,
                    help="Use appearance score sum to break RANSAC ties (off by default)")
    ap.add_argument("--roi_mode", type=str, choices=["none", "border"], default="none",
                    help="ROI gating for candidates before RANSAC (default none)")
    ap.add_argument("--roi_border_frac", type=float, default=0.06,
                    help="Border margin as fraction of min(H,W) when roi_mode=border")
    ap.add_argument("--roi_border_px", type=int, default=None,
                    help="Explicit border margin in pixels (overrides frac) when roi_mode=border")
    ap.add_argument("--pupil_roi", action="store_true", default=False,
                    help="Enable pupil-centric ROI crop (and optional gating) for candidates (off by default)")
    ap.add_argument("--pupil_roi_size", type=int, default=80,
                    help="Side length of square ROI around pupil center (px)")
    ap.add_argument("--pupil_roi_pad_mode", type=str, choices=["reflect", "constant", "edge"], default="reflect",
                    help="Padding mode when ROI extends outside the image")
    ap.add_argument("--pupil_roi_pad_value", type=int, default=0,
                    help="Pad value when pad_mode=constant")
    ap.add_argument("--pupil_roi_fail_policy", type=str, choices=["skip", "full_frame", "last_good"], default="skip",
                    help="If pupil center missing/invalid: skip frame, use full frame, or reuse last good center")
    ap.add_argument("--pupil_roi_debug", action="store_true", default=False,
                    help="Save ROI debug overlays under annotated/roi_debug")
    ap.add_argument("--pupil_npz", type=str, default=None,
                    help="Path to pupil NPZ (enables ROI centered on pupil; overrides pupil_roi/source)")
    ap.add_argument("--pupil_source", type=str, choices=["auto", "labels", "naive", "swirski", "npz", "none"], default="auto",
                    help="Pupil source: auto (labels if available else detector), labels, naive, swirski, or none")
    ap.add_argument("--pupil_axis_mode", type=str, choices=["auto", "radius", "diameter"], default="auto",
                    help="Interpretation of label ellipse axis values (auto/radius/diameter)")
    ap.add_argument("--pupil_dark_thresh", type=int, default=60,
                    help="Grayscale threshold for pupil detection when pupil ROI is enabled")
    ap.add_argument("--pupil_min_area", type=int, default=150,
                    help="Minimum area for pupil blob when pupil ROI is enabled")
    ap.add_argument("--pupil_rmin", type=float, default=0.3,
                    help="Minimum distance factor from pupil center to keep candidates")
    ap.add_argument("--pupil_rmax", type=float, default=1.2,
                    help="Maximum distance factor from pupil center to keep candidates")
    ap.add_argument("--pupil_fallback_center", type=str, choices=["image", "roi"], default="image",
                    help="Fallback center if pupil not found (image center or ROI center)")
    ap.add_argument("--pupil_method", type=str, choices=["naive", "swirski"], default="naive",
                    help="Pupil detector backend when pupil_source=auto (naive threshold CC or Swirski coarse+kmeans refine)")
    ap.add_argument("--pupil_radii", type=str, default="12,16,20,24,28,32",
                    help="Comma-separated radii for Swirski coarse search (pixels after scaling)")
    ap.add_argument("--pupil_sigma_frac", type=float, default=0.35,
                    help="Reserved for center prior weighting (Swirski); currently unused")
    ap.add_argument("--pupil_fail_open", action="store_true", default=True,
                    help="If pupil detection implausible, skip gating (fail open)")
    ap.add_argument("--pupil_force_gate", action="store_true", default=False,
                    help="Force pupil gating even if it drops candidates below min_k")
    ap.add_argument("--debug_pupil", action="store_true", default=False,
                    help="Print per-image pupil gating diagnostics to stdout")
    ap.add_argument("--auto_scale", action="store_true", default=True,
                    help="Scale pixel-based parameters per image based on width/ref_width")
    ap.add_argument("--no_auto_scale", dest="auto_scale", action="store_false",
                    help="Disable per-image parameter scaling")
    ap.add_argument("--ref_width", type=int, default=640,
                    help="Reference width for auto-scaling (default 640)")
    ap.add_argument("--min_kernel", type=int, default=3,
                    help="Minimum kernel size (odd) when auto-scaling")
    ap.add_argument("--min_inliers", type=int, default=3, help="Minimum inliers to accept a hypothesis")
    ap.add_argument("--vote_M", type=int, default=8, help="Star matcher: shortlist size per template point")
    ap.add_argument("--vote_ratio_tol", type=float, default=0.12, help="Star matcher: log-scale tolerance")
    ap.add_argument("--vote_max_hyp", type=int, default=2000, help="Star matcher: max hypotheses to verify")
    ap.add_argument("--vote_w_score2", type=float, default=0.0, help="Star matcher: score2 weight in vote ranking")
    ap.add_argument("--ratio_tol", type=float, default=0.12, help="SLA matcher: log-ratio tolerance")
    ap.add_argument("--pivot_P", type=int, default=8, help="SLA matcher: top-P pivots by score2")
    ap.add_argument("--max_seeds", type=int, default=500, help="SLA matcher: max seed hypotheses")
    ap.add_argument("--grow_resid_max", type=float, default=None, help="SLA matcher: max median residual when growing (default eps)")
    ap.add_argument("--sla_w_seed_score2", type=float, default=1.0,
                    help="SLA matcher: weight for seed score2 sum")
    ap.add_argument("--sla_w_seed_geom", type=float, default=1.0,
                    help="SLA matcher: weight for seed initial residual")
    ap.add_argument("--max_seeds_per_pivot", type=int, default=80,
                    help="SLA matcher: max seeds kept per observed pivot")
    ap.add_argument("--sla_adaptive_ratio_tol", action="store_true", default=True,
                    help="SLA matcher: adapt ratio_tol based on pool size (default on)")
    ap.add_argument("--no_sla_adaptive_ratio_tol", dest="sla_adaptive_ratio_tol", action="store_false",
                    help="SLA matcher: disable adaptive ratio_tol")
    ap.add_argument("--sla_ratio_tol_min", type=float, default=0.06,
                    help="SLA matcher: minimum adaptive ratio tolerance")
    ap.add_argument("--sla_ratio_tol_refN", type=int, default=12,
                    help="SLA matcher: reference N for adaptive ratio tolerance")
    ap.add_argument("--sla_scale_min", type=float, default=0.2,
                    help="SLA matcher: minimum allowed scale for hypothesis")
    ap.add_argument("--sla_scale_max", type=float, default=5.0,
                    help="SLA matcher: maximum allowed scale for hypothesis")
    ap.add_argument("--sla_g0_top2", action="store_true", default=False,
                    help="SLA matcher: hard reject if G0 not in top-2 highest matched points")
    ap.add_argument("--sla_semantic_prior", action="store_true", default=False,
                    help="SLA matcher: enable semantic geometry prior")
    ap.add_argument("--sla_semantic_lambda", type=float, default=1.5,
                    help="SLA matcher: semantic penalty weight")
    ap.add_argument("--sla_semantic_mode", type=str, choices=["full", "top_only"], default="full",
                    help="SLA semantic prior components. full uses top+base+sides (assumes K=5). "
                         "top_only uses only the top rule (supports K!=5).")
    ap.add_argument("--sla_semantic_hard", action="store_true", default=False,
                    help="SLA matcher: hard veto on semantic violations")
    ap.add_argument("--sla_mirror_reject", action="store_true", default=True,
                    help="SLA matcher: reject/penalize mirrored constellations")
    ap.add_argument("--sla_top2_margin", type=float, default=0.0,
                    help="SLA matcher: slack (px) for G0 top-2 check")
    ap.add_argument("--sla_base_ratio_min", type=float, default=0.80,
                    help="SLA matcher: base edge minimum ratio vs max edge")
    ap.add_argument("--sla_side_margin", type=float, default=0.0,
                    help="SLA matcher: slack (px) for left/right check")
    ap.add_argument("--sla_semantic_debug", action="store_true", default=False,
                    help="SLA matcher: print semantic debug for first few frames")
    ap.add_argument("--post_id_resolve", action="store_true", default=False,
                    help="Enable post-match identity resolution (G0..G4 relabeling)")
    ap.add_argument("--id_lambda", type=float, default=2.0,
                    help="Identity resolution: weight for layout prior")
    ap.add_argument("--id_gamma", type=float, default=1.0,
                    help="Identity resolution: weight for pairwise ratio penalty")
    ap.add_argument("--id_eta", type=float, default=0.10,
                    help="Identity resolution: weight for appearance tie-break (score2)")
    ap.add_argument("--id_tau", type=float, default=None,
                    help="Identity resolution: robust cap for geom residual (px); default 2*eps")
    ap.add_argument("--id_layout_mode", type=str, choices=["image", "template"], default="image",
                    help="Identity resolution: layout mode (image or template)")
    ap.add_argument("--id_debug", action="store_true", default=False,
                    help="Identity resolution: print debug for first few frames")
    ap.add_argument("--sla_layout_prior", action="store_true", default=False,
                    help="Enable soft layout prior for SLA hypothesis ranking (G0 in top-2 highest)")
    ap.add_argument("--sla_layout_lambda", type=float, default=0.25,
                    help="SLA layout prior: penalty weight (pixels)")
    ap.add_argument("--sla_layout_mode", type=str, choices=["image"], default="image",
                    help="SLA layout prior: coordinate mode (image only)")
    ap.add_argument("--sla_layout_debug", action="store_true", default=False,
                    help="SLA layout prior: overlay penalty for chosen hypothesis")
    ap.add_argument("--score2_mode", type=str, choices=["heuristic", "contrast", "contrast_support", "ml_cc"], default="heuristic",
                    help="Score2 mode: heuristic (default), contrast, contrast_support, ml_cc")
    ap.add_argument("--ml_model_path", type=str, default=None,
                    help="Path to ML model joblib for score2_mode=ml_cc")
    ap.add_argument("--ml_eps_label", type=float, default=5.0,
                    help="Labeling epsilon (px) for ML training scripts (not used at inference)")
    ap.add_argument("--ml_use_patch", action="store_true", default=False,
                    help="Reserved for future patch-based ML scoring (unused)")
    ap.add_argument("--contrast_r_inner", type=int, default=3, help="Inner radius for local contrast")
    ap.add_argument("--contrast_r_outer1", type=int, default=5, help="Outer ring inner radius for local contrast")
    ap.add_argument("--contrast_r_outer2", type=int, default=8, help="Outer ring outer radius for local contrast")
    ap.add_argument("--dog_sigma1", type=float, default=1.0, help="DoG sigma1")
    ap.add_argument("--dog_sigma2", type=float, default=2.2, help="DoG sigma2")
    ap.add_argument("--support_M", type=int, default=30, help="Top-M candidates for support voting")
    ap.add_argument("--support_tol", type=float, default=0.10, help="Relative tolerance for support distances")
    ap.add_argument("--support_w", type=float, default=0.15, help="Support weight in score2")
    ap.add_argument("--layout_prior", action="store_true", default=False,
                    help="Enable layout prior penalty on hypotheses")
    ap.add_argument("--layout_lambda", type=float, default=0.25,
                    help="Weight of layout penalty relative to geometric cost")
    ap.add_argument("--layout_mode", type=str, choices=["image", "pca"], default="image",
                    help="Layout prior mode: image or pca")
    ap.add_argument("--layout_debug", action="store_true", default=False,
                    help="Overlay layout prior penalty for chosen hypothesis")
    ap.add_argument("--cand_fallback", action="store_true", default=False,
                    help="Enable adaptive fallback for candidate detection (optional)")
    ap.add_argument("--cand_target_raw", type=int, default=8,
                    help="Minimum desired raw candidates before matching")
    ap.add_argument("--cand_fallback_passes", type=int, default=3,
                    help="Number of fallback passes")
    ap.add_argument("--cand_fallback_percentiles", type=str, default="99,98,97",
                    help="Comma-separated percentiles for fallback passes (descending strictness)")
    ap.add_argument("--cand_fallback_kernel_add", type=int, default=0,
                    help="Add to top-hat kernel size per fallback pass")
    ap.add_argument("--cand_fallback_debug", action="store_true", default=False,
                    help="Append fallback diagnostics to overlay title text")
    ap.add_argument("--cand_merge_eps", type=float, default=2.0,
                    help="Merge radius (px) when combining fallback candidates")
    ap.add_argument("--diag_candidate_recall", action="store_true", default=False,
                    help="Enable candidate recall diagnostic (optional)")
    ap.add_argument("--diag_out_csv", type=str, default="candidate_recall.csv",
                    help="CSV filename for candidate recall diagnostic")
    ap.add_argument("--diag_make_plots", action="store_true", default=False,
                    help="Save candidate recall diagnostic plots")
    ap.add_argument("--diag_plot_path", type=str, default="candidate_recall_plots.png",
                    help="Plot filename for candidate recall diagnostic")
    ap.add_argument("--diag_recall_eps", type=float, default=None,
                    help="Recall tolerance in pixels (default: use --eps)")
    ap.add_argument("--temporal_prior", action="store_true", default=False,
                    help="Use a temporal prior for template bank selection (requires previous frame transform)")
    ap.add_argument("--temporal_lambda", type=float, default=0.25,
                    help="Temporal prior weight")
    ap.add_argument("--temporal_w_scale", type=float, default=1.0,
                    help="Temporal prior weight for scale change")
    ap.add_argument("--temporal_w_rot", type=float, default=1.0,
                    help="Temporal prior weight for rotation change (radians)")
    ap.add_argument("--temporal_w_trans", type=float, default=1.0,
                    help="Temporal prior weight for translation change (px)")
    ap.add_argument("--sweep", action="store_true", default=False,
                    help="Run sweep mode instead of a single run")
    ap.add_argument("--sweep_grid", type=str, default=None,
                    help="JSON string or path to .json defining sweep grid")
    ap.add_argument("--sweep_out_csv", type=str, default="sweep_summary.csv",
                    help="Sweep summary CSV filename")
    ap.add_argument("--sweep_id", type=str, default=None,
                    help="Optional prefix for sweep run IDs")
    ap.add_argument("--sweep_keep_reports", action="store_true", default=False,
                    help="Keep per-run report.txt and per-image/per-subject CSVs")
    ap.add_argument("--sweep_skip_existing", action="store_true", default=False,
                    help="Skip sweep runs if run folder already exists")
    ap.add_argument("--sweep_max_runs", type=int, default=500,
                    help="Safety cap for total sweep runs")
    ap.add_argument("--tierb", type=str, choices=["b1", "b2", "b3"], default=None,
                    help="Generate Tier B sweep presets (b1/b2/b3)")
    ap.add_argument("--tierb_out_json", type=str, default=None,
                    help="If set, write generated Tier B sweep JSON to this path")
    ap.add_argument("--tierb_id", type=str, default=None,
                    help="Optional prefix for Tier B run IDs (used if sweep_id not set)")
    ap.add_argument("--tierb_matchers", type=str, default=None,
                    help="Comma-separated matcher list for Tier B (default hybrid,sla)")
    ap.add_argument("--tierb_strict", action="store_true", default=True,
                    help="Tier B strict mode (reject invalid or SLA-only knobs in B1)")
    ap.add_argument("--tierb_seed", type=int, default=None,
                    help="Optional seed for deterministic run ordering")
    ap.add_argument("--self_test", action="store_true", help="Run synthetic self-test and exit")
    args = ap.parse_args()

    if args.ui_settings:
        _apply_ui_settings(args, args.ui_settings)

    if args.self_test:
        _self_test_star_vs_ransac(args)
        return

    if args.tierb is not None:
        if args.sweep_grid is not None:
            print("Warning: --sweep_grid provided; ignoring --tierb presets.")
        else:
            grid = make_tierb_grid(args.tierb, args)
            if args.tierb_out_json:
                Path(args.tierb_out_json).write_text(json.dumps(grid, indent=2), encoding="utf-8")
            if args.sweep_id is None and args.tierb_id is not None:
                args.sweep_id = args.tierb_id
            args.sweep = True
            args.sweep_grid = json.dumps(grid)
            # print run plan summary
            runs, varied_keys = _expand_sweep_runs(grid)
            print(f"[tierb] subtier={args.tierb} runs={len(runs)} varied_keys={varied_keys[:6]}{'...' if len(varied_keys)>6 else ''}")
            for i, cfg in enumerate(runs[:3]):
                print(f"[tierb] run{i}: {cfg}")

    if args.sweep:
        return run_sweep(args)
    return run_eval(args)

if __name__ == "__main__":
    main()
