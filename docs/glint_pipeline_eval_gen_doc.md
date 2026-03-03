# Glint Pipeline (glint_pipeline_eval_gen.py) - Algorithm Documentation

## Pipeline Overview
This documentation traces the per-image pipeline implemented in `glint_pipeline_eval_gen.py`, using the stage names and function names from the code. The pipeline begins by reading an image, converting to grayscale, and scaling parameters with `scale_params_for_image` (auto-scale via `--ref_width` unless `--no_auto_scale`). If pupil ROI is enabled (`--pupil_roi` or `--pupil_npz`), the pupil center is obtained via `detect_pupil_center_for_frame` using `pupil_source` (labels, `npz`, `naive`, `swirski`, or `auto`). ROI center and fail policy are resolved by `resolve_pupil_roi_center` and the image is cropped using `compute_pupil_roi` with padding.

**Grayscale preprocessing** is part of `enhance_for_glints`: optional median denoise (`--denoise`, `--denoise_k` or `--median_ksize`), optional gamma correction (`--gamma`), and optional unsharp masking (`--unsharp`, `--unsharp_amount`, `--unsharp_sigma`). **Enhancement** is selected by `--enhance_mode`: `tophat` uses morphological white top-hat, `dog` uses Gaussian difference, and `highpass` subtracts a blur; then optional CLAHE (`--clahe`, `--clahe_clip`, `--clahe_tiles`). If enabled, the enhanced image is normalized to 0-255 with `cv2.normalize` (`--minmax`).

**Over-detection thresholding** is done by `threshold_candidates`: a global percentile threshold (`--percentile`) on the enhanced image, followed by morphology open/close (`--clean_k`, `--open_iter`, `--close_iter`). **Candidate extraction** uses connected components (`cv2.connectedComponentsWithStats`) and per-blob features in `extract_cc_candidates_and_features`. A rule prefilter (`rule_prefilter`) applies hard thresholds (`--min_area`, `--max_area`, `--min_circ`, `--min_maxI`).

**Appearance scoring (score2)** is computed in `detect_candidates_one_pass`. The heuristic score (`add_appearance_score`) is:

`score2 = w_maxI*max_intensity + w_circ*circularity + w_peak*peakiness - w_sym*radial_sym`

with defaults `w_maxI=1.0`, `w_circ=15.0`, `w_peak=2.0`, `w_sym=5.0`.

For `score2_mode=contrast`, the base score is:

`base_score = norm(score2_heur) + 0.6*norm(contrast) + 0.4*norm(dog)`

where `contrast = inner_mean - ring_mean` from `compute_local_contrast`, and `dog` is `max(0, DoG)` at the candidate location.

For `score2_mode=contrast_support`, candidates receive **support votes** from the expected template distance set (`compute_expected_pairwise_distances`), which is computed over **normalized** template constellations. Only the top `support_M` candidates by `base_score` participate. For each candidate pair `(i,j)` with distance `d`, if any expected distance `d_expected` satisfies `|d_expected - d| <= support_tol * d_expected`, both candidates get a vote. The final score is:

`cand_score2 = base_score * (1 + support_w * support)`

where `support` is the vote count per candidate.

For `score2_mode=ml_cc`, features listed by `cc_feature_names()` are fed to a joblib model (`--ml_model_path`).

**Candidate fallback** (`--cand_fallback`) runs additional detection passes when the initial raw count is below `--cand_target_raw`. Each pass uses a percentile from `cand_fallback_percentiles` (cycled if fewer than `cand_fallback_passes`) and increases the enhancement kernel by `cand_fallback_kernel_add * pass_index`. Candidates across passes are merged via `merge_candidates`: points within `cand_merge_eps` are deduplicated with score-sorted priority. Fallback stops when the merged count reaches `cand_target_raw` or when all passes are consumed.

**Candidate pooling** keeps the top `--max_pool` by score2. **ROI gating** then optionally removes candidates outside the inner border rectangle (`filter_candidates_roi`, `--roi_mode border`, `--roi_border_frac`, `--roi_border_px`). If pupil ROI is active, gating can further restrict candidates to an annulus around the pupil: `gate_candidates_by_pupil` keeps candidates with distance in `[rmin*r, rmax*r]` (`--pupil_rmin`, `--pupil_rmax`). For pupil failures, behavior follows `--pupil_fail_open`, `--pupil_force_gate`, and `--pupil_fallback_center`.

**Template selection** depends on `--template_mode`. `single` uses a canonical template derived from labeled constellations via `build_template_median` or `build_template_from_labeled_sets` (`--template_build_mode median|procrustes`). `bank` loads multiple templates (`load_template_bank`), then `run_matcher_for_template` is called per template and the best is chosen by `score_match_result` with `--bank_select_metric` (`strict` or `hybrid`).

**Constellation matching** calls one of `ransac_constellation`, `startracker_constellation`, `hybrid_constellation_match`, or `sla_pyramid_constellation` (selected by `--matcher`). All matchers use either greedy or Hungarian assignment (`--matching`) to finalize matches. Optional priors include a layout penalty (`layout_penalty_g0_top2` or `sla_layout_penalty_g0_top2`) and semantic penalties (`semantic_penalty`) inside SLA. After matching, `post_id_resolve` may reorder glint identities using `resolve_identity_permutation` (only when a full 5-point match exists).

**Evaluation** (if labels are provided) computes identity-aware metrics in `evaluate_matches` and identity-free metrics in `evaluate_identity_free`. Accuracy is computed as `correct / present` using the `--match_tol` threshold; localization error is the mean and median Euclidean error over present-and-predicted matches, and identity-free errors are computed after a tiny optimal assignment between GT and predictions.

## Exact Definitions (from Code)

### Enhancement (`enhance_for_glints`)
Given a grayscale image `gray` (uint8), the enhancement pipeline is:

1. **Median denoise** (optional): if `denoise` and (`denoise_k > 0` or `median_ksize > 0`), set `k_med = denoise_k if denoise_k > 0 else median_ksize`, make it odd, and apply `cv2.medianBlur(gray, k_med)`.
2. **Gamma** (optional): if `gamma != 1.0`, compute `work_f = (work/255) ** (1/gamma)` and rescale to 0-255 (uint8).
3. **Unsharp mask** (optional): if `unsharp != 0` and `unsharp_amount > 0`, compute `blur = cv2.GaussianBlur(work, (0,0), unsharp_sigma)` and update `work = addWeighted(work, 1+amount, blur, -amount, 0)`.
4. **Enhance mode** (optional, controlled by `enhance_enable`):
   - `enhance_mode="dog"`: `g1 = GaussianBlur(work, sigma1)`, `g2 = GaussianBlur(work, sigma2)`, `enh_base = subtract(g1, g2)` (with `sigma1/sigma2` swapped if needed so `sigma2 >= sigma1`).
   - `enhance_mode="highpass"`: `blur = cv2.blur(work, (k,k))` where `k` is odd and at least 3, `enh_base = subtract(work, blur)`.
   - default (`"tophat"`): `enh_base = morphologyEx(work, MORPH_TOPHAT, elliptical_kernel(kernel_size))`.
5. **CLAHE** (optional): if `clahe_enable` and `clahe_clip > 0` and `clahe_tiles > 0`, apply `cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tiles,clahe_tiles)).apply(enh_base)`.
6. **Min-max normalization** (optional): if `minmax != 0` and `enh.max() > 0`, apply `cv2.normalize(enh, None, 0, 255, NORM_MINMAX).astype(uint8)`.

### Candidate Extraction (`threshold_candidates`, `extract_cc_candidates_and_features`, `rule_prefilter`)
1. **Thresholding** (`threshold_candidates`): compute `thr = percentile(enh, percentile)` and form mask `mask = 255 * (enh >= thr)`.
2. **Morphology cleanup**: build an elliptical kernel of odd size `clean_k`, then apply:
   - Open: `morphologyEx(mask, MORPH_OPEN, kernel, iterations=open_iter)` if `open_iter > 0`.
   - Close: `morphologyEx(mask, MORPH_CLOSE, kernel, iterations=close_iter)` if `close_iter > 0`.
3. **Connected components**: run `cv2.connectedComponentsWithStats(mask, connectivity=8)` and compute per-component features (aligned with `cc_feature_names()`):
   - Geometry: `area`, `perimeter`, `circularity = 4*pi*area/(perimeter^2)`, `bbox_w`, `bbox_h`, `aspect = bbox_w/bbox_h`, `solidity = area/area(convex_hull)`, `eccentricity` (from eigenvalues of coordinate covariance).
   - Intensities: `mean_int`, `max_int` (on `gray`), plus `p95_int` and `std_int` (on enhanced intensities).
   - Duplicated gray stats (as implemented): `mean_gray`, `max_gray` (also on `gray`).
   - Local context: `ring_minus_inner = -compute_local_contrast(gray, cx, cy, r_inner, r_outer1, r_outer2)`, `dog` value at `(cx,cy)` from `compute_dog`.
   - Position: `x_norm = cx/W`, `y_norm = cy/H`, `dist_border_norm = min(cx,cy,W-1-cx,H-1-cy)/min(W,H)`.
4. **Rule prefilter** (`rule_prefilter`): keep blobs that satisfy:
   - `min_area <= area <= max_area`
   - `circularity >= min_circ`
   - `max_intensity >= min_maxI`

### Score2 Modes (`detect_candidates_one_pass`)
Let `score2_heur` be the heuristic score from `add_appearance_score`:

`score2_heur = 1.0*max_intensity + 15.0*circularity + 2.0*peakiness - 5.0*radial_sym`

where `peakiness = peakiness_at(cx,cy,gray)` computes `max(core) - mean(ring)`, and `radial_sym = std(samples_on_circle)` from `radial_symmetry` (lower is better, hence the subtraction).

Normalization helper (`normalize_scores`):

`norm(x) = (x - min(x)) / (max(x) - min(x))`, and returns all zeros if `max-min` is tiny.

- `score2_mode="heuristic"`: `cand_score2 = score2_heur`.
- `score2_mode="contrast"`: compute `contrast_vals = compute_local_contrast(gray, x, y, r_inner, r_outer1, r_outer2)` per candidate, and `dog_vals = max(0, DoG(x,y))`. Then:

  `base_score = norm(score2_heur) + 0.6*norm(contrast_vals) + 0.4*norm(dog_vals)`

  and `cand_score2 = base_score`.
- `score2_mode="contrast_support"`: start from the same `base_score` and compute support votes from expected normalized template distances `d_expected` (from `compute_expected_pairwise_distances`). Only the top `M = support_M` candidates by `base_score` participate.
  - Normalize their coordinates: `Cn = normalize(C)` (zero-mean, unit RMS).
  - For each pair `(i,j)`, compute `d = ||Cn[i]-Cn[j]||`. If `any(|d_expected - d| <= support_tol * d_expected)`, increment `support[i]` and `support[j]`.
  - Final score (exact code):

    `cand_score2 = base_score * (1 + support_w * support)`

  where `support` is the (unnormalized) vote count per candidate.
- `score2_mode="ml_cc"`: `cand_score2 = model.predict_proba(feature_matrix)[:,1]` (or `predict`) using the CC feature matrix.

### Candidate Fallback (`cand_fallback`, `merge_candidates`)
If `cand_fallback` is enabled and the initial raw count `cand_raw_pass0 < cand_target_raw`, additional passes are run:

1. Parse `cand_fallback_percentiles` as a list of floats. For pass `i=1..cand_fallback_passes` choose `perc = fallback_percentiles[min(i-1, len(list)-1)]`.
2. Increase the enhancement kernel by `kernel_add = cand_fallback_kernel_add * i` (this is applied inside `detect_candidates_one_pass` by updating `params["kernel_eff"]`).
3. Run `detect_candidates_one_pass(..., percentile_override=perc, kernel_add=kernel_add)` and append results.
4. Merge across passes with `merge_candidates(..., merge_eps=cand_merge_eps)`:
   - Stack all candidate points and scores, then sort by primary key `-score` (then `x`, then `y`).
   - Iterate in sorted order and keep a point only if it is farther than `merge_eps` from all already-kept points.
5. Stop fallback early when the merged count reaches `cand_target_raw`; otherwise stop after `cand_fallback_passes` passes.

### ROI Options (border ROI and pupil ROI)
- **Border ROI** (`filter_candidates_roi`): if `roi_mode="border"`, compute `margin = roi_border_px if set else int(roi_border_frac * min(H,W))` and keep candidates with `x in [margin, W-margin)` and `y in [margin, H-margin)`.
- **Pupil ROI crop** (`compute_pupil_roi`): crop a fixed-size square ROI centered at `round(pupil_center)` with padding when the ROI extends beyond the image.
  - Padding modes: `reflect` -> `BORDER_REFLECT_101`, `edge` -> `BORDER_REPLICATE`, `constant` -> `BORDER_CONSTANT` (value `pupil_roi_pad_value`).
  - Fail policy (`resolve_pupil_roi_center`): `skip`, `full_frame`, or `last_good` (use prior valid center if present).
  - Note: in `run_eval`, providing `--pupil_npz` forces `pupil_roi=True`, `pupil_source="npz"`, and `pupil_roi_fail_policy="full_frame"`.
- **Pupil annulus gate** (`gate_candidates_by_pupil`): keep candidate `p` if `rmin*r <= ||p - pupil_center|| <= rmax*r` where `r` is the pupil radius.

### Templates (construction, bank selection)
- `build_template_median`: normalize each labeled constellation (`normalize`: zero-mean, unit RMS), stack them, take per-point median, and normalize again.
- `build_template_from_labeled_sets`: generalized Procrustes (labels assumed consistent, no permutation search). Iteratively align each constellation to `ref` via `estimate_similarity`, average aligned shapes, normalize, and stop when `ref_delta < tol` or `iters` is reached.
- Template bank loading (`load_template_bank`): supports `default` JSON bank and `custom` `.npy` or `.json`; each template is normalized.
- Bank selection (`score_match_result`): lexicographic score.
  - `strict`: `(inliers, -median_residual, -mean_residual, app_val)`.
  - `hybrid`: `(inliers, pair_consistency, -median_residual, app_val)` where `pair_consistency = 1/(1+CV(residuals))`.

### Matchers and Assignment
All matchers ultimately produce `matches = [(template_idx, cand_idx, dist_px), ...]` using either:

- `match_template_to_candidates_greedy`: for each template point in order, take the nearest unused candidate within `eps`.
- `match_template_to_candidates_hungarian`: solve a 1-to-1 assignment minimizing total distance (SciPy `linear_sum_assignment` if available, else a brute-force tiny solver), then keep only pairs within `eps`.

**RANSAC** (`ransac_constellation`)
- Hypotheses: sample `min_k` template points and `min_k` candidate points, fit similarity via `estimate_similarity`.
- Verification: transform full template, then assign matches (greedy/Hungarian) within `eps`.
- Selection: maximize inliers; ties use either appearance sum (`appearance_tiebreak`) or lower mean residual; optional `layout_prior` adds `layout_lambda * layout_penalty_g0_top2` to the error when a full 5-point match is present.

**Star tracker** (`startracker_constellation`)
- Hypotheses: build a voting matrix `V` from pairwise distance ratios, shortlist `vote_M` candidates per template point by `(V[i,c], score2[c])`, then generate 2-point seeds (template pair + candidate pair) and fit similarity.
- Verification/selection: same assignment stage and tie-breaking as RANSAC; hypotheses limited by `vote_max_hyp`.

**Hybrid** (`hybrid_constellation_match`)
- Runs both RANSAC and star tracker and returns the higher-scoring result (using `score_match_result(..., bank_select_metric="strict")`).

**SLA** (`sla_pyramid_constellation`)
- Ratio-index seeds and pyramid growth as specified in Algorithm 2. Final selection minimizes:

  `total_cost = med_res + sla_layout_lambda*lp + sla_semantic_lambda*sem.p_total + mirror_pen`

  where `mirror_pen = 10` for reflected solutions when `sla_mirror_reject` is enabled (unless hard veto mode).

### Priors (semantic/layout/mirror)
- Layout prior for RANSAC/star: `layout_penalty_g0_top2(points_xy)` penalizes solutions where G0 is not among the top-2 highest points (image mode: smallest y). It enters as `err_eff = err + layout_lambda * lp`.
- Layout prior for SLA: `sla_layout_penalty_g0_top2(matched_xy_by_glint)` computes a similar penalty but tolerates missing glints; it enters via `sla_layout_lambda` inside `total_cost`.
- Semantic prior for SLA (`semantic_penalty`): returns `p_total = p_top + p_base + p_sides` (unweighted) where:
  - `p_top` enforces G0 top-2 in y (with margin `sla_top2_margin`).
  - `p_base` penalizes a too-short G2-G3 "base" edge relative to max pairwise distance (`sla_base_ratio_min`) and penalizes G2/G3 being in the top-2.
  - `p_sides` penalizes left/right ordering of glints relative to the median x (with margin `sla_side_margin`).
  It enters the SLA score as `sla_semantic_lambda * p_total` and can be used as a hard veto with `sla_semantic_hard`.
- Mirror rejection (`_is_reflection`): detects reflection by sign change of the oriented triangle (0,1,2) between template and prediction. In SLA, reflections can be hard-vetoed (`sla_semantic_hard`) or soft-penalized (`mirror_pen = 10`).

### Evaluation Metrics (`evaluate_matches`, `evaluate_identity_free`)
- **Identity-aware** (`evaluate_matches`): a GT glint is "present" if not NaN. A prediction for glint `i` exists if template index `i` appears in `matches`. A glint is **correct** if `||pred_i - gt_i|| <= match_tol`. Reports:
  - `match_accuracy = correct_count / present_count`
  - `precision = correct_count / pred_count`
  - `loc_err_mean/median`: mean/median over present+predicted glints (not only "correct" ones).
- **Identity-free** (`evaluate_identity_free`): ignores glint IDs. Solves a tiny optimal assignment between present GT points and predicted points, counts assigned pairs with distance `<= match_tol` as correct, and reports mean/median localization error over assigned pairs (plus correct-only variants).

## Algorithm 1 - Full Pipeline (per image)
```text
Input: image I, args, optional labels, optional pupil_npz
Output: glint_xy (Nx2), template_xy (Nx2), optional metrics

1. Read image, gray = cv2.cvtColor(I, BGR2GRAY)
2. params = scale_params_for_image(args, w, h)
3. If pupil ROI enabled:
   a) (pcx,pcy,pr,detected,ok,src) = detect_pupil_center_for_frame(...)
   b) roi_decision = resolve_pupil_roi_center(center, radius, W,H, policy, last_good)
   c) If action == "skip": continue (or return None in mp)
   d) If action == "use": roi_info = compute_pupil_roi(gray_full, center, size, pad_mode, pad_value)
      gray = roi_info.roi_img
4. cand_xy, rows, cand_score2, cand_raw, support = detect_candidates_one_pass(gray, params, args, d_expected)
5. If cand_fallback and cand_raw < cand_target_raw:
   For each pass i in 1..cand_fallback_passes:
     a) perc = fallback_percentiles[min(i-1, end)]
     b) kernel_add = cand_fallback_kernel_add * i
     c) detect_candidates_one_pass(gray, ..., percentile_override=perc, kernel_add=kernel_add)
     d) merge_candidates(..., merge_eps=cand_merge_eps)
     e) Stop if merged count >= cand_target_raw
6. If ROI was active: map_points_to_full(cand_xy_raw, roi_info.offset_x, roi_info.offset_y)
7. Pool candidates: keep top max_pool by score2
8. ROI gating: filter_candidates_roi (border)
9. If pupil ROI enabled and gating allowed:
   a) choose gate center/radius (pupil_ok or fallback)
   b) mask = gate_candidates_by_pupil(cand_xy, center, radius, rmin, rmax)
   c) apply mask (unless fail_open or force_gate overrides)
10. Template selection:
    a) If template_mode=single: res = run_matcher_for_template(template, cand_xy, ...)
    b) If template_mode=bank: run_matcher_for_template for each template, select best via score_match_result
11. If res exists: extract (s,R,t,matches,T_hat), assign glint_xy from matches
12. Optional post_id_resolve (full 5-point matches only)
13. Optional evaluation: evaluate_matches (identity-aware) and evaluate_identity_free
```

## Algorithm 2 - SLA Matcher (`sla_pyramid_constellation`)
```text
Input: template T (Kx2), candidates C (Nx2), score2, eps, ratio_index
Output: best (s, R, t, matches, T_hat, app_sum)

1. Build ratio index if missing: for each pivot i in T, store (log_ratio, ratio, j, k)
2. Set eff_ratio_tol:
   eff_ratio_tol = ratio_tol * sqrt(sla_ratio_tol_refN / max(N,1))
   clamp to [sla_ratio_tol_min, ratio_tol] if sla_adaptive_ratio_tol
3. Select top pivot candidates: piv_idx = top pivot_P by score2
4. Seed generation (per pivot candidate cp):
   For all pairs (a,b) of candidates:
     r_obs = min(|C[cp]-C[a]| / |C[cp]-C[b]|, inverse)
     For each template pivot ti:
       For each (log_rt, r_t, tj, tk) within eff_ratio_tol:
         Consider two seed mappings: (ti,tj,tk) -> (cp,a,b) and (cp,b,a)
         Fit similarity on 3 points; compute seed_resid
         If sla_semantic_prior: compute semantic_penalty and mirror check
           If sla_semantic_hard: veto if p_top>=1, p_sides>0.5, or d23 < sla_base_ratio_min*dmax
           If sla_mirror_reject and mirror: veto under hard mode
         Seed score = sla_w_seed_score2*(score2[cp]+score2[a]+score2[b]) - sla_w_seed_geom*seed_resid
   Keep max_seeds_per_pivot and then global top max_seeds

5. For each seed:
   a) Fit similarity; initialize matches with 3 seed pairs
   b) Grow: for each unmatched template point, add nearest candidate within eps
      Refit similarity; accept if median residual <= grow_resid_max
   c) Final assignment: greedy or Hungarian within eps
   d) Fit final similarity on matched pairs; reject if:
      s outside [sla_scale_min, sla_scale_max] or median residual > eps or max residual > 2*eps
   e) Apply semantic penalty and mirror rejection:
      sem = semantic_penalty(T_hat, eps, sla_top2_margin, sla_base_ratio_min, sla_side_margin)
      If sla_mirror_reject and mirror and sla_semantic_hard: veto
      If sla_semantic_hard and semantic violations: veto
   f) Optional G0-top2 gate (sla_g0_top2)
   g) Compute total_cost:
      total_cost = err + sla_layout_lambda*layout_penalty + sla_semantic_lambda*sem.p_total + mirror_pen
      mirror_pen = 10 if mirror and not hard
   h) Keep best by: max inliers, then min total_cost, then max app_sum, then min err

6. Return best if inliers >= min_inliers
```

## Key CLI parameters by stage (important subset)

| Stage | Key parameters |
|---|---|
| Preprocess + Enhancement | `--denoise`, `--denoise_k`, `--median_ksize`, `--gamma`, `--unsharp`, `--unsharp_amount`, `--unsharp_sigma`, `--enhance_mode`, `--kernel`, `--dog_sigma1`, `--dog_sigma2`, `--clahe`, `--clahe_clip`, `--clahe_tiles`, `--minmax`, `--enhance_enable` |
| Threshold + CC | `--percentile`, `--clean_k`, `--open_iter`, `--close_iter`, `--min_area`, `--max_area`, `--min_circ`, `--min_maxI` |
| Score2 | `--score2_mode`, `--contrast_r_inner`, `--contrast_r_outer1`, `--contrast_r_outer2`, `--support_M`, `--support_tol`, `--support_w`, `--ml_model_path` |
| Candidate fallback | `--cand_fallback`, `--cand_target_raw`, `--cand_fallback_passes`, `--cand_fallback_percentiles`, `--cand_fallback_kernel_add`, `--cand_merge_eps` |
| Pooling + ROI | `--max_pool`, `--roi_mode`, `--roi_border_frac`, `--roi_border_px` |
| Pupil ROI (crop + gate) | `--pupil_roi`, `--pupil_roi_size`, `--pupil_roi_pad_mode`, `--pupil_roi_pad_value`, `--pupil_roi_fail_policy`, `--pupil_source`, `--pupil_npz`, `--pupil_dark_thresh`, `--pupil_min_area`, `--pupil_method`, `--pupil_radii`, `--pupil_rmin`, `--pupil_rmax`, `--pupil_fail_open`, `--pupil_force_gate`, `--pupil_fallback_center` |
| Templates | `--template_mode`, `--template_build_mode`, `--template_bank_source`, `--template_bank_path`, `--bank_select_metric`, `--verbose_template` |
| Matchers (common) | `--matcher`, `--min_k`, `--min_inliers`, `--eps`, `--iters`, `--matching`, `--appearance_tiebreak`, `--scale_min`, `--scale_max`, `--disable_scale_gate`, `--seed` |
| Star matcher | `--vote_M`, `--vote_ratio_tol`, `--vote_max_hyp`, `--vote_w_score2` |
| SLA matcher | `--pivot_P`, `--ratio_tol`, `--max_seeds`, `--grow_resid_max`, `--sla_w_seed_score2`, `--sla_w_seed_geom`, `--max_seeds_per_pivot`, `--sla_adaptive_ratio_tol`, `--sla_ratio_tol_min`, `--sla_ratio_tol_refN`, `--sla_scale_min`, `--sla_scale_max`, `--sla_g0_top2` |
| Priors + identity | `--layout_prior`, `--layout_lambda`, `--layout_mode`, `--sla_layout_prior`, `--sla_layout_lambda`, `--sla_layout_mode`, `--sla_semantic_prior`, `--sla_semantic_lambda`, `--sla_semantic_hard`, `--sla_mirror_reject`, `--sla_top2_margin`, `--sla_base_ratio_min`, `--sla_side_margin`, `--post_id_resolve`, `--id_layout_mode`, `--id_lambda`, `--id_gamma`, `--id_eta`, `--id_tau` |
| Evaluation + output | `--labels`, `--match_tol`, `--visualize`, `--viz_metrics`, `--diag_candidate_recall`, `--diag_make_plots`, `--diag_plot_path`, `--save_glints_npz` |

## Parameters that appear unused or redundant
- `--ml_eps_label`
- `--ml_use_patch`
- `--pupil_sigma_frac`

