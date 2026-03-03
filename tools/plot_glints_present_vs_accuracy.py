#!/usr/bin/env python3
"""
Scatter plot for a single run:
- per-image: number of GT glints present vs match accuracy
- per-subject: mean GT glints present vs match accuracy

Input: a run directory containing report_per_image.csv (from glint_pipeline_eval_gen.py).
Output: a PNG saved to the run directory (or --out path).
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import List, Tuple


def _to_float(x: str) -> float:
    if x is None:
        return float("nan")
    s = str(x).strip().lower()
    if s in ("", "nan", "none", "null"):
        return float("nan")
    try:
        return float(s)
    except Exception:
        return float("nan")


def load_xy(report_csv: Path, x_col: str, y_col: str) -> Tuple[List[float], List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    with report_csv.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            raise ValueError("Empty CSV (missing header).")
        missing = [c for c in (x_col, y_col) if c not in r.fieldnames]
        if missing:
            raise ValueError(f"Missing columns in {report_csv.name}: {missing}. Found: {r.fieldnames}")
        for row in r:
            x = _to_float(row.get(x_col, ""))
            y = _to_float(row.get(y_col, ""))
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            xs.append(x)
            ys.append(y)
    return xs, ys


def load_subject_xy_from_subject_report(
    report_csv: Path,
    x_col_total: str,
    y_col: str,
    n_col: str = "images",
) -> Tuple[List[float], List[float], List[str]]:
    """
    Use report_per_subject.csv to compute x as mean(x_total)/images and y as y_col.
    """
    xs: List[float] = []
    ys: List[float] = []
    labels: List[str] = []
    with report_csv.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            raise ValueError("Empty CSV (missing header).")
        need = ["subject", x_col_total, n_col, y_col]
        missing = [c for c in need if c not in r.fieldnames]
        if missing:
            raise ValueError(f"Missing columns in {report_csv.name}: {missing}. Found: {r.fieldnames}")
        for row in r:
            subj = str(row.get("subject", "")).strip() or "unknown"
            n = _to_float(row.get(n_col, ""))
            x_total = _to_float(row.get(x_col_total, ""))
            y = _to_float(row.get(y_col, ""))
            if not (math.isfinite(n) and n > 0 and math.isfinite(x_total) and math.isfinite(y)):
                continue
            xs.append(float(x_total) / float(n))
            ys.append(float(y))
            labels.append(subj)
    return xs, ys, labels


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Run directory containing report_per_image.csv",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="",
        help="Output image path (.png). Default: <run_dir>/glints_present_vs_match_accuracy.png",
    )
    ap.add_argument(
        "--per_subject",
        action="store_true",
        default=False,
        help="Aggregate per subject using report_per_subject.csv (x=mean present/idf_present; y from y_col).",
    )
    ap.add_argument(
        "--x_col",
        type=str,
        default="present",
        help="X column (default: present)",
    )
    ap.add_argument(
        "--y_col",
        type=str,
        default="match_accuracy",
        help="Y column (default: match_accuracy)",
    )
    ap.add_argument(
        "--min_x",
        type=float,
        default=1.0,
        help="Drop points with x < min_x (default: 1, avoids present=0).",
    )
    ap.add_argument(
        "--xlabel",
        type=str,
        default="",
        help="Optional x-axis label override.",
    )
    ap.add_argument(
        "--ylabel",
        type=str,
        default="",
        help="Optional y-axis label override.",
    )
    ap.add_argument(
        "--title",
        type=str,
        default="",
        help="Optional title override.",
    )
    ap.add_argument(
        "--font",
        type=float,
        default=12.0,
        help="Base font size (default: 12).",
    )
    ap.add_argument(
        "--title_font",
        type=float,
        default=14.0,
        help="Title font size (default: 14).",
    )
    ap.add_argument(
        "--label_font",
        type=float,
        default=13.0,
        help="Axis label font size (default: 13).",
    )
    ap.add_argument(
        "--tick_font",
        type=float,
        default=11.0,
        help="Tick label font size (default: 11).",
    )
    ap.add_argument(
        "--legend_font",
        type=float,
        default=11.0,
        help="Legend font size (default: 11).",
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    labels: List[str] = []
    if args.per_subject:
        report_subject = run_dir / "report_per_subject.csv"
        if not report_subject.exists():
            raise FileNotFoundError(f"Missing subject report CSV: {report_subject}")

        # report_per_subject.csv uses *_total columns; mean is total/images.
        if args.x_col == "idf_present":
            x_total_col = "idf_present_total"
        elif args.x_col == "present":
            x_total_col = "present_total"
        else:
            raise ValueError("For --per_subject, x_col must be 'present' or 'idf_present'.")
        xs, ys, labels = load_subject_xy_from_subject_report(report_subject, x_total_col, args.y_col)
    else:
        report_csv = run_dir / "report_per_image.csv"
        if not report_csv.exists():
            raise FileNotFoundError(f"Missing report CSV: {report_csv}")
        xs, ys = load_xy(report_csv, args.x_col, args.y_col)
        if args.min_x is not None:
            keep = [(x, y) for x, y in zip(xs, ys) if x >= float(args.min_x)]
            xs = [x for x, _ in keep]
            ys = [y for _, y in keep]

    if not xs:
        raise RuntimeError("No finite points to plot (after filtering).")

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"matplotlib is required: {exc}") from exc

    default_name = "glints_present_vs_match_accuracy.png"
    if args.per_subject:
        default_name = f"{args.x_col}_mean_vs_{args.y_col}_by_subject.png"
    out = Path(args.out).expanduser().resolve() if args.out else (run_dir / default_name)

    # Use the same sizing as the default bubble chart in glint_pipeline_eval_gen.py so figures
    # are consistent in papers, while allowing font overrides.
    plt.rcParams.update(
        {
            "font.size": float(args.font),
            "axes.titlesize": float(args.title_font),
            "axes.labelsize": float(args.label_font),
            "xtick.labelsize": float(args.tick_font),
            "ytick.labelsize": float(args.tick_font),
            "legend.fontsize": float(args.legend_font),
        }
    )
    fig = plt.figure(figsize=(7, 5))
    out_dpi = 200
    plt.scatter(xs, ys, s=28 if args.per_subject else 18, alpha=0.65 if args.per_subject else 0.55, edgecolors="none")
    plt.xlabel(args.xlabel if args.xlabel else args.x_col)
    plt.ylabel(args.ylabel if args.ylabel else args.y_col)
    title = f"{run_dir.name}: {args.x_col} vs {args.y_col}"
    if args.per_subject:
        title = f"{run_dir.name}: {args.x_col} mean vs {args.y_col} (per subject)"
    plt.title(args.title if args.title else title)
    plt.grid(True, alpha=0.25)
    plt.ylim(-0.05, 1.05)

    if not args.per_subject:
        # Show per-x mean as a small overlay to make trends visible.
        by_x = {}
        for x, y in zip(xs, ys):
            by_x.setdefault(int(round(x)), []).append(float(y))
        x_means = sorted(by_x.keys())
        y_means = [sum(by_x[k]) / max(1, len(by_x[k])) for k in x_means]
        plt.plot(x_means, y_means, "-o", linewidth=1.2, markersize=4, alpha=0.9, label="mean")
        plt.legend(loc="lower right", frameon=True)
    else:
        # Light labeling for a small number of subjects.
        if 0 < len(labels) <= 40:
            for x, y, s in zip(xs, ys, labels):
                plt.text(x, y, s, fontsize=7, alpha=0.8)

    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=out_dpi)
    plt.close(fig)

    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
