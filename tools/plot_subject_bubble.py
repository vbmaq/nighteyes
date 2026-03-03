#!/usr/bin/env python3
"""
Re-generate the per-subject bubble chart (match accuracy vs mean localization error)
from report_per_subject.csv with controllable font sizes.

This is useful for making figures publication-ready without re-running the pipeline.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def _to_float(x) -> float:
    if x is None:
        return float("nan")
    s = str(x).strip().lower()
    if s in ("", "nan", "none", "null"):
        return float("nan")
    try:
        return float(s)
    except Exception:
        return float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Directory containing report_per_subject.csv")
    ap.add_argument(
        "--out",
        default="",
        help="Output PNG path. Default: <run_dir>/metrics_subject_bubble.png",
    )
    ap.add_argument("--title", default="Per-subject bubble chart (size = images)")
    ap.add_argument("--font", type=float, default=12.0, help="Base font size")
    ap.add_argument("--label_font", type=float, default=10.0, help="Subject label font size")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    csv_path = run_dir / "report_per_subject.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")

    out_path = Path(args.out).expanduser().resolve() if args.out else (run_dir / "metrics_subject_bubble.png")

    rows = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            a = _to_float(row.get("match_accuracy"))
            e = _to_float(row.get("loc_err_mean"))
            n = _to_float(row.get("images"))
            if not (math.isfinite(a) and math.isfinite(e) and math.isfinite(n) and n > 0):
                continue
            rows.append((str(row.get("subject", "unknown")), float(a), float(e), int(round(n))))

    if not rows:
        raise RuntimeError("No finite rows to plot (need match_accuracy, loc_err_mean, images).")

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"matplotlib is required: {exc}") from exc

    plt.rcParams.update(
        {
            "font.size": float(args.font),
            "axes.titlesize": float(args.font) + 2,
            "axes.labelsize": float(args.font) + 1,
            "xtick.labelsize": float(args.font) - 1,
            "ytick.labelsize": float(args.font) - 1,
        }
    )

    labels = [s for (s, _, _, _) in rows]
    acc = [a for (_, a, _, _) in rows]
    err = [e for (_, _, e, _) in rows]
    imgs = [n for (_, _, _, n) in rows]

    max_imgs = max(imgs) if imgs else 1
    sizes = [80 + 320 * (n / max_imgs) for n in imgs]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(acc, err, s=sizes, alpha=0.7, color="#4e79a7", edgecolor="k")
    ax.set_xlabel("Match accuracy")
    ax.set_ylabel("Mean localization error (px)")
    ax.set_title(str(args.title))
    for x, y, lab in zip(acc, err, labels):
        ax.annotate(lab, (x, y), fontsize=float(args.label_font), alpha=0.85, ha="center", va="center")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

