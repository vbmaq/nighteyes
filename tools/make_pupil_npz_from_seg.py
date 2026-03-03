import argparse
from pathlib import Path

import cv2
import numpy as np


def _fit_ellipse_from_mask(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    if area <= 0:
        return None
    if len(cnt) >= 5:
        (cx, cy), (maj, minr), angle = cv2.fitEllipse(cnt)
        return (float(cx), float(cy)), float(maj), float(minr), float(angle)
    m = cv2.moments(cnt)
    if abs(m["m00"]) < 1e-9:
        return None
    cx = m["m10"] / m["m00"]
    cy = m["m01"] / m["m00"]
    r = np.sqrt(area / np.pi)
    d = 2.0 * float(r)
    return (float(cx), float(cy)), d, d, 0.0


def _choose_pupil_label(arr: np.ndarray, candidate_labels: list[int]) -> int | None:
    areas = {}
    for v in candidate_labels:
        areas[v] = int(np.sum(arr == v))
    nonzero = {k: v for k, v in areas.items() if v > 0}
    if not nonzero:
        return None
    return min(nonzero, key=nonzero.get)


def build_pupil_npz(root: Path, labels_dir: Path, images_dir: Path, out_path: Path, candidate_labels: list[int]) -> None:
    label_files = sorted(labels_dir.glob("*.npy"))
    if not label_files:
        raise FileNotFoundError(f"No label .npy files in {labels_dir}")

    filenames = []
    ellipses = []
    total = len(label_files)
    for i, lp in enumerate(label_files, 1):
        arr = np.load(lp)
        label = _choose_pupil_label(arr, candidate_labels)
        if label is None:
            centroid = (float("nan"), float("nan"))
            ell = {"centroid": centroid, "major_axis_length": None, "minor_axis_length": None}
        else:
            mask = (arr == label).astype(np.uint8) * 255
            fit = _fit_ellipse_from_mask(mask)
            if fit is None:
                centroid = (float("nan"), float("nan"))
                ell = {"centroid": centroid, "major_axis_length": None, "minor_axis_length": None}
            else:
                (cx, cy), maj, minr, angle = fit
                ell = {
                    "centroid": (float(cx), float(cy)),
                    "major_axis_length": float(maj),
                    "minor_axis_length": float(minr),
                    "angle": float(angle),
                    "label": int(label),
                }
        img_name = lp.stem + ".png"
        if not (images_dir / img_name).exists():
            # Fallback: keep original label name if image is missing.
            img_name = lp.name
        filenames.append(img_name)
        ellipses.append(ell)
        if i % 1000 == 0 or i == total:
            print(f"[{root.name}] processed {i}/{total}")

    np.savez(out_path, filenames=np.array(filenames, dtype=object), ellipses=np.array(ellipses, dtype=object))
    print(f"Wrote {out_path} with {len(filenames)} entries")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=str, help="Split root (contains images/ and labels/)")
    ap.add_argument("--labels", type=str, default="labels", help="Labels folder (relative to root)")
    ap.add_argument("--images", type=str, default="images", help="Images folder (relative to root)")
    ap.add_argument("--out", type=str, default="pupil.npz", help="Output NPZ filename (relative to root)")
    ap.add_argument("--labels_set", type=str, default="1,2,3", help="Candidate label IDs (comma-separated)")
    args = ap.parse_args()

    root = Path(args.root)
    labels_dir = (root / args.labels).resolve()
    images_dir = (root / args.images).resolve()
    out_path = (root / args.out).resolve()
    candidate_labels = [int(x.strip()) for x in args.labels_set.split(",") if x.strip()]

    build_pupil_npz(root, labels_dir, images_dir, out_path, candidate_labels)


if __name__ == "__main__":
    main()
