import argparse
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import cv2
import numpy as np


def _fit_pupil_from_mask(mask: np.ndarray) -> Dict[str, object]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return {"centroid": (-1.0, -1.0), "major_axis_length": None, "minor_axis_length": None, "angle": None}
    cx = float(xs.mean())
    cy = float(ys.mean())
    area = float(xs.size)
    major = minor = None
    angle = None
    if xs.size >= 5:
        pts = np.column_stack([xs, ys]).astype(np.int32)
        try:
            (ecx, ecy), (ma, mi), ang = cv2.fitEllipse(pts)
            cx, cy = float(ecx), float(ecy)
            major, minor = float(max(ma, mi)), float(min(ma, mi))
            angle = float(ang)
        except Exception:
            pass
    if major is None or minor is None:
        r = float(np.sqrt(area / np.pi)) if area > 0 else 0.0
        major = minor = 2.0 * r if r > 0 else None
    return {"centroid": (cx, cy), "major_axis_length": major, "minor_axis_length": minor, "angle": angle}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Root of openEDS2020-SparseSegmentation")
    ap.add_argument("--labels_pkl", type=str, default=None, help="Path to test_sampleName_GT.pkl")
    ap.add_argument("--out", type=str, required=True, help="Output pupil.npz path")
    ap.add_argument("--per_participant", action="store_true", help="Also write pupil.npz inside each participant folder")
    args = ap.parse_args()

    root = Path(args.root)
    if args.labels_pkl is None:
        labels_pkl = root / "test_GT" / "test_sampleName_GT.pkl"
    else:
        labels_pkl = Path(args.labels_pkl)
    if not labels_pkl.exists():
        raise FileNotFoundError(f"Labels pkl not found: {labels_pkl}")

    with labels_pkl.open("rb") as f:
        masks = pickle.load(f)
    # Map "S_0/54.npy" -> mask
    mask_map: Dict[str, np.ndarray] = {}
    for k, v in masks.items():
        key = str(k).replace("\\", "/")
        if key.endswith(".npy"):
            key = key[:-4] + ".png"
        mask_map[key] = np.asarray(v, dtype=np.uint8)

    participant_root = root / "participant"
    if not participant_root.exists():
        raise FileNotFoundError(f"participant folder not found: {participant_root}")

    filenames: List[str] = []
    ellipses: List[Dict[str, object]] = []
    missing = 0
    total = 0
    for part_dir in sorted([p for p in participant_root.iterdir() if p.is_dir()]):
        rel_part = part_dir.name
        for img_path in sorted([p for p in part_dir.iterdir() if p.suffix.lower() == ".png"]):
            rel_key = f"{rel_part}/{img_path.name}"
            mask = mask_map.get(rel_key)
            if mask is None:
                missing += 1
                ell = {"centroid": (-1.0, -1.0), "major_axis_length": None, "minor_axis_length": None, "angle": None}
            else:
                # pupil label is 3 in openEDS masks
                ell = _fit_pupil_from_mask(mask == 3)
            filenames.append(rel_key)
            ellipses.append(ell)
            total += 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, filenames=np.array(filenames, dtype=object), ellipses=np.array(ellipses, dtype=object))
    print(f"Wrote {out_path} with {total} entries. Missing masks: {missing}")

    if args.per_participant:
        # write per participant with basename keys to avoid UI mismatch
        for part_dir in sorted([p for p in participant_root.iterdir() if p.is_dir()]):
            rel_part = part_dir.name
            part_files = []
            part_ell = []
            for img_path in sorted([p for p in part_dir.iterdir() if p.suffix.lower() == ".png"]):
                rel_key = f"{rel_part}/{img_path.name}"
                mask = mask_map.get(rel_key)
                if mask is None:
                    ell = {"centroid": (-1.0, -1.0), "major_axis_length": None, "minor_axis_length": None, "angle": None}
                else:
                    ell = _fit_pupil_from_mask(mask == 3)
                part_files.append(img_path.name)
                part_ell.append(ell)
            np.savez(part_dir / "pupil.npz", filenames=np.array(part_files, dtype=object), ellipses=np.array(part_ell, dtype=object))


if __name__ == "__main__":
    main()
