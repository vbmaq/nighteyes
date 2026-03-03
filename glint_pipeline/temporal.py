from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import math
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment as _hungarian
except Exception:  # pragma: no cover - fallback for environments without scipy
    _hungarian = None


def _build_F(dt: float) -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _build_H() -> np.ndarray:
    return np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=float)


def _build_Q(dt: float, q_pos: float, q_vel: float) -> np.ndarray:
    q = np.zeros((4, 4), dtype=float)
    q[0, 0] = q_pos
    q[1, 1] = q_pos
    q[2, 2] = q_vel
    q[3, 3] = q_vel
    return q


def _build_R(r_meas: float) -> np.ndarray:
    return np.array([[r_meas, 0.0], [0.0, r_meas]], dtype=float)


@dataclass
class TrackKF:
    id: int
    x: np.ndarray
    P: np.ndarray
    age: int = 0
    hits: int = 0
    miss_count: int = 0
    last_update_frame: int = -1
    conf: float = 0.0

    def predict(self, dt: float = 1.0, q_pos: float = 1.0, q_vel: float = 0.5) -> None:
        F = _build_F(dt)
        Q = _build_Q(dt, q_pos, q_vel)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.age += 1
        self._update_conf()

    def update(self, z_xy: np.ndarray, r_meas: float = 4.0) -> None:
        H = _build_H()
        R = _build_R(r_meas)
        z = z_xy.reshape(2, 1)
        x = self.x.reshape(4, 1)
        y = z - H @ x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        x = x + K @ y
        I = np.eye(4, dtype=float)
        self.P = (I - K @ H) @ self.P
        self.x = x.ravel()
        self.hits += 1
        self.miss_count = 0
        self._update_conf()

    def mark_missed(self) -> None:
        self.miss_count += 1
        self._update_conf()

    def pos(self) -> np.ndarray:
        return self.x[:2].copy()

    def vel(self) -> np.ndarray:
        return self.x[2:].copy()

    def is_active(self, max_missed: int) -> bool:
        return self.hits > 0 and self.miss_count <= max_missed

    def _update_conf(self) -> None:
        if self.age <= 0:
            self.conf = 0.0
        else:
            self.conf = float(self.hits) / float(self.age) * math.exp(-float(self.miss_count))


def _greedy_assignment(cost: np.ndarray) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    if cost.size == 0:
        return pairs
    c = cost.copy()
    n_rows, n_cols = c.shape
    used_r = set()
    used_c = set()
    while True:
        idx = np.unravel_index(np.argmin(c), c.shape)
        r, col = int(idx[0]), int(idx[1])
        if np.isinf(c[r, col]):
            break
        if r in used_r or col in used_c:
            c[r, col] = float("inf")
            continue
        pairs.append((r, col))
        used_r.add(r)
        used_c.add(col)
        c[r, :] = float("inf")
        c[:, col] = float("inf")
        if len(used_r) >= n_rows or len(used_c) >= n_cols:
            break
    return pairs


class MultiGlintTracker:
    def __init__(
        self,
        n_tracks: int = 4,
        gate_px: float = 25.0,
        max_missed: int = 5,
        q_pos: float = 1.0,
        q_vel: float = 0.5,
        r_meas: float = 4.0,
        init_strategy: str = "top_score2",
        allow_reseed: bool = True,
    ) -> None:
        self.n_tracks = int(n_tracks)
        self.gate_px = float(gate_px)
        self.max_missed = int(max_missed)
        self.q_pos = float(q_pos)
        self.q_vel = float(q_vel)
        self.r_meas = float(r_meas)
        self.init_strategy = init_strategy
        self.allow_reseed = allow_reseed
        self.tracks: List[TrackKF] = []
        self.frame_idx = -1
        self._init_tracks()

    def _init_tracks(self) -> None:
        self.tracks = []
        for i in range(self.n_tracks):
            x = np.zeros(4, dtype=float)
            P = np.eye(4, dtype=float) * 100.0
            self.tracks.append(TrackKF(id=i, x=x, P=P))

    def reset(self) -> None:
        self.frame_idx = -1
        self._init_tracks()

    def step(
        self,
        dets_xy: np.ndarray,
        dets_score2: Optional[np.ndarray],
        frame_idx: int,
    ) -> Tuple[np.ndarray, List[dict]]:
        self.frame_idx = int(frame_idx)
        dets_xy = np.asarray(dets_xy, dtype=float) if dets_xy is not None else np.empty((0, 2), dtype=float)
        dets_score2 = np.asarray(dets_score2, dtype=float) if dets_score2 is not None else np.zeros((len(dets_xy),), dtype=float)

        # Initialize if no hits yet
        if all(t.hits == 0 for t in self.tracks) and dets_xy.size > 0:
            order = np.argsort(dets_score2)[::-1]
            for i, t in enumerate(self.tracks):
                if i >= len(order):
                    break
                j = int(order[i])
                t.x[:2] = dets_xy[j]
                t.x[2:] = 0.0
                t.P = np.eye(4, dtype=float) * 25.0
                t.age = 1
                t.hits = 1
                t.miss_count = 0
                t._update_conf()

        # Predict
        for t in self.tracks:
            t.predict(dt=1.0, q_pos=self.q_pos, q_vel=self.q_vel)

        n = self.n_tracks
        m = len(dets_xy)
        cost = np.full((n, m), float("inf"), dtype=float)
        for i, t in enumerate(self.tracks):
            pred = t.pos()
            if not np.isfinite(pred).all():
                continue
            for j in range(m):
                d = float(np.hypot(pred[0] - dets_xy[j, 0], pred[1] - dets_xy[j, 1]))
                if d <= self.gate_px:
                    cost[i, j] = d

        if _hungarian is not None:
            rows, cols = _hungarian(cost)
            pairs = [(int(r), int(c)) for r, c in zip(rows, cols)]
        else:
            pairs = _greedy_assignment(cost)

        assigned_tracks = set()
        assigned_dets = set()
        for r, c in pairs:
            if r < 0 or c < 0:
                continue
            if cost[r, c] == float("inf"):
                continue
            self.tracks[r].update(dets_xy[c], r_meas=self.r_meas)
            self.tracks[r].last_update_frame = self.frame_idx
            assigned_tracks.add(r)
            assigned_dets.add(c)

        # Mark missed
        for i, t in enumerate(self.tracks):
            if i not in assigned_tracks:
                t.mark_missed()

        # Reseed inactive tracks if allowed
        if self.allow_reseed and m > 0:
            unused = [j for j in range(m) if j not in assigned_dets]
            if unused:
                order = sorted(unused, key=lambda j: dets_score2[j], reverse=True)
                for i, t in enumerate(self.tracks):
                    if t.is_active(self.max_missed):
                        continue
                    if not order:
                        break
                    j = order.pop(0)
                    t.x[:2] = dets_xy[j]
                    t.x[2:] = 0.0
                    t.P = np.eye(4, dtype=float) * 25.0
                    t.age = 1
                    t.hits = 1
                    t.miss_count = 0
                    t.last_update_frame = self.frame_idx
                    t._update_conf()

        tracked_xy = np.full((self.n_tracks, 2), np.nan, dtype=float)
        meta: List[dict] = []
        for i, t in enumerate(self.tracks):
            active = t.is_active(self.max_missed)
            if active:
                tracked_xy[i] = t.pos()
            meta.append(
                {
                    "id": t.id,
                    "age": t.age,
                    "hits": t.hits,
                    "miss_count": t.miss_count,
                    "conf": t.conf,
                    "active": active,
                }
            )
        return tracked_xy, meta

    def step_labeled(self, dets_xy: np.ndarray, frame_idx: int) -> Tuple[np.ndarray, List[dict]]:
        """
        Update tracks with labeled detections by track index.
        dets_xy: (n_tracks, 2) with NaNs for missing.
        """
        self.frame_idx = int(frame_idx)
        dets_xy = np.asarray(dets_xy, dtype=float)
        if dets_xy.shape[0] != self.n_tracks:
            raise ValueError("dets_xy must have shape (n_tracks, 2)")

        # Bootstrap labeled tracks directly from the first valid measurements.
        if all(t.hits == 0 for t in self.tracks):
            for i, t in enumerate(self.tracks):
                z = dets_xy[i]
                if not np.isfinite(z).all():
                    continue
                t.x[:2] = z
                t.x[2:] = 0.0
                t.P = np.eye(4, dtype=float) * 25.0
                t.age = 1
                t.hits = 1
                t.miss_count = 0
                t.last_update_frame = self.frame_idx
                t._update_conf()
            tracked_xy = np.full((self.n_tracks, 2), np.nan, dtype=float)
            meta: List[dict] = []
            for i, t in enumerate(self.tracks):
                active = t.is_active(self.max_missed)
                if active:
                    tracked_xy[i] = t.pos()
                meta.append(
                    {
                        "id": t.id,
                        "age": t.age,
                        "hits": t.hits,
                        "miss_count": t.miss_count,
                        "conf": t.conf,
                        "active": active,
                    }
                )
            return tracked_xy, meta

        # Predict
        for t in self.tracks:
            t.predict(dt=1.0, q_pos=self.q_pos, q_vel=self.q_vel)

        for i, t in enumerate(self.tracks):
            z = dets_xy[i]
            if not np.isfinite(z).all():
                t.mark_missed()
                continue
            # Allow direct reseed for dead tracks in labeled mode.
            if self.allow_reseed and not t.is_active(self.max_missed):
                t.x[:2] = z
                t.x[2:] = 0.0
                t.P = np.eye(4, dtype=float) * 25.0
                t.age = max(1, t.age)
                t.hits = max(1, t.hits)
                t.miss_count = 0
                t.last_update_frame = self.frame_idx
                t._update_conf()
                continue
            d = float(np.hypot(t.pos()[0] - z[0], t.pos()[1] - z[1]))
            if d <= self.gate_px:
                t.update(z, r_meas=self.r_meas)
                t.last_update_frame = self.frame_idx
            else:
                t.mark_missed()

        tracked_xy = np.full((self.n_tracks, 2), np.nan, dtype=float)
        meta: List[dict] = []
        for i, t in enumerate(self.tracks):
            active = t.is_active(self.max_missed)
            if active:
                tracked_xy[i] = t.pos()
            meta.append(
                {
                    "id": t.id,
                    "age": t.age,
                    "hits": t.hits,
                    "miss_count": t.miss_count,
                    "conf": t.conf,
                    "active": active,
                }
            )
        return tracked_xy, meta
