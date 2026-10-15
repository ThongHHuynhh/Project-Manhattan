"""Cut scheduler for a stream of trays on a moving belt (no fixed tray or piece counts).

Inputs are tray messages, one per tray, sent when the tray passes the photoeye:

    {"tray_id": "T0007", "latch_ticks": 123456789, "latch_x": -0.20,
     "pieces": [{"id": "p0", "kind": "long",
                 "contour": [[x, y, z], ...],      # top surface along the cut line, tray coordinates (m)
                 "direction": [dx, dy, 0]}]}        # cut direction

The scheduler turns each contour into the knife path (approach, cut 3 mm under the contour, retract;
see cut_paths.cut_path). Messages with a ready-made "cut" polyline plus "approach"/"retract" points are
also accepted.

`latch_x` is where the tray origin is on the belt (robot/world X) at `latch_ticks`. From then on its
position comes only from the encoder: x(t) = latch_x + (ticks(t) - latch_ticks) / counts_per_m.
On the line, AI_CORE (RealSense detections) or the PLC can send the same message.

Policy: of the cuts that can be finished inside the reach window, take the one whose tray leaves the
window first (earliest deadline first). Cuts that can no longer be finished are reported as missed.
"""
from dataclasses import dataclass, field

import numpy as np

from cut_paths import cut_path
from encoder import tick_delta


@dataclass
class CutTask:
    tray_id: str
    piece_id: str
    kind: str
    poly_local: np.ndarray  # approach -> cut points -> retract, tray coordinates
    direction: np.ndarray
    latch_ticks: int
    latch_x: float
    status: str = "pending"  # pending | active | done | missed
    result: dict = field(default_factory=dict)

    @property
    def length(self):
        return float(np.linalg.norm(np.diff(self.poly_local, axis=0), axis=1).sum())


class CutScheduler:
    def __init__(self, counts_per_m, reach_x=(0.05, 1.10), cut_speed=0.05, transit_s=1.5, bits=32, depth=0.003):
        # reach_x: belt x range of the training cuts; a per-cut `check` (e.g. IK at the start and drifted
        # end) makes the final call.
        self.counts_per_m, self.reach_x, self.cut_speed = float(counts_per_m), reach_x, cut_speed
        self.transit_s, self.bits, self.depth = transit_s, bits, depth
        self.tasks: list[CutTask] = []

    def add_tray(self, msg):
        for p in msg["pieces"]:
            if "contour" in p:
                poly, _, _ = cut_path(p["contour"], p["direction"], -self.depth)
            else:  # ready-made cut line
                poly = np.array([p["approach"]] + p["cut"] + [p["retract"]], float)
            self.tasks.append(CutTask(msg["tray_id"], p["id"], p.get("kind", ""), poly, np.array(p["direction"], float),
                                      int(msg["latch_ticks"]), float(msg["latch_x"])))

    def tray_x(self, task, ticks):
        return task.latch_x + tick_delta(ticks, task.latch_ticks, self.bits) / self.counts_per_m

    def world_poly(self, task, ticks):
        return task.poly_local + [self.tray_x(task, ticks), 0.0, 0.0]

    def next_task(self, ticks, belt_speed, check=None):
        """Pick the next cut, or None (wait). Marks cuts that can no longer be finished as missed.

        check(task, poly_now, drift) -> bool can veto a candidate (e.g. an IK reachability test at the
        start and at the end after the belt has moved `drift` metres).
        """
        v = max(belt_speed, 0.0)
        candidates = []
        for t in self.tasks:
            if t.status != "pending":
                continue
            poly = self.world_poly(t, ticks)
            drift = v * (self.transit_s + t.length / self.cut_speed)
            x_end = poly[:, 0].max() + drift
            if x_end > self.reach_x[1]:
                t.status, t.result = "missed", {"reason": "left the reach window"}
                continue
            if poly[:, 0].min() + v * self.transit_s >= self.reach_x[0]:
                slack = (self.reach_x[1] - x_end) / v if v > 1e-9 else np.inf
                candidates.append((slack, t))
        for slack, t in sorted(candidates, key=lambda c: c[0]):
            drift = v * (self.transit_s + t.length / self.cut_speed)
            if check is None or check(t, self.world_poly(t, ticks), drift):
                t.status = "active"
                return t
        return None

    def summary(self):
        out = {}
        for t in self.tasks:
            out.setdefault(t.tray_id, {"done": 0, "missed": 0, "pending": 0, "active": 0, "in_tolerance": 0})
            out[t.tray_id][t.status] += 1
            out[t.tray_id]["in_tolerance"] += int(bool(t.result.get("pass")))
        return out
