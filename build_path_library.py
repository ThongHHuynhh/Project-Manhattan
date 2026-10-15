"""Cut-path library for RL training: random trays of procedural dough (1-5 pieces, long and round,
any rotation), anywhere in the reach window; one cut per piece with an IK start pose.
Vertical plunge, 3 mm deep, cutting with the belt.

    ~/_isaac_sim/python.sh ~/TH/Project-Manhattan/build_path_library.py --trays 900 --out .../path_library.npz
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paths import DOUGHSDG_DIR as ROOT  # noqa: E402  (the dough generator)
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--trays", type=int, default=900)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--window", type=float, nargs=2, default=[0.35, 0.80], help="tray centre x range on the belt (m)")
parser.add_argument("--out", default=os.path.join(HERE, "path_library.npz"))
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})  # only for the dough plugin's imports; nothing is rendered
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import importlib.util  # noqa: E402

import numpy as np  # noqa: E402

from cut_paths import piece_cut  # noqa: E402
from kinematics import Knife, ik, target_rotation  # noqa: E402

spec = importlib.util.spec_from_file_location("dough_plugin", os.path.join(ROOT, "plugins", "dough_procedural.py"))
dough_plugin = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dough_plugin)
P = {}
for cls in reversed(dough_plugin.ProceduralDough.__mro__):
    P.update(cls.__dict__.get("defaults", {}))
P["mesh_resolution"] = [64, 48]  # cut lines only need the top surface; coarser is plenty

TRAY_Z, TRAY_TOP, HALF = 0.9114, 0.025, (0.33, 0.32)  # product origin z, tray top (local), usable half size
knife = Knife()
rng = np.random.default_rng(args.seed)
N_MAX = 136
paths, counts, q0s, meta = [], [], [], []
seed_q = [np.zeros(6)] + [rng.uniform(-1.2, 1.2, 6) for _ in range(6)]
t_start, tried = time.time(), 0
for t in range(args.trays):
    x0 = rng.uniform(*args.window)
    n = int(rng.integers(1, 6))
    yaws_free = rng.random() < 0.4
    for _ in range(n):
        tried += 1
        round_piece = rng.random() < 0.3
        width = rng.uniform(0.06, 0.13)
        length = width * rng.uniform(0.9, 1.25) if round_piece else rng.uniform(0.18, 0.54)
        height = rng.uniform(0.04, 0.10)
        m = dough_plugin.dough_mesh(rng, length, width, height, P, tray_top=TRAY_TOP)
        yaw = rng.uniform(-np.pi / 2, np.pi / 2) if yaws_free else rng.normal(0, 0.05)
        reach = max(length, width) / 2
        cx = rng.uniform(-HALF[0] + reach, HALF[0] - reach) if HALF[0] > reach else 0.0
        cy = rng.uniform(-HALF[1] + reach, HALF[1] - reach) if HALF[1] > reach else 0.0
        c, s = np.cos(yaw), np.sin(yaw)
        pts = m["points"] @ np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]]) + [x0 + cx, cy, TRAY_Z]
        cut = piece_cut(pts, faces=(m["counts"], m["indices"]))
        poly = np.array(cut["path"])
        if len(poly) > N_MAX:
            continue
        R = target_rotation(cut["direction"])
        q, ok = ik(knife, R, poly[0], seed_q[0])
        for sq in seed_q[1:]:
            if ok:
                break
            q, ok = ik(knife, R, poly[0], sq)
        if not ok:
            continue
        q_end, ok_end = ik(knife, R, poly[-1], q)  # must also reach the far end
        if not ok_end:
            continue
        pad = np.repeat(poly[-1:], N_MAX - len(poly), axis=0)
        paths.append(np.vstack([poly, pad]))
        counts.append(len(poly))
        q0s.append(q)
        meta.append([cut["direction"][0], cut["direction"][1], 1.0 if cut["kind"] == "long" else 0.0, x0])
    if t % 50 == 0:
        print(f"[lib] tray {t}/{args.trays}: {len(paths)} cuts kept of {tried} ({time.time() - t_start:.0f}s)", flush=True)
np.savez_compressed(args.out, paths=np.array(paths, np.float32), counts=np.array(counts, np.int32),
                    q0=np.array(q0s, np.float32), meta=np.array(meta, np.float32), n_max=N_MAX)
print(f"[lib] saved {len(paths)} cuts ({tried - len(paths)} unreachable or too long dropped) -> {args.out}", flush=True)
app.close()
