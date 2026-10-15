"""Find where along the belt the tray must stop so the knife can cut anywhere on it."""
import json
import sys

import numpy as np

sys.path.insert(0, sys.argv[1])
from kinematics import Knife, fk_tool0, ik, target_rotation  # noqa: E402

knife = Knife()
R0, p0 = fk_tool0(np.zeros(6))
print("CHECK tool0 at zero joints", np.round(p0, 3), "(scene: 1.114, 0, 1.428)")
rng = np.random.default_rng(0)
seeds = [np.zeros(6)] + [rng.uniform([-1.5, -0.5, -1.5, -2, -1.5, -3], [1.5, 1.5, 0.5, 2, 1.5, 3]) for _ in range(10)]
z_cut, half = 0.965, (0.32, 0.31)  # bun top height; tray 0.70 x 0.68 m minus edge
dirs = {"belt": [(1, 0, 0), (-1, 0, 0)], "across": [(0, 1, 0), (0, -1, 0)]}
results = []
for xs in np.arange(0.3, 1.55, 0.1):
    ok_count, total = 0, 0
    for gx in np.linspace(-half[0], half[0], 7):
        for gy in np.linspace(-half[1], half[1], 7):
            p = np.array([xs + gx, gy, z_cut])
            both = True
            for name, options in dirs.items():
                hit = False
                for d in options:
                    for s in seeds:
                        q, ok = ik(knife, target_rotation(d), p, s, iters=120)
                        if ok:
                            hit = True
                            break
                    if hit:
                        break
                both &= hit
            ok_count += both
            total += 1
    results.append((round(float(xs), 2), ok_count / total))
    print("STATION x=%.2f reachable %.0f%% of tray points (both cut directions)" % (xs, 100 * ok_count / total), flush=True)
best = max(results, key=lambda r: r[1])
json.dump({"stations": results, "best_x": best[0], "best_fraction": best[1], "z_cut": z_cut}, open("/tmp/reach.json", "w"))
print("BEST", best)
