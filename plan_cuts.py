"""Cut plan for the doughs on the tray in K2-CV-STEP-006: cut lines, order, IK check; drawn into 006.

    cd ~/_isaac_sim && ./python.sh .../DoughSDG/cutting/plan_cuts.py [--scene ...006-CUT.usd] [--seed N]
With --seed the procedural dough block regenerates the pieces first (same generator as the dataset).
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paths import DOUGHSDG_DIR as ROOT, USD_DIR  # noqa: E402  (ROOT = the dough generator)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--scene", default=os.path.join(USD_DIR, "K2-CV-STEP-006-CUT.usd"))
parser.add_argument("--config", default=os.path.join(ROOT, "configs", "default.json"))
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--out", default=os.path.join(HERE, "cut_plan.json"))
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402
from pxr import Gf, Usd, UsdGeom, Vt  # noqa: E402

from cut_paths import order_pieces, piece_cut  # noqa: E402
from kinematics import Knife, ik, target_rotation  # noqa: E402

stage = Usd.Stage.Open(args.scene)
PRODUCT = "/World/Products/Bagg_01"
if args.seed is not None:
    from sdg_core.pipeline import Pipeline, load_config
    cfg = load_config(args.config, ["blocks.3.params.probability=0"])  # doughs only, no defects
    pipe = Pipeline(cfg, stage, args.scene)
    pipe.randomize(stage, np.random.default_rng(args.seed))
doughs = [p for p in Usd.PrimRange(stage.GetPrimAtPath(PRODUCT + "/SDG/dough")) if p.IsA(UsdGeom.Mesh)
          and UsdGeom.Imageable(p).ComputeVisibility() != UsdGeom.Tokens.invisible]
knife = Knife()
cuts = []
for prim in doughs:
    P = np.array(prim.GetAttribute("points").Get(), float)
    M = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
    W = (np.c_[P, np.ones(len(P))] @ M)[:, :3]
    cut = piece_cut(W, faces=(np.array(prim.GetAttribute("faceVertexCounts").Get()), np.array(prim.GetAttribute("faceVertexIndices").Get())))
    cut["piece"] = str(prim.GetPath())
    cuts.append(cut)
order = order_pieces(cuts, knife.tcp(np.zeros(6))[1][:2])
seeds = [np.zeros(6)] + [np.random.default_rng(i).uniform(-1.2, 1.2, 6) for i in range(8)]
q = np.zeros(6)
for rank, k in enumerate(order):
    c = cuts[k]
    c["order"] = rank
    R = target_rotation(c["direction"])
    ok_all, worst = True, 0.0
    for pt in [c["approach"]] + c["points"] + [c["retract"]]:
        sol, ok = ik(knife, R, np.asarray(pt), q)
        if not ok:
            for s in seeds:
                sol, ok = ik(knife, R, np.asarray(pt), s)
                if ok:
                    break
        ok_all &= ok
        if ok:
            q = sol
    c["ik_reachable"] = bool(ok_all)
    print("CUT %d %-40s %-5s %4.1f cm, %2d points, surface %.1f-%.1f mm above tray, IK %s" % (
        rank, c["piece"].split("/")[-1], c["kind"], c["length_m"] * 100, len(c["points"]),
        (min(c["surface_z"]) - 0.936) * 1000, (max(c["surface_z"]) - 0.936) * 1000, "ok" if ok_all else "FAILED"), flush=True)
json.dump({"scene": args.scene, "seed": args.seed, "depth_m": 0.003, "cuts": [cuts[k] for k in order]}, open(args.out, "w"), indent=1)

# Draw: one curve per cut (approach -> cut -> retract) under /World/CutPlan, in yellow.
if stage.GetPrimAtPath("/World/CutPlan"):
    stage.RemovePrim("/World/CutPlan")
UsdGeom.Scope.Define(stage, "/World/CutPlan")
for rank, k in enumerate(order):
    c = cuts[k]
    pts = [c["approach"]] + c["points"] + [c["retract"]]
    curve = UsdGeom.BasisCurves.Define(stage, f"/World/CutPlan/Cut_{rank:02d}")
    curve.CreateTypeAttr("linear")
    curve.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts]))
    curve.CreateCurveVertexCountsAttr([len(pts)])
    curve.CreateWidthsAttr([0.003] * len(pts))
    curve.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.85, 0.0)])
stage.GetRootLayer().Save()
print(f"PLAN {len(cuts)} cuts -> {args.out}; drawn in {args.scene} at /World/CutPlan", flush=True)
app.close()
