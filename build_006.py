"""Build K2-CV-STEP-006.usd from 005: knife on the robot, tray at the cut station. 004/005 are not modified.

    cd ~/_isaac_sim && ./python.sh .../DoughSDG/cutting/build_006.py [--station-x 0.65] [--force]
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paths import USD_DIR  # noqa: E402
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--src", default=os.path.join(USD_DIR, "K2-CV-STEP-005-SDG.usd"))
parser.add_argument("--dst", default=os.path.join(USD_DIR, "K2-CV-STEP-006-CUT.usd"))
parser.add_argument("--station-x", type=float, default=0.65)
parser.add_argument("--force", action="store_true")
args = parser.parse_args()
if os.path.exists(args.dst) and not args.force:
    sys.exit(f"{args.dst} exists; pass --force")
if "004" in os.path.basename(args.dst) or os.path.abspath(args.dst) == os.path.abspath(args.src):
    sys.exit("refusing to write over 004 or the source scene")

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom  # noqa: E402

from kinematics import Knife, fk_tool0  # noqa: E402

KNIFE_SRC = ("/World/A_900023_INT_00_16_JUL_25_1/tn__A900023INT0016JUL251_lT8CFIMq0/tn__3453A0001_zH/"
             "tn__00RB025COG2_pGu4")
ROBOT = "/World/abb_irb1200_5_90"
TOOL0 = ROBOT + "/link_6/tool0"
TRAY_PRODUCT = "/World/Products/Bagg_01"

Sdf.Layer.FindOrOpen(args.src).Export(args.dst)
stage = Usd.Stage.Open(args.dst)
layer = stage.GetRootLayer()
knife = Knife()


def gf_frame(R, p):
    """Row-vector USD matrix for rotation R (columns = axes) and translation p."""
    return Gf.Matrix4d(*R[:, 0], 0, *R[:, 1], 0, *R[:, 2], 0, *p, 1)


# 1. Knife on tool0: its CAD frame (plate centre, tool z out of the flange) lands on tool0.
src = stage.GetPrimAtPath(KNIFE_SRC)
M_root = UsdGeom.Xformable(src).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
dst_path = TOOL0 + "/Knife"
# The knife is defined inside the referenced CAD assembly, not in this file: reference it from there.
defs = [s for s in src.GetPrimStack() if s.specifier == Sdf.SpecifierDef]
if not defs:
    sys.exit("could not find where the knife is defined")
knife_prim = stage.DefinePrim(dst_path, "Xform")
knife_prim.GetReferences().AddReference(defs[0].layer.identifier, defs[0].path)
print(f"KNIFE referenced from {defs[0].layer.identifier} {defs[0].path}", flush=True)
local = M_root * gf_frame(knife.R_world_tool, knife.mount).GetInverse()
xf = UsdGeom.Xformable(knife_prim)
xf.ClearXformOpOrder()
xf.AddTransformOp().Set(local)
src.SetActive(False)  # the floating knife is gone; the one on the robot replaces it
# TCP marker: blade tip, z along the blade into the product, x = cutting direction.
tcp = UsdGeom.Xform.Define(stage, TOOL0 + "/Knife_TCP")
tcp.AddTransformOp().Set(gf_frame(knife.R_tcp, knife.p_tcp))

# 2. Tray to the cut station (same height on the belt), scanned crops off, generator doughs shown.
tray = stage.GetPrimAtPath(TRAY_PRODUCT)
ops = {op.GetOpType(): op for op in UsdGeom.Xformable(tray).GetOrderedXformOps()}
old = ops[UsdGeom.XformOp.TypeTranslate].Get()
ops[UsdGeom.XformOp.TypeTranslate].Set(Gf.Vec3d(args.station_x, old[1], old[2]))
if stage.GetPrimAtPath(TRAY_PRODUCT + "/ScanPieces"):
    stage.GetPrimAtPath(TRAY_PRODUCT + "/ScanPieces").SetActive(False)
if stage.GetPrimAtPath(TRAY_PRODUCT + "/SDG"):
    UsdGeom.Imageable(stage.GetPrimAtPath(TRAY_PRODUCT + "/SDG")).MakeVisible()
stage.GetRootLayer().Save()

# 3. Checks.
cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
t0 = UsdGeom.Xformable(stage.GetPrimAtPath(TOOL0)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
tcp_w = UsdGeom.Xformable(tcp.GetPrim()).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
R_np, p_np = knife.tcp(np.zeros(6))
print("CHECK tool0 world", np.round(list(t0.ExtractTranslation()), 4), "numpy", np.round(fk_tool0(np.zeros(6))[1], 4), flush=True)
print("CHECK TCP world (USD)", np.round(list(tcp_w), 4), "numpy", np.round(p_np, 4), flush=True)
kb = cache.ComputeWorldBound(knife_prim).ComputeAlignedRange()
print("CHECK knife bbox on robot", np.round(list(kb.GetMin()), 3), np.round(list(kb.GetMax()), 3), flush=True)
tb = cache.ComputeWorldBound(stage.GetPrimAtPath(TRAY_PRODUCT + "/Cube")).ComputeAlignedRange()
print("CHECK tray at station", np.round(list(tb.GetMin()), 3), np.round(list(tb.GetMax()), 3), flush=True)
# belt top under the station: highest non-product mesh below 1.0 m within 5 cm of (station_x, 0)
best = -1.0
for p in stage.Traverse():
    if p.IsA(UsdGeom.Mesh) and not str(p.GetPath()).startswith(("/World/Products", ROBOT)):
        r = cache.ComputeWorldBound(p).ComputeAlignedRange()
        if r.IsEmpty():
            continue
        lo, hi = r.GetMin(), r.GetMax()
        if lo[0] <= args.station_x + 0.05 and hi[0] >= args.station_x - 0.05 and lo[1] <= 0.05 and hi[1] >= -0.05 and hi[2] < 1.0:
            best = max(best, hi[2])
print("CHECK belt top under station z=%.3f (tray bottom z=%.3f)" % (best, tb.GetMin()[2]), flush=True)
print(f"SAVED {args.dst}", flush=True)
app.close()
