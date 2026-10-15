"""Run the cutting line in K2-CV-STEP-006: trays back to back on a moving belt, encoder, photoeye,
scheduler and the trained policy. Every cut is scored against the true geometry.

    cd ~/IsaacLab
    ./isaaclab.sh -p ~/TH/Project-Manhattan/tray_runner.py --headless --policy <run>/exported/policy.pt \
        --trays 4 --belt_speed 0.02 [--video]

Only the session layer is edited; the 006 file is not changed.
"""
import argparse
import importlib.util
import json
import os
import sys
import time

from isaaclab.app import AppLauncher

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paths import DOUGHSDG_DIR, USD_DIR  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--scene", default=os.path.join(USD_DIR, "K2-CV-STEP-006-CUT.usd"))
parser.add_argument("--policy", required=True)
parser.add_argument("--trays", type=int, default=4)
parser.add_argument("--min_pieces", type=int, default=1)
parser.add_argument("--max_pieces", type=int, default=5)
parser.add_argument("--belt_speed", type=float, default=0.02)
parser.add_argument("--counts_per_m", type=float, default=10000.0)
parser.add_argument("--photoeye_x", type=float, default=-0.10, help="belt x where trays are latched and reported")
parser.add_argument("--gap", type=float, default=0.05, help="m between trays")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_time", type=float, default=900.0, help="simulated seconds")
parser.add_argument("--out", default=os.path.join(HERE, "runs", time.strftime("%Y%m%d-%H%M%S")))
parser.add_argument("--video", action="store_true", help="save frames of an overview camera (15 fps)")
parser.add_argument("--hold", action="store_true", help="with a window: keep it open after the run until you close it")
parser.add_argument("--no_paths", action="store_true", help="do not draw the planned cut lines (yellow) and knife trail (red)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.video:
    args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
import torch  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402
from isaaclab.utils import math as M  # noqa: E402

from cut_env import CutEnvCfg, _skew  # noqa: E402
from cut_paths import piece_cut  # noqa: E402
from encoder import BeltTracker, SimBeltEncoder  # noqa: E402
from kinematics import Knife, ry, target_rotation  # noqa: E402
from scheduler import CutScheduler  # noqa: E402

sys.path.insert(0, DOUGHSDG_DIR)  # the dough plugin imports sdg_core from there
spec = importlib.util.spec_from_file_location("dough_plugin", os.path.join(DOUGHSDG_DIR, "plugins", "dough_procedural.py"))
dough_plugin = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dough_plugin)
DP = {}
for c in reversed(dough_plugin.ProceduralDough.__mro__):
    DP.update(c.__dict__.get("defaults", {}))
DP["mesh_resolution"] = [96, 64]

ECFG = CutEnvCfg()  # same gains, speeds and tolerances as training
_env_json = os.path.join(os.path.dirname(os.path.abspath(args.policy)), "env_cfg.json")
if os.path.isfile(_env_json):  # settings the policy was trained with (train_cut_ppo.py --env ...)
    for _k, _v in json.load(open(_env_json)).items():
        setattr(ECFG, _k, _v)
    print(f"[line] training settings from {_env_json}", flush=True)
TRANSIT_V = 0.35  # m/s between cuts
HOME_X = 0.65  # belt x where the knife waits between cuts
ROBOT = "/World/abb_irb1200_5_90"
TRAY_Z, TRAY_TOP, TRAY_L, TRAY_W = 0.9114, 0.025, 0.70, 0.676
dev = "cuda:0"
os.makedirs(args.out, exist_ok=True)

# ------------------------------------------------------------------ stage
omni.usd.get_context().open_stage(args.scene)
for _ in range(10):
    app.update()
stage = omni.usd.get_context().get_stage()
stage.SetEditTarget(stage.GetSessionLayer())
for p in ("/World/Products/Bagg_01", "/World/CutPlan"):
    if stage.GetPrimAtPath(p):
        stage.GetPrimAtPath(p).SetActive(False)
sim_utils.modify_rigid_body_properties(ROBOT, sim_utils.RigidBodyPropertiesCfg(disable_gravity=True))
rng = np.random.default_rng(args.seed)
dough_looks = [str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdShade.Material) and "/SDG_Looks/dough/" in str(p.GetPath())]
tray_look = "/World/SDG_Looks/tray" if stage.GetPrimAtPath("/World/SDG_Looks/tray") else None


def make_tray(k, x0):
    """A tray with 1-N non-overlapping dough pieces; returns its prim, pieces' cut lines (tray coords)."""
    root = UsdGeom.Xform.Define(stage, f"/World/LineTrays/Tray_{k:03d}")
    UsdGeom.XformCommonAPI(root).SetTranslate(Gf.Vec3d(x0, 0.0, TRAY_Z))
    cube = UsdGeom.Cube.Define(stage, root.GetPath().AppendChild("Board"))
    cube.CreateSizeAttr(1.0)
    UsdGeom.XformCommonAPI(cube).SetScale(Gf.Vec3f(TRAY_L, TRAY_W, 0.0217))
    UsdGeom.XformCommonAPI(cube).SetTranslate(Gf.Vec3d(0, 0, TRAY_TOP - 0.0217 / 2))
    if tray_look:
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(UsdShade.Material(stage.GetPrimAtPath(tray_look)))
    pieces, circles = [], []
    for i in range(int(rng.integers(args.min_pieces, args.max_pieces + 1))):
        for _ in range(40):  # rejection sampling against overlap
            rnd = rng.random() < 0.3
            w = rng.uniform(0.06, 0.13)
            L = w * rng.uniform(0.9, 1.25) if rnd else rng.uniform(0.18, 0.54)
            h = rng.uniform(0.04, 0.10)
            yaw = rng.uniform(-np.pi / 2, np.pi / 2) if rng.random() < 0.4 else rng.normal(0, 0.05)
            reach = max(L, w) / 2
            cx, cy = rng.uniform(-TRAY_L / 2 + reach, TRAY_L / 2 - reach), rng.uniform(-TRAY_W / 2 + reach, TRAY_W / 2 - reach)
            d = np.array([np.cos(yaw), np.sin(yaw)])
            cs = [(np.array([cx, cy]) + d * f * (L - w) / 2, w / 2 + 0.012) for f in (-1, 0, 1)]
            if all(np.linalg.norm(a - b) > ra + rb for a, ra in cs for b, rb in circles):
                break
        else:
            continue
        circles += cs
        m = dough_plugin.dough_mesh(rng, L, w, h, DP, tray_top=TRAY_TOP)
        c, s = np.cos(yaw), np.sin(yaw)
        pts = m["points"] @ np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]]) + [cx, cy, 0]
        mesh = UsdGeom.Mesh.Define(stage, root.GetPath().AppendChild(f"Dough_{i}"))
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(m["counts"]))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(m["indices"]))
        mesh.CreateSubdivisionSchemeAttr("none")
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(
            Vt.Vec2fArray.FromNumpy(m["st"]))
        if dough_looks:
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
                UsdShade.Material(stage.GetPrimAtPath(dough_looks[i % len(dough_looks)])))
        # tray coordinates in x/y, world height in z; the scheduler builds the knife path from the contour
        cut = piece_cut(pts + [0, 0, TRAY_Z], faces=(m["counts"], m["indices"]))
        pieces.append({"id": f"p{i}", "kind": cut["kind"], "contour": cut["contour"], "direction": cut["direction"]})
        if not args.no_paths:  # planned cut drawn on the surface (the cut itself runs 3 mm under it)
            line = np.array(cut["contour"]) + [0, 0, 0.0005 - TRAY_Z]
            draw_curve(root.GetPath().AppendChild(f"PlannedCut_{i}"), line, (1.0, 0.85, 0.0))
    return root, pieces


def draw_curve(path, pts, color, width=0.003):
    c = UsdGeom.BasisCurves.Define(stage, path)
    c.CreateTypeAttr("linear")
    c.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*[float(v) for v in p]) for p in pts]))
    c.GetCurveVertexCountsAttr().Set([len(pts)])
    c.GetWidthsAttr().Set([width] * len(pts))
    c.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    return c


trays = []
x = args.photoeye_x - TRAY_L / 2 - 0.15
for k in range(args.trays):
    prim, pieces = make_tray(k, x)
    trays.append({"id": f"T{k:03d}", "prim": prim, "x": x, "pieces": pieces, "latched": False})
    x -= TRAY_L + args.gap
print(f"[line] {args.trays} trays, {sum(len(t['pieces']) for t in trays)} pieces, belt {args.belt_speed * 1000:.0f} mm/s", flush=True)

# ------------------------------------------------------------------ robot, IK, policy
sim = SimulationContext(SimulationCfg(dt=1 / 120, render_interval=4, device=dev))
robot = Articulation(ECFG.robot_cfg.replace(prim_path=ROBOT, spawn=None))
# Overview camera of the cut station (also the window's view when running with a GUI).
cam_prim = UsdGeom.Camera.Define(stage, "/World/RunCam")
cam_prim.CreateFocalLengthAttr(18)
cam_prim.CreateClippingRangeAttr(Gf.Vec2f(0.05, 100))
eye, at = Gf.Vec3d(0.55, -1.7, 1.8), Gf.Vec3d(0.6, 0.0, 1.0)
f = (at - eye).GetNormalized(); sv = Gf.Cross(f, Gf.Vec3d(0, 0, 1)).GetNormalized(); u = Gf.Cross(sv, f)
UsdGeom.Xformable(cam_prim).AddTransformOp().Set(Gf.Matrix4d(sv[0], sv[1], sv[2], 0, u[0], u[1], u[2], 0,
                                                              -f[0], -f[1], -f[2], 0, eye[0], eye[1], eye[2], 1))
cam = None
if args.video:
    import omni.replicator.core as rep
    rp = rep.create.render_product("/World/RunCam", (960, 540))
    cam = rep.AnnotatorRegistry.get_annotator("rgb")
    cam.attach(rp)
    os.makedirs(os.path.join(args.out, "frames"), exist_ok=True)
t_wall0 = time.time()
sim.reset()
print(f"[line] simulation ready in {time.time() - t_wall0:.1f}s", flush=True)
GUI = not args.headless
if GUI:
    from omni.kit.viewport.utility import get_active_viewport
    get_active_viewport().camera_path = "/World/RunCam"
t_wall0 = time.time()
robot.update(sim.get_physics_dt())
policy = torch.jit.load(args.policy, map_location=dev).eval()

knife = Knife()
R_off, p_off = ry(np.pi / 2) @ knife.R_tcp, ry(np.pi / 2) @ knife.p_tcp
off_pos = torch.tensor(p_off, device=dev, dtype=torch.float32).unsqueeze(0)
off_quat = M.quat_from_matrix(torch.tensor(R_off, device=dev, dtype=torch.float32)).unsqueeze(0)
ee_idx = robot.find_bodies("link_6")[0][0]
joint_ids = robot.find_joints("joint_[1-6]")[0]
lim = robot.data.soft_joint_pos_limits[0, joint_ids]
q_lo, q_hi = lim[:, 0], lim[:, 1]
ik = DifferentialIKController(DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls",
                                                          ik_params={"lambda_val": 0.05}), 1, dev)
ex, ez = torch.tensor([[1.0, 0, 0]], device=dev), torch.tensor([[0, 0, 1.0]], device=dev)
step_dt = sim.get_physics_dt() * ECFG.decimation


def tcp_w():
    return M.combine_frame_transforms(robot.data.body_pos_w[:, ee_idx], robot.data.body_quat_w[:, ee_idx], off_pos, off_quat)


def ik_step():
    rp_, rq_ = robot.data.root_pos_w, robot.data.root_quat_w
    J = robot.root_physx_view.get_jacobians()[:, ee_idx - 1, :, :][:, :, joint_ids].clone()
    Rb = M.matrix_from_quat(M.quat_inv(rq_))
    J[:, :3], J[:, 3:] = torch.bmm(Rb, J[:, :3]), torch.bmm(Rb, J[:, 3:])
    ee_b, _ = M.subtract_frame_transforms(rp_, rq_, robot.data.body_pos_w[:, ee_idx], robot.data.body_quat_w[:, ee_idx])
    tp, tq = tcp_w()
    tp_b, tq_b = M.subtract_frame_transforms(rp_, rq_, tp, tq)
    J[:, :3] = J[:, :3] - torch.bmm(_skew(tp_b - ee_b), J[:, 3:])
    q = robot.data.joint_pos[:, joint_ids]
    robot.set_joint_position_target(ik.compute(tp_b, tq_b, J, q).clamp(q_lo, q_hi), joint_ids=joint_ids)


def command_pose_w(pos_w, quat_w):
    p_b, q_b = M.subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, pos_w, quat_w)
    ik.set_command(torch.cat([p_b, q_b], 1))


encoder = SimBeltEncoder(args.counts_per_m, start_ticks=(1 << 32) - 50000, clock=lambda: sim_t)  # wraps early in the run
tracker = BeltTracker(args.counts_per_m)
sched = CutScheduler(args.counts_per_m, cut_speed=ECFG.cut_speed)


def ik_check(task, poly_now, drift):
    """The knife can reach the approach point now and the retract point after the belt has moved `drift` m."""
    from kinematics import ik as np_ik
    R = target_rotation(task.direction)
    q = robot.data.joint_pos[0, joint_ids].cpu().numpy().astype(float)
    q1, ok1 = np_ik(knife, R, poly_now[0], q, iters=80)
    if not ok1:
        return False
    _, ok2 = np_ik(knife, R, poly_now[-1] + [drift, 0.0, 0.0], q1, iters=80)
    return ok2

sim_t = 0.0
frame = 0


def physics_step():
    global sim_t
    for k in range(ECFG.decimation):
        ik_step()
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(sim.get_physics_dt())
        encoder.advance(args.belt_speed * sim.get_physics_dt())
        sim_t += sim.get_physics_dt()
    disp = args.belt_speed * sim_t
    for t in trays:
        tx = t["x"] + disp
        UsdGeom.XformCommonAPI(t["prim"]).SetTranslate(Gf.Vec3d(tx, 0.0, TRAY_Z))
        if not t["latched"] and tx + TRAY_L / 2 >= args.photoeye_x:  # front edge reaches the photoeye
            t["latched"] = True
            reading = encoder.read()
            sched.add_tray({"tray_id": t["id"], "latch_ticks": reading.ticks, "latch_x": args.photoeye_x - TRAY_L / 2,
                            "pieces": t["pieces"]})
            print(f"[line] t={sim_t:6.1f}s photoeye {t['id']}: {len(t['pieces'])} pieces queued", flush=True)
    tracker.update(encoder.read())
    if GUI:
        sim.render()
    global frame
    if cam is not None and frame % 2 == 0:
        sim.render()
        img = np.asarray(cam.get_data())
        if img.size:
            from PIL import Image
            Image.fromarray(img[..., :3]).save(os.path.join(args.out, "frames", f"f{frame // 2:05d}.jpg"), quality=80)
    frame += 1
    if frame % 60 == 0:
        print(f"[line] sim {sim_t:6.1f}s  wall {time.time() - t_wall0:6.1f}s  belt {tracker.position * 1000:7.1f} mm", flush=True)


def move_to(pos_w_fn, quat_w, tol=0.003, max_speed=0.35, timeout=8.0):
    """Straight-line transit with a speed limit; the goal may move with the belt."""
    t0 = sim_t
    while sim_t - t0 < timeout:
        goal = pos_w_fn()
        tp, tq = tcp_w()
        d = goal - tp
        n = torch.linalg.norm(d)
        rot = M.quat_error_magnitude(tq, quat_w).item()
        if n < tol and rot < np.radians(2):
            return True
        rv = M.axis_angle_from_quat(M.quat_mul(quat_w, M.quat_inv(tq)))  # remaining rotation
        ang = torch.linalg.norm(rv, dim=1)
        lim_ang = ang.clamp(max=0.05)  # at most ~3 deg per control step
        axis = rv / ang.clamp(min=1e-9).unsqueeze(1) if ang.item() > 1e-9 else ex
        q_cmd = M.quat_mul(M.quat_from_angle_axis(lim_ang, axis), tq)
        stepv = d * min(1.0, (max_speed * step_dt) / max(n.item(), 1e-9))
        command_pose_w(tp + stepv, q_cmd)
        physics_step()
    return False


def run_cut(task):
    """The trained policy cuts one piece; observations exactly as in training (cut_env.CutEnv)."""
    ticks0 = encoder.read().ticks
    poly0 = torch.tensor(sched.world_poly(task, ticks0), device=dev, dtype=torch.float32)
    tray = next(t for t in trays if t["id"] == task.tray_id)
    x_enc0 = sched.tray_x(task, ticks0)  # where the encoder says the tray is now

    def true_shift():
        """Metres to add to the encoder-based path to get the true geometry (latch + quantisation error)."""
        return tray["x"] + args.belt_speed * sim_t - x_enc0
    seg = torch.linalg.norm(poly0[1:] - poly0[:-1], dim=1)
    cum = torch.cat([torch.zeros(1, device=dev), torch.cumsum(seg, 0)])
    total, cut_start, cut_end = cum[-1], cum[1], cum[-2]
    c_speed = ECFG.cut_speed
    q_des = M.quat_from_matrix(torch.tensor(target_rotation(task.direction), device=dev, dtype=torch.float32)).unsqueeze(0)

    def interp(s):
        i = int(torch.searchsorted(cum, s).clamp(1, len(cum) - 1))
        w = ((s - cum[i - 1]) / (cum[i] - cum[i - 1]).clamp(min=1e-9)).clamp(0, 1)
        tg = poly0[i] - poly0[i - 1]
        return poly0[i - 1] * (1 - w) + poly0[i] * w, tg / torch.linalg.norm(tg).clamp(min=1e-9)

    act = torch.zeros(1, 6, device=dev)
    v_est, prev_dq, t_ep = 0.0, 0.0, 0.0
    errs, in_tol, in_cut_n = [], 0, 0
    trail = []
    T = float(total) / c_speed + ECFG.wait_s + 3.0
    while t_ep < T:
        dq = (encoder.read().ticks - ticks0) % (1 << 32) / args.counts_per_m  # belt metres since cut start
        s = torch.clamp(torch.tensor(c_speed * (t_ep - ECFG.wait_s), device=dev), min=0.0, max=float(total))
        p, tg = interp(s)
        p_look, _ = interp(torch.clamp(s + 0.02, max=float(total)))
        tp, tq = tcp_w()
        target_true = p + ex[0] * true_shift()
        e_obs = (p + ex[0] * dq) - tp[0]
        look = (p_look + ex[0] * dq) - tp[0]
        rot_vec = M.axis_angle_from_quat(M.quat_mul(q_des, M.quat_inv(tq)))
        moving = float(t_ep > ECFG.wait_s and float(s) < float(total))
        ref_vel = tg * c_speed * moving + ex[0] * v_est
        in_cut = bool(cut_start < s < cut_end)
        to_b = M.quat_inv(robot.data.root_quat_w)
        q = robot.data.joint_pos[:, joint_ids]
        obs = torch.cat([M.quat_apply(to_b, e_obs.unsqueeze(0)) * 20.0, M.quat_apply(to_b, rot_vec),
                         M.quat_apply(to_b, ref_vel.unsqueeze(0)) * 20.0,
                         M.quat_apply(to_b, look.unsqueeze(0)) * 20.0,
                         (s / total).view(1, 1), torch.tensor([[float(in_cut)]], device=dev),
                         2.0 * (q - q_lo) / (q_hi - q_lo) - 1.0, robot.data.joint_vel[:, joint_ids] * 0.1, act], 1)
        with torch.no_grad():
            act = policy(torch.nan_to_num(obs).clamp(-10, 10)).clamp(-1, 1)
        # same action -> IK command as training
        tp_b, tq_b = M.subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, tp, tq)
        rvec = act[:, 3:] * ECFG.max_step_rot
        ang = torch.linalg.norm(rvec, dim=1)
        axis = rvec / ang.clamp(min=1e-9).unsqueeze(1) if ang.item() > 1e-9 else ex
        ik.set_command(torch.cat([tp_b + act[:, :3] * ECFG.max_step_pos, M.quat_mul(M.quat_from_angle_axis(ang, axis), tq_b)], 1))
        v = (dq - prev_dq) / step_dt
        v_est += 0.3 * (v - v_est)
        prev_dq = dq
        # score against the true geometry
        e = target_true - tp[0]
        if in_cut:
            lat_dir = torch.linalg.cross(ez[0], tg)
            lat_dir = lat_dir / torch.linalg.norm(lat_dir).clamp(min=1e-9)
            blade = M.quat_apply(tq, ez)[0]
            tilt = float(torch.rad2deg(torch.acos((-blade[2]).clamp(-1, 1))))
            dep, lat = float(e[2]), float(e @ lat_dir)
            errs.append((dep, lat, tilt))
            in_cut_n += 1
            in_tol += int(abs(dep) < ECFG.tol_depth and abs(lat) < ECFG.tol_lateral and tilt < ECFG.tol_tilt_deg)
            if not args.no_paths:  # knife trail, lifted to where the dough surface should be
                trail.append(tp[0].cpu().numpy() - [tray["x"] + args.belt_speed * sim_t, 0.0, TRAY_Z] + [0, 0, 0.0035])
                if len(trail) >= 2 and len(trail) % 3 == 0:
                    draw_curve(tray["prim"].GetPath().AppendChild(f"KnifeTrail_{task.piece_id}"), trail, (0.9, 0.1, 0.1))
        physics_step()
        t_ep += step_dt
        if torch.linalg.norm(e) > ECFG.lost_dist:
            break
        # done once the reference has reached the retract point and the knife is out of the dough with it
        # (the policy does not settle within 5 mm there: ending the episode stopped its reward in training).
        if float(s) >= float(total) - 1e-6 and torch.linalg.norm(e) < 0.015:
            break
    E = np.array(errs) if errs else np.zeros((1, 3))
    frac = in_tol / max(in_cut_n, 1)
    return {"in_tolerance_fraction": round(frac, 3), "pass": frac >= 0.9,
            "max_depth_err_mm": round(float(np.abs(E[:, 0]).max() * 1000), 2),
            "max_lateral_err_mm": round(float(np.abs(E[:, 1]).max() * 1000), 2),
            "max_tilt_deg": round(float(E[:, 2].max()), 2), "cut_s": round(t_ep, 2),
            "end_error_mm": round(float(torch.linalg.norm(e)) * 1000, 1)}


# ------------------------------------------------------------------ main loop
home_pos = torch.tensor([[HOME_X, 0.0, 1.15]], device=dev)
home_quat = M.quat_from_matrix(torch.tensor(target_rotation([1, 0, 0]), device=dev, dtype=torch.float32)).unsqueeze(0)
t_wall = time.time()
while sim_t < args.max_time:
    reading = encoder.read()
    task = sched.next_task(reading.ticks, max(tracker.speed, 0.0), check=ik_check)
    if task is None:
        last_tray_x = min(t["x"] for t in trays) + args.belt_speed * sim_t
        if all(t["latched"] for t in trays) and all(k.status in ("done", "missed") for k in sched.tasks):
            break
        if last_tray_x - TRAY_L / 2 > sched.reach_x[1] + 0.1:
            break
        # Nothing reachable yet: head home (or hold there) and let the belt run.
        if not move_to(lambda: home_pos[0], home_quat, timeout=step_dt):
            continue
        tp, tq = tcp_w()
        command_pose_w(tp, tq)
        physics_step()
        continue
    q_des = M.quat_from_matrix(torch.tensor(target_rotation(task.direction), device=dev, dtype=torch.float32)).unsqueeze(0)
    approach = lambda: torch.tensor(sched.world_poly(task, encoder.read().ticks)[0], device=dev, dtype=torch.float32)  # noqa: E731
    if not move_to(approach, q_des, max_speed=TRANSIT_V):
        task.status, task.result = "missed", {"reason": "transit timeout"}
        continue
    task.result = run_cut(task)
    task.status = "done"
    print(f"[line] t={sim_t:6.1f}s {task.tray_id}/{task.piece_id} ({task.kind}): "
          f"{task.result['in_tolerance_fraction'] * 100:.0f}% in tolerance, depth max {task.result['max_depth_err_mm']} mm, "
          f"lateral max {task.result['max_lateral_err_mm']} mm, tilt max {task.result['max_tilt_deg']} deg "
          f"-> {'PASS' if task.result['pass'] else 'FAIL'}", flush=True)

done = [k for k in sched.tasks if k.status == "done"]
report = {"belt_speed": args.belt_speed, "trays": args.trays, "pieces": len(sched.tasks),
          "cut": len(done), "missed": sum(k.status == "missed" for k in sched.tasks),
          "never_reached": sum(k.status == "pending" for k in sched.tasks),
          "passed": sum(bool(k.result.get("pass")) for k in done),
          "pass_rate_of_all_pieces": round(sum(bool(k.result.get("pass")) for k in done) / max(len(sched.tasks), 1), 3),
          "sim_seconds": round(sim_t, 1), "wall_seconds": round(time.time() - t_wall, 1),
          "per_tray": sched.summary(),
          "cuts": [{"tray": k.tray_id, "piece": k.piece_id, "kind": k.kind, "status": k.status, **k.result} for k in sched.tasks]}
json.dump(report, open(os.path.join(args.out, "report.json"), "w"), indent=1)
if args.video:
    import glob

    import cv2
    shots = sorted(glob.glob(os.path.join(args.out, "frames", "f*.jpg")))
    if shots:
        h, w = cv2.imread(shots[0]).shape[:2]
        vw = cv2.VideoWriter(os.path.join(args.out, "run.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 15, (w, h))
        for f in shots:
            vw.write(cv2.imread(f))
        vw.release()
        print(f"[line] video {os.path.join(args.out, 'run.mp4')} ({len(shots)} frames)", flush=True)
print(f"[line] RESULT {report['passed']}/{report['pieces']} pieces cut in tolerance "
      f"({report['missed']} missed, {report['never_reached']} not reached) in {report['sim_seconds']} s -> {args.out}", flush=True)
# Results are on disk; Isaac Sim can hang in close() after camera/replicator use, so exit directly.
if GUI and args.hold:
    print("[line] run finished; close the Isaac Sim window to exit", flush=True)
    while app.is_running():
        sim.render()
sys.stdout.flush()
os._exit(0)
