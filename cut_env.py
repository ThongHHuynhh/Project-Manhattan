"""Isaac Lab environment: score one dough piece with the ultrasonic knife while the belt moves.

One episode = one cut from path_library.npz (random trays of procedural dough, 1-5 pieces,
long and round, any rotation). The robot starts at the approach pose; a reference point travels the
cut line (approach -> 3 mm under the surface -> retract) at `cut_speed` while the whole line moves
with the belt. The policy only sees belt motion through quantised encoder ticks (`counts_per_m`).

Actions (30 Hz): knife-tip moves in the robot base frame, up to `max_step_pos` m and `max_step_rot` rad;
differential IK on the TCP turns them into joint targets every physics step.
"""
import json
import os

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as M

from kinematics import Knife, ry, target_rotation

HERE = os.path.dirname(os.path.abspath(__file__))
from paths import ROBOT_USD  # noqa: E402
BASE_POS = (0.225, 0.0, 1.9608)  # as in the K2 scenes
BASE_ROT = (0.7071068, 0.0, 0.7071068, 0.0)


@configclass
class CutEnvCfg(DirectRLEnvCfg):
    decimation = 4
    episode_length_s = 16.0
    action_space = 6
    observation_space = 32
    state_space = 0
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=4)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=3.0, replicate_physics=True)
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ROBOT_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=1),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=BASE_POS, rot=BASE_ROT, joint_pos={"joint_[1-6]": 0.0}),
        actuators={"arm": ImplicitActuatorCfg(joint_names_expr=["joint_[1-6]"], stiffness=4000.0, damping=200.0,
                                              effort_limit_sim=2000.0, velocity_limit_sim=5.0)},
    )
    library: str = os.path.join(HERE, "path_library.npz")
    cut_speed: float = 0.05  # m/s along the cut
    belt_speed_max: float = 0.03  # m/s, sampled 0..max per episode
    counts_per_m: float = 10000.0  # encoder resolution (0.1 mm/tick)
    reach_x_max: float = 1.0  # the belt never carries a cut beyond this x during an episode
    wait_s: float = 0.3  # reference waits this long before moving
    max_step_pos: float = 0.004
    max_step_rot: float = 0.03
    tol_depth: float = 0.001
    tol_lateral: float = 0.002
    tol_tilt_deg: float = 5.0
    lost_dist: float = 0.04
    start_noise: float = 0.01  # rad on the start joints
    # Reward weights. Tracking terms are exp(-(err/scale)^2).
    w_track_fine: float = 1.0  # cross-track (depth + sideways) error, scale track_fine_m
    track_fine_m: float = 0.002
    w_track_coarse: float = 0.5  # same error, wide scale: gradient when far away
    track_coarse_m: float = 0.01
    w_along: float = 0.3  # lag/lead along the cut, scale along_m
    along_m: float = 0.01
    w_rot: float = 0.5  # blade orientation, scale rot_rad
    rot_rad: float = 0.05
    w_in_tol: float = 1.0  # bonus per cut step inside all three tolerances
    w_action: float = 0.01  # penalty on action size
    w_smooth: float = 0.05  # penalty on action change (jerk)
    w_joint_vel: float = 0.0005
    w_success: float = 5.0  # reaching the end of the path
    w_lost: float = 5.0  # penalty when the tip is > lost_dist from the reference


# Fields that train_cut_ppo.py --env KEY=VALUE may override; saved to env_cfg.json with the policy.
TUNABLE = ["cut_speed", "belt_speed_max", "counts_per_m", "reach_x_max", "wait_s", "max_step_pos", "max_step_rot",
           "tol_depth", "tol_lateral", "tol_tilt_deg", "lost_dist", "start_noise", "episode_length_s",
           "w_track_fine", "track_fine_m", "w_track_coarse", "track_coarse_m", "w_along", "along_m", "w_rot",
           "rot_rad", "w_in_tol", "w_action", "w_smooth", "w_joint_vel", "w_success", "w_lost"]


def apply_overrides(cfg, items=()):
    """KEY=VALUE items (values in JSON) applied to the config."""
    vals = {}
    for item in items:
        key, _, raw = item.partition("=")
        vals[key] = json.loads(raw)
    for key, val in vals.items():
        if key in TUNABLE:
            setattr(cfg, key, type(getattr(cfg, key))(val))
        else:
            raise SystemExit(f"{key}: not tunable; choose from {TUNABLE}")
    return vals


def _skew(v):
    z = torch.zeros_like(v[:, 0])
    return torch.stack([torch.stack([z, -v[:, 2], v[:, 1]], -1),
                        torch.stack([v[:, 2], z, -v[:, 0]], -1),
                        torch.stack([-v[:, 1], v[:, 0], z], -1)], 1)


class CutEnv(DirectRLEnv):
    cfg: CutEnvCfg

    def __init__(self, cfg: CutEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        dev = self.device
        lib = np.load(cfg.library)
        self.paths = torch.tensor(lib["paths"], device=dev, dtype=torch.float32)  # (K, N, 3), K2 world frame
        self.counts = torch.tensor(lib["counts"], device=dev, dtype=torch.long)
        self.q0_lib = torch.tensor(lib["q0"], device=dev, dtype=torch.float32)
        dirs = lib["meta"][:, :2]
        Rs = np.stack([target_rotation([d[0], d[1], 0.0]) for d in dirs]).astype(np.float32)
        self.quat_des_lib = M.quat_from_matrix(torch.tensor(Rs, device=dev))
        seg = torch.linalg.norm(self.paths[:, 1:] - self.paths[:, :-1], dim=-1)
        self.cum = torch.cat([torch.zeros_like(seg[:, :1]), torch.cumsum(seg, dim=1)], dim=1)  # arc length
        k = torch.arange(len(self.paths), device=dev)
        self.total = self.cum[k, self.counts - 1]
        self.cut_start, self.cut_end = self.cum[:, 1], self.cum[k, self.counts - 2]  # between approach and retract
        self.max_x = torch.stack([self.paths[i, : self.counts[i], 0].max() for i in range(len(self.paths))])
        knife = Knife()
        R_off, p_off = ry(np.pi / 2) @ knife.R_tcp, ry(np.pi / 2) @ knife.p_tcp  # TCP in the link_6 frame
        self.off_pos = torch.tensor(p_off, device=dev, dtype=torch.float32).repeat(self.num_envs, 1)
        self.off_quat = M.quat_from_matrix(torch.tensor(R_off, device=dev, dtype=torch.float32)).repeat(self.num_envs, 1)
        self.ee_idx = self.robot.find_bodies("link_6")[0][0]
        self.jac_idx = self.ee_idx - 1  # fixed base: the root body has no jacobian row
        self.joint_ids = self.robot.find_joints("joint_[1-6]")[0]
        self.ik = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls",
                                        ik_params={"lambda_val": 0.05}), self.num_envs, dev)
        n = self.num_envs
        self.pid = torch.zeros(n, dtype=torch.long, device=dev)
        self.belt_speed = torch.zeros(n, device=dev)
        self.belt_v_est = torch.zeros(n, device=dev)
        self.prev_disp_q = torch.zeros(n, device=dev)
        self.act = torch.zeros(n, 6, device=dev)
        self.prev_act = torch.zeros(n, 6, device=dev)
        self.tol_steps = torch.zeros(n, device=dev)
        self.cut_steps = torch.zeros(n, device=dev)
        self.success = torch.zeros(n, dtype=torch.bool, device=dev)
        lim = self.robot.data.soft_joint_pos_limits[0, self.joint_ids]
        self.q_lo, self.q_hi = lim[:, 0], lim[:, 1]
        self.ex = torch.tensor([1.0, 0.0, 0.0], device=dev)
        self.ez = torch.tensor([0.0, 0.0, 1.0], device=dev)

    # ---------------------------------------------------------------- scene
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        light = sim_utils.DomeLightCfg(intensity=1500.0)
        light.func("/World/Light", light)

    # ---------------------------------------------------------------- helpers
    def _tcp_w(self):
        ee_pos = self.robot.data.body_pos_w[:, self.ee_idx]
        ee_quat = self.robot.data.body_quat_w[:, self.ee_idx]
        return M.combine_frame_transforms(ee_pos, ee_quat, self.off_pos, self.off_quat)

    def _interp(self, pid, s):
        cum = self.cum[pid]
        idx = torch.searchsorted(cum, s.unsqueeze(1)).squeeze(1).clamp(1, cum.shape[1] - 1)
        a, b = cum.gather(1, (idx - 1).unsqueeze(1)).squeeze(1), cum.gather(1, idx.unsqueeze(1)).squeeze(1)
        w = ((s - a) / (b - a).clamp(min=1e-9)).clamp(0, 1).unsqueeze(1)
        P = self.paths[pid]
        p0 = P.gather(1, (idx - 1).view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)
        p1 = P.gather(1, idx.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)
        tangent = p1 - p0
        tangent = tangent / torch.linalg.norm(tangent, dim=1, keepdim=True).clamp(min=1e-9)
        return p0 * (1 - w) + p1 * w, tangent

    def _state(self):
        """Reference, encoder reading and all errors for the current step."""
        t = self.episode_length_buf.float() * self.step_dt
        disp = self.belt_speed * t
        disp_q = torch.floor(disp * self.cfg.counts_per_m) / self.cfg.counts_per_m  # what the encoder reports
        s = (self.cfg.cut_speed * (t - self.cfg.wait_s)).clamp(min=0.0)
        s = torch.minimum(s, self.total[self.pid])
        p, tangent = self._interp(self.pid, s)
        p_look, _ = self._interp(self.pid, torch.minimum(s + 0.02, self.total[self.pid]))
        origin = self.scene.env_origins
        target_w = origin + p + disp.unsqueeze(1) * self.ex
        target_obs_w = origin + p + disp_q.unsqueeze(1) * self.ex
        look_obs_w = origin + p_look + disp_q.unsqueeze(1) * self.ex
        tcp_pos, tcp_quat = self._tcp_w()
        e = target_w - tcp_pos
        lat_dir = torch.linalg.cross(self.ez.expand_as(tangent), tangent)
        lat_dir = lat_dir / torch.linalg.norm(lat_dir, dim=1, keepdim=True).clamp(min=1e-9)
        depth, lat, along = e[:, 2], (e * lat_dir).sum(1), (e * tangent).sum(1)
        q_des = self.quat_des_lib[self.pid]
        rot_vec_w = M.axis_angle_from_quat(M.quat_mul(q_des, M.quat_inv(tcp_quat)))
        blade = M.quat_apply(tcp_quat, self.ez.expand(self.num_envs, 3))
        tilt = torch.rad2deg(torch.acos((-blade[:, 2]).clamp(-1, 1)))
        in_cut = (s > self.cut_start[self.pid]) & (s < self.cut_end[self.pid])
        in_tol = in_cut & (depth.abs() < self.cfg.tol_depth) & (lat.abs() < self.cfg.tol_lateral) & (tilt < self.cfg.tol_tilt_deg)
        moving = (t > self.cfg.wait_s) & (s < self.total[self.pid])
        ref_vel = tangent * (self.cfg.cut_speed * moving.float()).unsqueeze(1) + self.belt_v_est.unsqueeze(1) * self.ex
        return dict(t=t, s=s, disp_q=disp_q, tcp_pos=tcp_pos, tcp_quat=tcp_quat, e=e, e_obs=target_obs_w - tcp_pos,
                    look=look_obs_w - tcp_pos, depth=depth, lat=lat, along=along, rot_vec_w=rot_vec_w,
                    rot_err=torch.linalg.norm(rot_vec_w, dim=1), tilt=tilt, in_cut=in_cut, in_tol=in_tol, ref_vel=ref_vel)

    # ---------------------------------------------------------------- RL interface
    def _pre_physics_step(self, actions):
        self.prev_act[:] = self.act
        self.act[:] = actions.clamp(-1.0, 1.0)
        root_pos, root_quat = self.robot.data.root_pos_w, self.robot.data.root_quat_w
        tcp_pos_w, tcp_quat_w = self._tcp_w()
        tcp_pos_b, tcp_quat_b = M.subtract_frame_transforms(root_pos, root_quat, tcp_pos_w, tcp_quat_w)
        dpos = self.act[:, :3] * self.cfg.max_step_pos
        rvec = self.act[:, 3:] * self.cfg.max_step_rot
        ang = torch.linalg.norm(rvec, dim=1)
        axis = torch.where(ang.unsqueeze(1) > 1e-9, rvec / ang.clamp(min=1e-9).unsqueeze(1), self.ex.expand_as(rvec))
        cmd_quat = M.quat_mul(M.quat_from_angle_axis(ang, axis), tcp_quat_b)
        self.ik.set_command(torch.cat([tcp_pos_b + dpos, cmd_quat], dim=1))
        # encoder-based belt speed estimate, as a controller on the line would have it
        st = self._state()
        v = (st["disp_q"] - self.prev_disp_q) / self.step_dt
        self.belt_v_est += 0.3 * (v - self.belt_v_est)
        self.prev_disp_q[:] = st["disp_q"]

    def _apply_action(self):
        root_pos, root_quat = self.robot.data.root_pos_w, self.robot.data.root_quat_w
        J = self.robot.root_physx_view.get_jacobians()[:, self.jac_idx, :, :][:, :, self.joint_ids].clone()
        Rb = M.matrix_from_quat(M.quat_inv(root_quat))
        J[:, :3] = torch.bmm(Rb, J[:, :3])
        J[:, 3:] = torch.bmm(Rb, J[:, 3:])
        ee_pos_b, _ = M.subtract_frame_transforms(root_pos, root_quat, self.robot.data.body_pos_w[:, self.ee_idx],
                                                  self.robot.data.body_quat_w[:, self.ee_idx])
        tcp_pos_w, tcp_quat_w = self._tcp_w()
        tcp_pos_b, tcp_quat_b = M.subtract_frame_transforms(root_pos, root_quat, tcp_pos_w, tcp_quat_w)
        J[:, :3] = J[:, :3] - torch.bmm(_skew(tcp_pos_b - ee_pos_b), J[:, 3:])  # jacobian of the knife tip
        q = self.robot.data.joint_pos[:, self.joint_ids]
        q_des = self.ik.compute(tcp_pos_b, tcp_quat_b, J, q).clamp(self.q_lo, self.q_hi)
        self.robot.set_joint_position_target(q_des, joint_ids=self.joint_ids)

    def _get_observations(self):
        st = self._state()
        to_b = M.quat_inv(self.robot.data.root_quat_w)
        q = self.robot.data.joint_pos[:, self.joint_ids]
        obs = torch.cat([
            M.quat_apply(to_b, st["e_obs"]) * 20.0,
            M.quat_apply(to_b, st["rot_vec_w"]),
            M.quat_apply(to_b, st["ref_vel"]) * 20.0,
            M.quat_apply(to_b, st["look"]) * 20.0,
            (st["s"] / self.total[self.pid]).unsqueeze(1),
            st["in_cut"].float().unsqueeze(1),
            2.0 * (q - self.q_lo) / (self.q_hi - self.q_lo) - 1.0,
            self.robot.data.joint_vel[:, self.joint_ids] * 0.1,
            self.act,
        ], dim=1)
        return {"policy": torch.nan_to_num(obs).clamp(-10, 10)}

    def _get_dones(self):
        st = self._state()
        self._st = st
        dist = torch.linalg.norm(st["e"], dim=1)
        lost = dist > self.cfg.lost_dist
        self.cut_steps += st["in_cut"].float()
        self.tol_steps += st["in_tol"].float()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        self.success = (st["s"] >= self.total[self.pid] - 1e-6) & (dist < 0.005)
        return lost | self.success, time_out

    def _get_rewards(self):
        st, c = self._st, self.cfg
        cross = torch.sqrt(st["depth"] ** 2 + st["lat"] ** 2)
        r = (c.w_track_fine * torch.exp(-(cross / c.track_fine_m) ** 2)
             + c.w_track_coarse * torch.exp(-(cross / c.track_coarse_m) ** 2)
             + c.w_along * torch.exp(-(st["along"] / c.along_m) ** 2)
             + c.w_rot * torch.exp(-(st["rot_err"] / c.rot_rad) ** 2)
             + c.w_in_tol * st["in_tol"].float()
             - c.w_action * (self.act ** 2).sum(1) - c.w_smooth * ((self.act - self.prev_act) ** 2).sum(1)
             - c.w_joint_vel * (self.robot.data.joint_vel[:, self.joint_ids] ** 2).sum(1))
        dist = torch.linalg.norm(st["e"], dim=1)
        r = r + c.w_success * self.success.float() - c.w_lost * (dist > c.lost_dist).float()
        return r

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        if len(env_ids) and self.cut_steps[env_ids].sum() > 0:
            frac = self.tol_steps[env_ids] / self.cut_steps[env_ids].clamp(min=1)
            self.extras["log"] = {"cut_in_tolerance_fraction": frac.mean().item(),
                                  "cuts_90pct_in_tolerance": (frac >= 0.9).float().mean().item(),
                                  "episode_success": self.success[env_ids].float().mean().item()}
        super()._reset_idx(env_ids)
        n = len(env_ids)
        pid = torch.randint(0, len(self.paths), (n,), device=self.device)
        self.pid[env_ids] = pid
        q0 = (self.q0_lib[pid] + (torch.rand(n, 6, device=self.device) * 2 - 1) * self.cfg.start_noise)
        q0 = q0.clamp(self.q_lo, self.q_hi)
        # belt speed: random, but never so fast that the belt carries the cut out of reach
        T = self.total[pid] / self.cfg.cut_speed + self.cfg.wait_s + 1.0
        v = torch.rand(n, device=self.device) * self.cfg.belt_speed_max
        v = torch.minimum(v, ((self.cfg.reach_x_max - self.max_x[pid]) / T).clamp(min=0.0))
        self.belt_speed[env_ids] = v
        self.robot.write_joint_state_to_sim(q0, torch.zeros_like(q0), joint_ids=self.joint_ids, env_ids=env_ids)
        self.robot.set_joint_position_target(q0, joint_ids=self.joint_ids, env_ids=env_ids)
        self.belt_v_est[env_ids] = 0.0
        self.prev_disp_q[env_ids] = 0.0
        self.act[env_ids] = 0.0
        self.prev_act[env_ids] = 0.0
        self.tol_steps[env_ids] = 0.0
        self.cut_steps[env_ids] = 0.0
        self.success[env_ids] = False
        self.ik.reset(env_ids)
