"""Feasibility check: a scripted follower (steer straight at the reference) through the same IK.

With --policy it runs a trained policy instead (settings from its env_cfg.json, then --env),
e.g. to measure it with faster belts: --env belt_speed_max=0.04
"""
import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

HERE = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--library", default=None)
parser.add_argument("--episodes", type=int, default=64)
parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
parser.add_argument("--policy", default=None, help="exported policy.pt to evaluate instead of the scripted follower")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
sys.path.insert(0, HERE)
import torch  # noqa: E402
from isaaclab.utils import math as M  # noqa: E402

from cut_env import TUNABLE, CutEnv, CutEnvCfg, apply_overrides  # noqa: E402

cfg = CutEnvCfg()
cfg.scene.num_envs = args.num_envs
if args.policy:  # the settings the policy was trained with
    trained = json.load(open(os.path.join(os.path.dirname(os.path.abspath(args.policy)), "env_cfg.json")))
    apply_overrides(cfg, [f"{k}={json.dumps(v)}" for k, v in trained.items() if k in TUNABLE])
apply_overrides(cfg, args.env)
if args.library:
    cfg.library = args.library
env = CutEnv(cfg)
policy = torch.jit.load(args.policy, map_location=env.device).eval() if args.policy else None
obs, _ = env.reset()
done_eps, fracs, c90, succ, worst_depth = 0, [], [], [], []
max_depth = torch.zeros(env.num_envs, device=env.device)
while done_eps < args.episodes:
    if policy is not None:
        with torch.no_grad():
            a = policy(obs["policy"]).clamp(-1, 1)
    else:
        st = env._state()
        to_b = M.quat_inv(env.robot.data.root_quat_w)
        e_b = M.quat_apply(to_b, st["e_obs"] + st["ref_vel"] * env.step_dt)
        r_b = M.quat_apply(to_b, st["rot_vec_w"])
        a = torch.cat([e_b / cfg.max_step_pos, r_b / cfg.max_step_rot], 1).clamp(-1, 1)
    obs, rew, term, trunc, extras = env.step(a)
    max_depth = torch.maximum(max_depth, torch.where(env._st["in_cut"], env._st["depth"].abs(), torch.zeros_like(max_depth)))
    done = term | trunc
    if done.any():
        log = extras.get("log", {})
        if "cut_in_tolerance_fraction" in log:
            fracs.append(log["cut_in_tolerance_fraction"]); c90.append(log["cuts_90pct_in_tolerance"]); succ.append(log["episode_success"])
        worst_depth += max_depth[done].tolist()
        max_depth[done] = 0
        done_eps += int(done.sum())
mean = lambda v: sum(v) / max(len(v), 1)  # noqa: E731
who = f"policy {args.policy}" if policy is not None else "scripted follower"
print(f"CHECK {who}: {done_eps} episodes, cut steps in tolerance {mean(fracs):.3f}, cuts >=90% in tolerance {mean(c90):.3f}, "
      f"success {mean(succ):.3f}, worst depth error median {sorted(worst_depth)[len(worst_depth) // 2] * 1000:.2f} mm", flush=True)
env.close()
app.close()
