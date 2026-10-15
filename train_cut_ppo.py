"""Train the dough-scoring policy with PPO (rsl_rl), then export it for the tray runner.

    cd ~/IsaacLab
    ./isaaclab.sh -p ~/TH/Project-Manhattan/train_cut_ppo.py \
        --headless --num_envs 1024 --max_iterations 1500
"""
import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher

HERE = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--max_iterations", type=int, default=1500)
parser.add_argument("--run_name", default=time.strftime("%Y%m%d-%H%M%S"))
parser.add_argument("--resume", default=None, help="checkpoint (.pt) to continue from")
parser.add_argument("--library", default=None, help="cut-path library (.npz); default path_library.npz next to this script")
parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                    help="override a CutEnvCfg field, e.g. --env belt_speed_max=0.04 (repeatable; see cut_env.TUNABLE)")
parser.add_argument("--export_only", action="store_true", help="with --resume: only export that checkpoint (no training)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

sys.path.insert(0, HERE)
from isaaclab_rl.rsl_rl import (RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg,  # noqa: E402
                                RslRlVecEnvWrapper, export_policy_as_jit)
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import json  # noqa: E402

from cut_env import TUNABLE, CutEnv, CutEnvCfg, apply_overrides  # noqa: E402

cfg = CutEnvCfg()
cfg.scene.num_envs = args.num_envs
apply_overrides(cfg, args.env)
if args.library:
    cfg.library = args.library
env = RslRlVecEnvWrapper(CutEnv(cfg))

agent = RslRlOnPolicyRunnerCfg(
    num_steps_per_env=48,
    max_iterations=args.max_iterations,
    save_interval=100,
    experiment_name="dough_cut",
    run_name=args.run_name,
    obs_groups={"policy": ["policy"], "critic": ["policy"]},
    policy=RslRlPpoActorCriticCfg(
        init_noise_std=0.6,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    ),
    algorithm=RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.005,
        num_learning_epochs=5, num_mini_batches=4, learning_rate=3.0e-4, schedule="adaptive",
        gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0,
    ),
)
log_dir = os.path.join(HERE, "logs", "dough_cut", args.run_name)
runner = OnPolicyRunner(env, agent.to_dict(), log_dir=log_dir, device=agent.device)
if args.resume:
    runner.load(args.resume)
os.makedirs(os.path.join(log_dir, "exported"), exist_ok=True)
env_cfg = {k: getattr(cfg, k) for k in TUNABLE}
with open(os.path.join(log_dir, "exported", "env_cfg.json"), "w") as f:  # the runner reads this next to policy.pt
    json.dump(env_cfg, f, indent=1)
print(f"[train] {args.num_envs} envs, {args.max_iterations} iterations -> {log_dir}; overrides: {args.env}", flush=True)
if not args.export_only:
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)

# Export the actor (with its observation normaliser) as TorchScript for the tray runner.
policy = runner.alg.policy
normalizer = getattr(policy, "actor_obs_normalizer", None)
export_policy_as_jit(policy, normalizer, path=os.path.join(log_dir, "exported"), filename="policy.pt")
print(f"[train] exported {os.path.join(log_dir, 'exported', 'policy.pt')}", flush=True)
env.close()
app.close()
