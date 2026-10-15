# Project-Manhattan — robotic dough scoring (v1)

The ABB IRB 1200 on the K2 line scores every dough piece on every tray with the ultrasonic knife:
one straight cut, 3 mm deep (±1 mm), ±2 mm sideways, blade within 5° of vertical, over the middle 80%
of the piece (long pieces along their length, round pieces along the belt). Trays arrive back to back
on a moving belt (trained up to 30 mm/s); how many trays and pieces is never fixed.

Full guide (architecture, every launch option, tuning): `docs/K2-dough-scoring.html`.

## Pipeline

| Step | File | What it does |
|---|---|---|
| Knife | `knife.json`, `kinematics.py` | Knife geometry measured from the CAD (`tn__00RB025COG2_pGu4`, ATI 9120 changer plate on `tool0`); TCP = blade tip, z along the blade, x = cutting direction. Numpy FK/IK of the robot. |
| Reach | `reach_study.py` | Tray positions where the whole tray is cuttable with the blade vertical: x = 0.4–0.9 m (station at 0.65 m). |
| Scene | `build_006.py` | `USD/K2-CV-STEP-006-CUT.usd` from 005: knife referenced onto `tool0`, floating knife off, tray at the station. |
| Cut lines | `cut_paths.py`, `plan_cuts.py` | Top surface of each piece from its mesh triangles (`top_height`); cut 3 mm under it; straight down in, straight up out (`cut_path`); nearest-next order; drawn in 006 (`/World/CutPlan`). |
| Training data | `build_path_library.py` → `path_library.npz` | 2,610 cuts from random generator trays (1–5 pieces, long and round, any rotation), each with an IK start pose. |
| RL environment | `cut_env.py` | Isaac Lab, robot only, 1,024 parallel copies. Reference moves along the cut at 50 mm/s while the line drifts with the belt (0–30 mm/s); belt seen only through quantised encoder ticks. PPO outputs knife-tip moves; differential IK on the TCP gives joint targets. |
| Training | `train_cut_ppo.py`, `run_train.sh` | rsl_rl PPO; logs and checkpoints in `logs/dough_cut/<run>/`, exported policy in `.../exported/policy.pt` (+ `env_cfg.json`). |
| Line interfaces | `encoder.py`, `scheduler.py` | Encoder ticks (32-bit, wraps) → belt position/speed; tray messages → cut tasks; earliest-deadline-first across all trays in reach; unreachable cuts reported as missed. |
| Line run | `tray_runner.py` | 006 scene: generator trays back to back, photoeye latch, encoder, scheduler, transit by IK, cut by the policy, every cut scored against the true geometry; `runs/<name>/report.json` (+ `run.mp4` with `--video`). |
| Checks | `check_env.py` | Scripted follower (or `--policy`) through the same IK. |

## Commands (GPU box)

```bash
cd ~/IsaacLab
N=~/TH/Project-Manhattan
# run the line with the trained policy: trays back to back, belt 20 mm/s
./isaaclab.sh -p $N/tray_runner.py --headless --policy $N/logs/dough_cut/v1/exported/policy.pt \
    --trays 6 --belt_speed 0.02 [--min_pieces 1 --max_pieces 5] [--video] --out $N/runs/my-run
# train (about 36 min on the RTX 3090)
RUN=v1b ITERS=1500 $N/run_train.sh
# train with tuned settings (any name in cut_env.TUNABLE; saved to exported/env_cfg.json, used by the runner)
./isaaclab.sh -p $N/train_cut_ppo.py --headless --run_name v1-tuned --env w_success=20
# is the task feasible? / how good is a policy?
./isaaclab.sh -p $N/check_env.py --headless --num_envs 32 --episodes 64 [--policy <run>/exported/policy.pt]
# rebuild 006 / the library / the cut plan
~/_isaac_sim/python.sh $N/build_006.py --force
~/_isaac_sim/python.sh $N/build_path_library.py --trays 900
~/_isaac_sim/python.sh $N/plan_cuts.py --seed 3
```

Paths outside this folder (K2 scenes, DoughSDG generator, robot asset) are in `paths.py`
(override with `K2_DIR`, `DOUGHSDG_DIR`, `K2_ROBOT_USD`).

## Watch the cuts live (GPU box desktop)

```bash
./isaaclab.sh -p ~/TH/Project-Manhattan/tray_runner.py \
    --policy ~/TH/Project-Manhattan/logs/dough_cut/v1/exported/policy.pt --trays 3 --belt_speed 0.02 --hold
```

No `--headless`: the Isaac Sim window opens on the overview camera (`/World/RunCam`). Yellow = planned cut
(drawn on the surface; the cut is 3 mm inside), red = where the knife tip actually went (also lifted to
the surface). `--hold` keeps the window open after the run.

## Hooking up the real line

- **Encoder:** implement `read() -> EncoderReading(ticks, t)` for the PLC/OPC UA counter (same as
  `SimBeltEncoder`); set `counts_per_m` to the real encoder. Training used 10,000 counts/m.
- **Tray message** (one per tray, at the photoeye), from AI_CORE detections or the PLC:

```json
{"tray_id": "T0007", "latch_ticks": 123456789, "latch_x": -0.45,
 "pieces": [{"id": "p0", "kind": "long", "contour": [[x, y, z], ...], "direction": [dx, dy, 0]}]}
```

  `latch_x` is the tray origin's belt position when `latch_ticks` was latched; `contour` is the top
  surface along the cut line in tray coordinates (x along the belt, y across, z height in the
  robot/world frame). The scheduler builds the knife path (approach, cut 3 mm under the contour,
  retract). A ready-made `cut` polyline plus `approach`/`retract` is also accepted.

## Known limits

- Trained for belts up to 30 mm/s, one cut speed (50 mm/s) and one depth (3 mm).
- Dough is not simulated as a material: a cut is a geometric path at depth, scored against the mesh.
- The knife has no collision model; robot-to-conveyor clearance is not checked.
- The policy never settles exactly on the retract point (ending the episode stopped its reward in
  training); the runner ends a cut once the reference reaches retract and the knife is within 1.5 cm.
- The shipped `path_library.npz` was built with the earlier surface estimate (highest vertex within 4 mm),
  which dips to the underside where the mesh is sparse; rebuilding it uses the exact surface.

## Results — policy v1 (`logs/dough_cut/v1`, 1,500 iterations, 1,024 envs, ~36 min)

Training (random library cuts): 98.1% of cut steps in tolerance; 97.5% of cuts ≥90% in tolerance.

Line runs in 006 (6 trays back to back, 16 generator pieces, seed 11, encoder 10,000 counts/m, photoeye latch):

| Belt | Cut in tolerance | Missed | Run folder |
|---|---|---|---|
| 20 mm/s | 16/16 | 0 | `runs/v1-belt0.02` |
| 30 mm/s | 14/16 | 0 | `runs/v1-belt0.03` (2 long pieces wandered sideways: 59%, 83%) |
| 20 mm/s, first scheduler (window 0.40–0.95 m) | 12/16 | 4 | `runs/v1-belt20-video` (`run.mp4`) |

A cut passes when ≥90% of its cut steps keep depth 3 mm ±1 mm, sideways ±2 mm and blade tilt <5°.
# Project-Manhattan
