#!/bin/bash
# Train the dough-scoring policy (logs in logs/dough_cut/<run>).
N=/home/zadmin/TH/Project-Manhattan
cd ~/IsaacLab && PYTHONUNBUFFERED=1 ./isaaclab.sh -p $N/train_cut_ppo.py --headless --num_envs ${NUM_ENVS:-1024} --max_iterations ${ITERS:-1500} --run_name ${RUN:-v1} > $N/logs/train-${RUN:-v1}.log 2>&1
