# CPSC 583 Final Project

This repo contains project code for CPSC 483/583: Deep Learning on Graph-Structured Data at Yale University. This project explores Spatial-Temporal Graph Neural Networks to infer robot performance in a navigation task.

Note: The environment for this project is installed at the lab computer of Yale Interactive Machines Group. If the environment cannot be installed locally, please directly check the model implementations within `/src/module` folder.

## Setup

- Clone repository within your `~/sim_ws/src/` folder.
- install nvidia-docker if not installed by running: https://gist.github.com/nathantsoi/e668e83f8cadfa0b87b67d18cc965bd3?permalink_comment_id=3466326#gistcomment-3466326

```
#one line install (sudo apt install curl first, if not already installed)
curl -L https://gist.githubusercontent.com/nathantsoi/e668e83f8cadfa0b87b67d18cc965bd3/raw/setup_docker.sh | sudo bash
```

- Build the container `./container build`

## Usage

- Run all the commands in the container by first opening a `./container shell`

### Dataset

The dataset is available at this link: https://drive.google.com/file/d/15m5cfM_g8d2qT_0_BeCThe7Axzi4CCRH/view?usp=drive_link

### Model Training

e.g. to train node-level ST-MPGNN model, run the following commands:

```
./src/main.py --mode train --model simple --human-annotation-split --module stgnn_mpgnn --data-dir /data/test_gnn/sean-vr/datasets/by-participant-720_cm-16_agents-40_ts-hz_5-2023-08-23/ --timesteps 40 --max-epochs 700 --loss bce --label 0 --dataset simple --loader-num-workers 0 --features 'gaze,goal' --gnn-node-features 64 --gnn-edge-features 2 --gnn-sizes "64" --ffnet-sizes "1024,512,256,64" --batch-size 8 --experiment 404-mpgnn-graph
```
