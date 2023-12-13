#!/bin/bash
export ROS_HOSTNAME=knight
export ROS_MASTER_URI=http://sean1:11311

source ~/sim_ws/devel/setup.bash

if [ -f /.dockerenv ]; then
    echo "I'm running in a docker container (this is required)!";
else
    echo "Run this script in a docker container.";
		exit 1
fi

if [[ $# -eq 0 ]] ; then
    echo "Usage: $0 <experiment_name>"
    exit 1
fi

echo "Found rosmaster with topics:"
rosnode info /rosout || exit 1

./src/main.py --mode rollout --model controller --experiment $1 --ats-slop 1.0