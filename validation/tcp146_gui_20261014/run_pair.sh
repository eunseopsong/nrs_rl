#!/bin/bash
set -e
source /opt/ros/humble/setup.bash
source /home/eunseop/dev_ws/install/setup.bash
export ROS_DOMAIN_ID=181
export ROS_LOCALHOST_ONLY=1
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export PYTHONDONTWRITEBYTECODE=1
exec /usr/bin/python3 -B /home/eunseop/nrs_rl/logs/experiments/ppo_tcp146_gui_20261014/run_pair.py
