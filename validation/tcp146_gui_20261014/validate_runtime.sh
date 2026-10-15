#!/bin/bash
set -e
source /opt/ros/humble/setup.bash
source /home/eunseop/dev_ws/install/setup.bash
export ROS_DOMAIN_ID=229
export ROS_LOCALHOST_ONLY=1
export PYTHONDONTWRITEBYTECODE=1
/usr/bin/python3 -B /home/eunseop/dev_ws/src/y2_ur10skku_control/Y2RobMotion/test/validate_fixed_force_launch.py /home/eunseop/dev_ws/install/Y2RobMotion
/usr/bin/python3 -B /home/eunseop/dev_ws/src/y2_ur10skku_control/Y2RobMotion/test/test_preston_d14_preflight.py
/usr/bin/python3 -B /home/eunseop/dev_ws/src/y2_ur10skku_control/Y2RobMotion/test/validate_final_ppo_ros.py --mode 5 --output /home/eunseop/nrs_rl/logs/experiments/ppo_tcp146_gui_20261014/runtime_validation
/usr/bin/python3 -B /home/eunseop/dev_ws/src/y2_ur10skku_control/Y2RobMotion/test/validate_final_ppo_ros.py --mode 6 --output /home/eunseop/nrs_rl/logs/experiments/ppo_tcp146_gui_20261014/runtime_validation
