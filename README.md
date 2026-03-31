# nrs_rl (v30)

Custom Isaac Lab task package for a UR10e + spindle polishing environment.

This repository is organized for:
- HDF5-based trajectory tracking
- End-effector pose observation through a custom FK Python binding
- 6-axis force/torque observation through a fixed-joint FT sensor utility
- Debug and visualization utilities separated from the MDP logic
- skrl-based training and play scripts

---

## How to Run

### 1) Activate environment
```bash
conda activate env_isaaclab
```

### 2) Training
```bash
~/IsaacLab/isaaclab.sh -p ~/nrs_rl/scripts/skrl/train.py --task Template-Nrs-Rl-v0
```

### 3) Play
Run this after training is finished.

```bash
~/IsaacLab/isaaclab.sh -p ~/nrs_rl/scripts/skrl/play.py --task Template-Nrs-Rl-v0
```

---

## How to Push into GitHub

```bash
cd ~/nrs_rl/source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl

git init
git add .
git commit -m "commit message"
```

Then push with GitKraken.

---

## Repository Structure

```text
nrs_rl/
├── agents/
├── assets/
│   └── assets/
│       ├── robots/
│       │   └── ur10e_w_spindle.py
│       └── sensors/
│           └── six_axis_ft_sensor.py
├── datasets/
├── mdp/
│   ├── action.py
│   ├── observation.py
│   ├── rewards.py
│   └── terminations.py
├── nrs_ik_py_bind/
├── utils/
│   ├── debug.py
│   └── visualization.py
└── nrs_rl_env_cfg.py
```

---

## Folder and File Roles

### `agents/`
Configuration files for the RL agent.
- `skrl_ppo_cfg.yaml`: PPO configuration used by the skrl training pipeline.

### `assets/`
Asset-related code for robot and sensor definitions.

#### `assets/assets/robots/`
Robot asset configuration.
- `ur10e_w_spindle.py`: defines the UR10e + spindle robot asset used in the task.

#### `assets/assets/sensors/`
Sensor-related utilities.
- `six_axis_ft_sensor.py`: contains the 6-axis force/torque sensor functions moved out of `mdp/observation.py`.  
  It reads FT information from the fixed joint and is used as the FT observation source.

### `datasets/`
Trajectory and dataset files used by the environment.
- `.h5` files: trajectory recordings used for tracking and observation targets.
- conversion scripts: utilities for converting recorded data into HDF5 format.

### `mdp/`
Core task logic used by the RL environment.

#### `mdp/action.py`
Contains the custom action term.
Main role:
- loads the HDF5 target trajectory
- performs trajectory-following control
- computes desired end-effector pose
- runs Jacobian-based IK
- sends joint targets to the robot
- prints action debug information through `utils/debug.py`

#### `mdp/observation.py`
Contains non-FT observation functions.
Main role:
- load HDF5 target positions
- return future target trajectory observations
- compute end-effector pose using the custom FK binding
- provide camera-related observation utilities

FT-related functions were removed from this file and moved to:
- `assets/assets/sensors/six_axis_ft_sensor.py`

#### `mdp/rewards.py`
Currently kept as a minimal placeholder module.
Reward functions that were no longer needed were removed.

#### `mdp/terminations.py`
Contains task termination logic.
Main role:
- terminate the episode when the trajectory is finished

### `nrs_ik_py_bind/`
Python bindings for C++ kinematics/control-side functions.

This folder exists to call C++-implemented functions from Python.
Main contents include:
- FK/IK solver source code in C++
- pybind bindings
- built shared libraries such as:
  - `nrs_fk_core.cpython-311-x86_64-linux-gnu.so`
  - `nrs_ik_core.cpython-311-x86_64-linux-gnu.so`

This is used by the Python task code to access fast kinematics computation implemented in C++.

### `utils/`
Utility code separated from the MDP files.

#### `utils/debug.py`
Debug-print helper functions.
Main role:
- centralize debug output
- combine action debug with FT sensor debug
- avoid scattering print logic across `action.py` and observation/sensor code

#### `utils/visualization.py`
Plot-saving and visualization helpers.
Main role:
- save episode plots
- keep plotting logic separated from the reward module

### `nrs_rl_env_cfg.py`
Top-level Isaac Lab environment configuration.
Main role:
- define scene configuration
- register action/observation/event/termination configuration
- connect MDP modules, robot asset config, and FT sensor functions

---

## Current Code Design (v30)

The current version is **v30**.

The code has been reorganized with the following design goals:
- keep MDP logic clean and focused
- separate FT sensor utilities from generic observation code
- separate debug logic from action/observation code
- separate visualization logic from reward code
- make the repository easier to maintain and hand off

In the current structure:
- trajectory-following is handled in `mdp/action.py`
- general observations are handled in `mdp/observation.py`
- 6-axis FT sensing is handled in `assets/assets/sensors/six_axis_ft_sensor.py`
- debug printing is handled in `utils/debug.py`
- plotting is handled in `utils/visualization.py`

---

## Execution Flow

1. `nrs_rl_env_cfg.py` builds the environment configuration.
2. The robot asset is loaded from `assets/assets/robots/ur10e_w_spindle.py`.
3. The action term in `mdp/action.py` loads the HDF5 trajectory and generates joint commands.
4. `mdp/observation.py` provides EE pose and trajectory target observations.
5. `assets/assets/sensors/six_axis_ft_sensor.py` provides 6-axis FT observations.
6. `utils/debug.py` formats and prints debug outputs.
7. skrl training/play scripts use this environment through Isaac Lab.

---

## Notes

- The FT sensor debug is currently integrated into the action debug output.
- The end-effector pose is computed through the custom FK binding from `nrs_ik_py_bind/`.
- The repository is structured for easier maintenance, debugging, and future handoff.
