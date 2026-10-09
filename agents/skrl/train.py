# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))


"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default=None,
    help=(
        "Name of the RL agent configuration entry point. Defaults to None, in which case the argument "
        "--algorithm is used to determine the default agent configuration entry point."
    ),
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument("--policy-action-repeat", type=int, default=None,
                    help="Hold each PPO decision for this many 125 Hz control ticks (default: 1)")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import logging
import os
import random
import time
from datetime import datetime

import gymnasium as gym
import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# import logger
logger = logging.getLogger(__name__)

import nrs_rl.tasks  # noqa: F401

if args_cli.ml_framework.startswith("torch"):
    import torch
    # Mode-3/IK uses many small CPU tensor operations, even with GPU physics.
    # A large BLAS pool costs more than the tiny operations themselves.
    torch.set_num_threads(1)

# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with skrl agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    policy_repeat = args_cli.policy_action_repeat
    if policy_repeat is None:
        policy_repeat = 1
    if policy_repeat < 1:
        raise ValueError("policy-action-repeat must be positive")
    if args_cli.task == "Template-Nrs-Rl-v0" and algorithm == "ppo":
        from nrs_rl.tasks.manager_based.nrs_rl.utils.control_action_repeat import ppo_config_for_action_repeat
        agent_cfg = ppo_config_for_action_repeat(agent_cfg, policy_repeat)

    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training config
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    # max iterations for training
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
    # The Ray Tune workflow extracts experiment name using the logging line below, hence,
    # do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f"_{agent_cfg['agent']['experiment']['experiment_name']}"
    # set directory into agent config
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    if args_cli.task == "Template-Nrs-Rl-v0":
        import json
        from nrs_rl.tasks.manager_based.nrs_rl.utils.velocity_policy import (
            OBSERVATION_FIELDS, OBSERVATION_SIZE, OBSERVATION_VERSION,
        )
        integration = env_cfg.actions.arm_action.integration
        contract = {
            "schema_version": OBSERVATION_VERSION, "observation_size": OBSERVATION_SIZE,
            "observation_fields": OBSERVATION_FIELDS, "control_period_s": env_cfg.sim.dt * env_cfg.decimation,
            "robot": "ur10_cb3", "nominal_speed_mm_s": integration.nominal_speed_mm_s,
            "objective": "preston_force_tcp_sliding_rate",
            "policy_backend": "skrl_shared_ppo",
            "target_normal_force_n": integration.target_normal_force_n,
            "tool_diameter_mm": integration.tool_diameter_mm,
            "physics_tool_diameter_mm": integration.tool_diameter_mm,
            "spindle_rpm": integration.spindle_rpm,
            "target_mrr_n_mm_s": integration.target_mrr_n_mm_s,
            "residual_speed_fraction": integration.residual_speed_fraction,
            "force_rate_compensation": integration.force_rate_compensation,
            "policy_signal_tau_s": integration.policy_signal_tau_s,
            "turn_preview_mm": integration.turn_preview_mm,
            "action_filter_tau_s": integration.action_filter_tau_s,
            "action_slew_per_s": integration.action_slew_per_s,
            "max_speed_acceleration_mm_s2": integration.max_speed_acceleration_mm_s2,
            "max_speed_jerk_mm_s3": integration.max_speed_jerk_mm_s3,
            "speed_limiter_type": integration.speed_limiter_type,
            "policy_action_repeat": policy_repeat,
            "policy_period_s": env_cfg.sim.dt * env_cfg.decimation * policy_repeat,
        }
        with open(os.path.join(log_dir, "params", "velocity_policy.json"), "w") as stream:
            json.dump(contract, stream, indent=2)
        if args_cli.checkpoint:
            from nrs_rl.tasks.manager_based.nrs_rl.runtime.ppo_policy_node import load_policy_contract, SCHEDULER_CONTRACT
            previous = load_policy_contract(retrieve_file_path(args_cli.checkpoint))
            for name in ("policy_backend", "schema_version", "observation_size", "control_period_s", "physics_tool_diameter_mm",
                         "policy_action_repeat", "policy_period_s", *SCHEDULER_CONTRACT):
                if previous.get(name) != contract.get(name):
                    raise ValueError(
                        f"Cannot resume with changed policy semantics: {name} "
                        f"was {previous.get(name)}, requested {contract.get(name)}. "
                        "Start a fresh policy or explicitly restore the saved environment contract."
                    )

    # get checkpoint path (to resume training)
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    repeated_env = None
    if policy_repeat > 1 or args_cli.task == "Template-Nrs-Rl-v0":
        from nrs_rl.tasks.manager_based.nrs_rl.utils.control_action_repeat import ControlActionRepeat
        env = repeated_env = ControlActionRepeat(env, policy_repeat, agent_cfg["agent"]["discount_factor"])

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    runner = Runner(env, agent_cfg)

    # load checkpoint (if specified)
    if resume_path:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)
    elif args_cli.task == "Template-Nrs-Rl-v0":
        # Zero mean starts at nominal feed speed (or the explicitly enabled
        # diagnostic prior). Keep hidden layers random to permit learning.
        import torch
        with torch.no_grad():
            actor = runner.agent.models["policy"].policy_layer
            actor.weight.zero_()
            actor.bias.zero_()

    # run training
    runner.run()

    if repeated_env is not None:
        import json
        coverage = repeated_env.coverage()
        coverage["wall_seconds"] = time.time() - start_time
        with open(os.path.join(log_dir, "training_coverage.json"), "w") as stream:
            json.dump(coverage, stream, indent=2)
        print("TRAINING_COVERAGE", json.dumps(coverage), flush=True)

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
