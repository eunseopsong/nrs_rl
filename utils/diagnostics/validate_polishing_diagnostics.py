# SPDX-License-Identifier: BSD-3-Clause
"""Bounded Isaac test: real path termination -> legacy PNGs, without policy changes."""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

import argparse
import json
from pathlib import Path
import sys
import tempfile
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not 1 <= args.num_envs <= 2:
    parser.error("Use one or two environments for this bounded diagnostics test.")
simulation_app = AppLauncher(args).app

import gymnasium as gym
import h5py
import numpy as np
import torch
from PIL import Image
from isaaclab_tasks.utils import parse_env_cfg
import nrs_rl.tasks  # noqa: F401
from nrs_rl.tasks.manager_based.nrs_rl.utils import visualization as vis


def validate():
    output = Path(tempfile.mkdtemp(prefix="nrs_polishing_diagnostics_"))
    cfg = parse_env_cfg("Template-Nrs-Rl-v0", device=args.device, num_envs=args.num_envs)
    cfg.seed = 42
    # Use a 30 mm prefix of the actual trajectory to exercise natural completion
    # in ~1125 steps. Only this diagnostic invocation uses the shortened file.
    trajectory = output / "diagnostic_flat.h5"
    with h5py.File(cfg.actions.arm_action.integration.hdf5_file_path, "r") as original:
        with h5py.File(trajectory, "w") as shortened:
            for name in ("position", "force"):
                shortened.create_dataset(name, data=original[name][:751])
    cfg.actions.arm_action.integration.hdf5_file_path = str(trajectory)
    cfg.events.load_hdf5_trajectory.params["file_path"] = str(trajectory)
    cfg.actions.arm_action.integration.enable_debug_print = True
    cfg.actions.arm_action.integration.debug_print_interval = 50
    cfg.visualization.enable_visualizer = True
    cfg.visualization.save_interval_episodes = 1
    cfg.sim.physx.gpu_max_rigid_contact_count = 2**18
    cfg.sim.physx.gpu_max_rigid_patch_count = 2**14
    cfg.sim.physx.gpu_found_lost_pairs_capacity = 2**18
    cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**18
    cfg.sim.physx.gpu_collision_stack_size = 2**24
    vis.RUN_LOG_DIR = output / "plots"
    vis.REWARD_LOG_DIR = vis.RUN_LOG_DIR / "reward_logs"
    terminal = {}

    def finalize_checked(raw, env_ids):
        if (env_ids == 0).any():
            term = raw.action_manager.get_term("arm_action")
            if term.path_done[0]:
                terminal["removal"] = float(term.cumulative_removal[0])
                terminal["samples"] = int(raw.episode_length_buf[0])
        vis.on_episode_reset(raw, env_ids)

    cfg.events.finalize_visualization_episode.func = finalize_checked
    print(f"Diagnostics artifact directory: {output}", flush=True)
    env = gym.make("Template-Nrs-Rl-v0", cfg=cfg)
    try:
        with torch.inference_mode():
            env.reset(seed=42)
            term = env.unwrapped.action_manager.get_term("arm_action")
            messages = []
            original_write = term._debug_printer.write

            def capture(message):
                messages.append(message)
                original_write(message)

            term._debug_printer.write = capture
            action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            reward_sum = 0.
            for step in range(1800):
                obs, reward, done, truncated, _ = env.step(action)
                assert set(obs) == {"policy"} and obs["policy"].shape[-1] == 14
                assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
                reward_sum += float(reward[0])
                if done[0]:
                    break
                assert not truncated.any()
            else:
                raise AssertionError("No natural path completion within 1800 steps")
            assert terminal["samples"] == step + 1
            assert vis._summary_metrics["samples"] == [step + 1], "terminal sample omitted"
            assert np.isclose(vis._summary_metrics["total_removal"][0], terminal["removal"], rtol=2e-5)
            assert np.isclose(vis._summary_metrics["episode_reward"][0], reward_sum, rtol=2e-5)
            assert vis._summary_metrics["contact_samples"][0] > 50
            assert 5 < vis._summary_metrics["mean_normal_force"][0] < 15
            assert len(messages) == len(range(0, step + 1, 50))
            assert "[Polishing Live] ep1 step=0 env=0" in messages[0]
            assert "normal_force_N=nan" in messages[0], "calibration must not claim valid force"
            assert "rewards          =" in messages[-1]
            files = sorted((vis.RUN_LOG_DIR / "ep1").glob("*.png"))
            assert len(files) == 7, [p.name for p in files]
            for path in files + [vis.REWARD_LOG_DIR / "00_reward_components.png"]:
                with Image.open(path) as png:
                    png.verify()
            assert (vis.RUN_LOG_DIR / "00_episode_summary.csv").exists()
            assert (vis.RUN_LOG_DIR / "ep1/00_summary.txt").exists()
            assert not vis._rl_time_buffer, "episode samples leaked across reset"
            env.step(action)
            assert "[Polishing Live] ep2 step=0 env=0" in messages[-1]
            assert len(vis._rl_time_buffer) == 1
            assert not (vis.RUN_LOG_DIR / "ep2").exists(), "saved before episode end"
        print(json.dumps({
            "status": "PASS", "steps_to_natural_completion": step + 1,
            "policy_observation_dim": 14, "action_dim": term.action_dim,
            "episode_pngs": [p.name for p in files], "artifacts": str(output),
            "episode_removal_proxy_n_mm": terminal["removal"],
            "episode_reward": reward_sum,
            "note": "Constant residual, shortened test path; not policy-quality evidence.",
        }, indent=2), flush=True)
    finally:
        env.close()


exit_code = 0
try:
    validate()
except Exception:
    traceback.print_exc()
    exit_code = 1
finally:
    sys.stdout.flush()
    sys.stderr.flush()
    simulation_app.app.post_quit(exit_code)
    simulation_app.close()
sys.exit(exit_code)
