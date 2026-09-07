# SPDX-License-Identifier: BSD-3-Clause
"""Production-parity force control with a deliberately narrow RL interface.

The policy controls only a bounded residual on path speed.  Cartesian force
control and inverse kinematics are compiled from the same C++ sources used by
Y2RobMotion.  Removal is measured from realized TCP motion, never from the
scheduled command velocity.
"""

from __future__ import annotations

import importlib
import math
import os

import h5py
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from ..utils import debug as local_debug, visualization as local_vis
from ..utils.adaptive_velocity_debug import EpisodeDebugPrinter, arc_to_index, format_polishing_live

y2_cfg = importlib.import_module(
    "nrs_rl.tasks.manager_based.nrs_rl.y2_control_pybind.y2_control_py.config"
)
y2_pb = importlib.import_module(
    "nrs_rl.tasks.manager_based.nrs_rl.y2_control_pybind.y2_control_py._y2_control_pybind"
)
if not hasattr(y2_pb, "Mode3ForceController") or not hasattr(y2_pb, "RobotKinematics"):
    raise RuntimeError(
        "The y2_control_pybind extension is stale. Rebuild it with "
        "python setup.py build_ext --inplace in y2_control_pybind."
    )
local_ft_sensor = importlib.import_module(
    "nrs_rl.tasks.manager_based.nrs_rl.assets.assets.sensors.six_axis_ft_sensor"
)


def spatial_to_rotmat(spatial: torch.Tensor) -> torch.Tensor:
    angle = torch.linalg.norm(spatial, dim=-1, keepdim=True)
    axis = spatial / torch.clamp(angle, min=1.0e-10)
    x, y, z = axis.unbind(-1)
    theta = angle.squeeze(-1)
    c, s, one_c = torch.cos(theta), torch.sin(theta), 1.0 - torch.cos(theta)
    result = torch.empty((*spatial.shape[:-1], 3, 3), device=spatial.device, dtype=spatial.dtype)
    result[..., 0, 0] = c + x * x * one_c
    result[..., 0, 1] = x * y * one_c - z * s
    result[..., 0, 2] = x * z * one_c + y * s
    result[..., 1, 0] = y * x * one_c + z * s
    result[..., 1, 1] = c + y * y * one_c
    result[..., 1, 2] = y * z * one_c - x * s
    result[..., 2, 0] = z * x * one_c - y * s
    result[..., 2, 1] = z * y * one_c + x * s
    result[..., 2, 2] = c + z * z * one_c
    small = angle.squeeze(-1) < 1.0e-10
    if torch.any(small):
        result[small] = torch.eye(3, device=spatial.device, dtype=spatial.dtype)
    return result


def rotmat_to_spatial(rotation: torch.Tensor) -> torch.Tensor:
    cos_angle = torch.clamp(
        (rotation[..., 0, 0] + rotation[..., 1, 1] + rotation[..., 2, 2] - 1.0) * 0.5,
        -1.0,
        1.0,
    )
    angle = torch.acos(cos_angle)
    vector = torch.stack(
        (
            rotation[..., 2, 1] - rotation[..., 1, 2],
            rotation[..., 0, 2] - rotation[..., 2, 0],
            rotation[..., 1, 0] - rotation[..., 0, 1],
        ),
        dim=-1,
    )
    scale = angle / torch.clamp(2.0 * torch.sin(angle), min=1.0e-7)
    output = vector * scale.unsqueeze(-1)
    output[angle < 1.0e-6] = 0.0
    return output


@configclass
class ActionIntegrationCfg:
    body_name: str = "spindle_link"
    fixed_joint_name: str = "tool0_to_spindle"
    joint_prim_relpath: str = "joints"
    hdf5_file_path: str = ""
    position_dataset_key: str = "position"
    force_dataset_key: str = "force"
    action_dim: int = 1

    nominal_speed_mm_s: float = 6.0
    residual_speed_fraction: float = 0.67
    min_speed_mm_s: float = 1.0
    max_speed_mm_s: float = 12.0
    action_filter_tau_s: float = 0.08
    action_slew_per_s: float = 4.0

    force_overload_ratio: float = 1.6
    tracking_stop_mm: float = 10.0
    contact_force_n: float = 1.5
    projection_window: int = 200
    surface_bins: int = 256
    approach_duration_s: float = 2.0
    # Model the gravity feed-forward of the robot's inner position servo.
    # Gravity remains enabled in PhysX and the FT sensor still sees tool weight.
    joint_gravity_compensation: bool = True

    # Sim-to-real randomization in measurement/latency channels.  These are
    # redrawn independently per environment at reset.
    force_scale_range: tuple[float, float] = (0.90, 1.10)
    force_bias_range_n: tuple[float, float] = (-0.75, 0.75)
    max_action_delay_steps: int = 3
    # One selected environment only; interval is in control steps (8 ms each).
    enable_debug_print: bool = True
    debug_print_interval: int = 50
    debug_env_id: int = 0


@configclass
class AdmittanceControlActionCfg(ActionTermCfg):
    class_type: type | None = None
    asset_name: str = "robot"
    integration: ActionIntegrationCfg = ActionIntegrationCfg()


class AdmittanceControlAction(ActionTerm):
    cfg: AdmittanceControlActionCfg

    def __init__(self, cfg: AdmittanceControlActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.cfg = cfg
        self.int_cfg = cfg.integration
        self.robot = env.scene[cfg.asset_name]
        self._num_envs_local = env.num_envs
        self._step_dt_local = float(env.step_dt)
        self._debug_printer = None
        if self.int_cfg.enable_debug_print:
            if not 0 <= self.int_cfg.debug_env_id < env.num_envs:
                raise ValueError(f"debug_env_id must be in [0, {env.num_envs - 1}]")
            self._debug_printer = EpisodeDebugPrinter(
                self.int_cfg.debug_env_id, self.int_cfg.debug_print_interval
            )
        if abs(self._step_dt_local - float(y2_cfg.CONTROL_PERIOD)) > 1.0e-9:
            raise RuntimeError(
                f"Isaac step_dt={self._step_dt_local} differs from production "
                f"CONTROL_PERIOD={y2_cfg.CONTROL_PERIOD}"
            )

        body_ids = self.robot.find_bodies(self.int_cfg.body_name)[0]
        if not body_ids:
            raise ValueError(f"body '{self.int_cfg.body_name}' was not found")
        self.ee_idx = int(body_ids[0])

        self._raw_actions = torch.zeros((env.num_envs, 1), device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        delay_slots = max(1, int(self.int_cfg.max_action_delay_steps) + 1)
        self._action_history = torch.zeros((env.num_envs, delay_slots), device=self.device)
        self._action_delay = torch.zeros(env.num_envs, dtype=torch.long, device=self.device)

        self.traj_positions, self.traj_forces = self._load_trajectory()
        self.traj_length = int(self.traj_positions.shape[0])
        lengths = torch.linalg.norm(
            self.traj_positions[1:, :3] - self.traj_positions[:-1, :3], dim=-1
        )
        if lengths.numel() == 0 or float(torch.sum(lengths)) <= 0.0:
            raise ValueError("trajectory must contain at least two distinct positions")
        self.segment_lengths_mm = torch.clamp(lengths, min=1.0e-6)
        self.arc_mm = torch.cat(
            (torch.zeros(1, device=self.device), torch.cumsum(self.segment_lengths_mm, dim=0))
        )
        self.path_length_mm = float(self.arc_mm[-1])
        self._diagnostic_arc_mm = self.arc_mm.detach().cpu().tolist()

        n, bins = env.num_envs, int(self.int_cfg.surface_bins)
        self.path_cursor_mm = torch.zeros(n, device=self.device)
        self.path_cursor = self.path_cursor_mm  # compatibility for diagnostics
        self.path_index = torch.zeros(n, dtype=torch.long, device=self.device)
        self.current_target_index = torch.zeros_like(self.path_index)
        self.physical_path_index = torch.zeros_like(self.path_index)
        self.path_done = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.current_index_delta = torch.zeros(n, device=self.device)
        self.commanded_speed_mm_s = torch.zeros(n, device=self.device)
        self.current_sliding_velocity_mm_s = torch.zeros(n, device=self.device)
        self.current_abs_fz = torch.zeros(n, device=self.device)
        self.force_error_n = torch.zeros(n, device=self.device)
        self.force_derivative_n_s = torch.zeros(n, device=self.device)
        self.current_path_tracking_error_mm = torch.zeros(n, device=self.device)
        self.prev_mrr_n_mm_s = torch.zeros(n, device=self.device)
        self.current_mrr_n_mm_s = torch.zeros(n, device=self.device)
        self.current_mrr_delta_n_mm_s = torch.zeros(n, device=self.device)
        self.realized_removal_step = torch.zeros(n, device=self.device)
        self.cumulative_removal = torch.zeros(n, device=self.device)
        self.surface_removal_by_index = torch.zeros((n, bins), device=self.device)
        self.surface_visit_counts = torch.zeros((n, bins), device=self.device)
        self.surface_last_index = torch.zeros(n, dtype=torch.long, device=self.device)
        self.safety_shield_active = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.policy_state = torch.zeros((n, 12), device=self.device)
        self.polishing_active = torch.zeros(n, dtype=torch.bool, device=self.device)

        self._filtered_action = torch.zeros(n, device=self.device)
        self._previous_force = torch.zeros(n, device=self.device)
        self._previous_tcp_mm = torch.zeros((n, 3), device=self.device)
        self._previous_tcp_valid = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._force_scale = torch.ones(n, device=self.device)
        self._force_bias = torch.zeros(n, device=self.device)
        self._approach_step = torch.zeros(n, dtype=torch.long, device=self.device)
        self._calibration_step = torch.zeros(n, dtype=torch.long, device=self.device)
        self._calibration_steps = (
            local_ft_sensor.FT_BIAS_WARMUP_SAMPLES + local_ft_sensor.FT_BIAS_INIT_SAMPLES
            if local_ft_sensor.FT_USE_BIAS else 0
        )
        self._approach_steps = max(1, round(self.int_cfg.approach_duration_s / self._step_dt_local))
        self._approach_start = torch.zeros((n, 6), device=self.device)
        self._previous_q_command = torch.zeros((n, 6), device=self.device)
        self._previous_q_valid = torch.zeros(n, dtype=torch.bool, device=self.device)

        self.kinematics = [
            y2_pb.RobotKinematics(
                robot_model=y2_cfg.ROBOT_KINEMATICS,
                dt=y2_cfg.CONTROL_PERIOD,
                ee2tcp=y2_cfg.EE2TCP,
            )
            for _ in range(n)
        ]
        self.force_controllers = [
            y2_pb.Mode3ForceController(
                y2_cfg.NAF_MDGRADI_CKPT,
                y2_cfg.CONTROL_PERIOD,
                y2_cfg.FORCE_CON_COORDINATE,
                y2_cfg.FORCE_SWITCH_DESIRED_FORCE_THRESHOLD,
                y2_cfg.FORCE_SWITCH_ACTUAL_FORCE_THRESHOLD,
                y2_cfg.FORCE_SWITCH_PRECONTACT_FORCE_HOLD,
                y2_cfg.FORCE_SWITCH_RETURN_TAU,
            )
            for _ in range(n)
        ]

    @property
    def action_dim(self):
        return 1

    @property
    def raw_actions(self):
        return self._raw_actions

    @property
    def processed_actions(self):
        return self._processed_actions

    def _load_trajectory(self) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.int_cfg.hdf5_file_path
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as stream:
            position = stream[self.int_cfg.position_dataset_key][:, :6]
            force = stream[self.int_cfg.force_dataset_key][:, :3]
        if position.shape[0] != force.shape[0]:
            raise ValueError("position and force trajectories have different lengths")
        return (
            torch.as_tensor(position, device=self.device, dtype=torch.float32),
            torch.as_tensor(force, device=self.device, dtype=torch.float32),
        )

    def _fk(self, env_id: int, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        transform = torch.tensor(
            self.kinematics[env_id].forward_kinematics(q.detach().cpu().double().tolist()),
            device=self.device,
            dtype=torch.float32,
        )
        rotation = transform[:3, :3]
        pose = torch.cat((transform[:3, 3], rotmat_to_spatial(rotation.unsqueeze(0))[0]))
        return pose, transform[:3, 3], rotation

    def _trajectory_at(self, distance_mm: float) -> tuple[torch.Tensor, torch.Tensor, int]:
        distance = torch.tensor(distance_mm, device=self.device)
        upper = int(torch.searchsorted(self.arc_mm, distance, right=True).item())
        lower = min(max(0, upper - 1), self.traj_length - 2)
        fraction = (distance_mm - float(self.arc_mm[lower])) / float(self.segment_lengths_mm[lower])
        fraction = min(1.0, max(0.0, fraction))
        pose = torch.lerp(self.traj_positions[lower], self.traj_positions[lower + 1], fraction)
        force = torch.lerp(self.traj_forces[lower], self.traj_forces[lower + 1], fraction)
        return pose, force, lower

    def _nearest_physical_index(self, env_id: int, tcp_mm: torch.Tensor) -> int:
        center = int(self.physical_path_index[env_id])
        window = int(self.int_cfg.projection_window)
        begin = max(0, center - window)
        end = min(self.traj_length, center + window + 1)
        distance = torch.linalg.norm(self.traj_positions[begin:end, :3] - tcp_mm, dim=-1)
        return begin + int(torch.argmin(distance).item())

    def _command_ik(self, env_id: int, q_seed: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        transform = torch.eye(4, device=self.device, dtype=torch.float32)
        transform[:3, :3] = spatial_to_rotmat(pose[3:6].unsqueeze(0))[0]
        transform[:3, 3] = pose[:3]
        result = self.kinematics[env_id].solve_ik(
            q_seed.detach().cpu().double().tolist(), transform.detach().cpu().double().tolist()
        )
        return torch.tensor(result, device=self.device, dtype=torch.float32)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self._num_envs_local, device=self.device)
        printer = self._debug_printer
        if printer is not None and printer.env_id in env_ids.tolist():
            printer.reset()
        tensors_zero = (
            self._raw_actions, self._processed_actions, self._action_history,
            self.path_cursor_mm, self.path_index, self.current_target_index,
            self.physical_path_index, self.current_index_delta, self.commanded_speed_mm_s,
            self.current_sliding_velocity_mm_s, self.current_abs_fz, self.force_error_n,
            self.force_derivative_n_s, self.current_path_tracking_error_mm,
            self.prev_mrr_n_mm_s, self.current_mrr_n_mm_s,
            self.current_mrr_delta_n_mm_s, self.realized_removal_step,
            self.cumulative_removal, self.surface_removal_by_index,
            self.surface_visit_counts, self.surface_last_index,
            self.policy_state, self._filtered_action, self._previous_force,
            self._previous_tcp_mm, self._approach_step, self._calibration_step,
            self._previous_q_command,
        )
        for tensor in tensors_zero:
            tensor[env_ids] = 0
        self.path_done[env_ids] = False
        self.polishing_active[env_ids] = False
        self.safety_shield_active[env_ids] = False
        self._previous_tcp_valid[env_ids] = False
        self._previous_q_valid[env_ids] = False
        count = len(env_ids)
        low, high = self.int_cfg.force_scale_range
        self._force_scale[env_ids] = low + (high - low) * torch.rand(count, device=self.device)
        low, high = self.int_cfg.force_bias_range_n
        self._force_bias[env_ids] = low + (high - low) * torch.rand(count, device=self.device)
        self._action_delay[env_ids] = torch.randint(
            0, int(self.int_cfg.max_action_delay_steps) + 1, (count,), device=self.device
        )
        q = self.robot.data.joint_pos[:, :6]
        for env_id in env_ids.tolist():
            pose, _, _ = self._fk(env_id, q[env_id])
            self._approach_start[env_id] = pose
            self.force_controllers[env_id].reset(pose.detach().cpu().double().tolist())

    def _record_visualization(self, env_id, measured_pose, normal_force):
        # One-way diagnostics: use the controller's already measured TCP and
        # metrics, without extra FT reads, physics steps or policy observations.
        if env_id == 0 and local_vis._visualization_enabled(self._env):
            local_vis.record_control_step(self._env, self, measured_pose, normal_force)

    def _print_debug(
        self, env_id, phase, measured_pose, *, reference_pose=None,
        command_pose=None, wrench=None, desired_force=None, normal_force=None,
    ):
        printer = self._debug_printer
        if printer is None or env_id != printer.env_id or not printer.tick():
            return
        i = env_id
        if reference_pose is None:
            reference_pose, _, _ = self._fk(i, self.robot.data.default_joint_pos[i, :6])
            command_pose = reference_pose
            desired_force = torch.zeros(3, device=self.device)
        normal = spatial_to_rotmat(reference_pose[3:6].unsqueeze(0))[0, :, 2]
        normal_offset = float(torch.dot(command_pose[:3] - reference_pose[:3], normal))
        cursor_mm = float(self.path_cursor_mm[i])
        cursor = arc_to_index(cursor_mm, self._diagnostic_arc_mm)
        previous_cursor = arc_to_index(
            cursor_mm - float(self.current_index_delta[i]), self._diagnostic_arc_mm
        )
        printer.write(format_polishing_live(
            episode=printer.episode, step=printer.step - 1, env_id=i,
            current_index=int(self.path_index[i]), last_index=self.traj_length - 1,
            target_index=int(self.current_target_index[i]), cursor=cursor,
            current_pose=measured_pose.detach().cpu().tolist(),
            target_pose=reference_pose.detach().cpu().tolist(),
            command_pose=command_pose.detach().cpu().tolist(),
            target_force=abs(float(desired_force[2])),
            normal_force=float("nan") if normal_force is None else abs(normal_force),
            sliding_velocity=float(self.current_sliding_velocity_mm_s[i]),
            removal_rate=float(self.current_mrr_n_mm_s[i]),
            cumulative_removal=float(self.cumulative_removal[i]),
            fn_offset=normal_offset, action=float(self._processed_actions[i, 0]),
            index_rate=cursor - previous_cursor,
            path_error_xy=float(torch.linalg.norm(measured_pose[:2] - reference_pose[:2])),
            reward_debug=local_debug.format_reward_debug(self._env, i),
        ))

    @torch.no_grad()
    def process_actions(self, actions: torch.Tensor):
        self._raw_actions.copy_(torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0))
        self._action_history[:, 1:] = self._action_history[:, :-1].clone()
        self._action_history[:, 0] = torch.clamp(self._raw_actions[:, 0], -1.0, 1.0)
        delayed = self._action_history.gather(1, self._action_delay[:, None]).squeeze(1)
        alpha = 1.0 - math.exp(-self._step_dt_local / max(self.int_cfg.action_filter_tau_s, 1.0e-6))
        target = self._filtered_action + alpha * (delayed - self._filtered_action)
        max_delta = self.int_cfg.action_slew_per_s * self._step_dt_local
        self._filtered_action += torch.clamp(target - self._filtered_action, -max_delta, max_delta)
        self._processed_actions[:, 0] = self._filtered_action

    @torch.no_grad()
    def apply_actions(self):
        q_all = self.robot.data.joint_pos
        q = q_all[:, :6]
        wrench = local_ft_sensor.get_6axis_ft_fixed_joint(
            env=self._env, asset_name=self.cfg.asset_name,
            fixed_joint_name=self.int_cfg.fixed_joint_name,
            joint_prim_relpath=self.int_cfg.joint_prim_relpath, verbose=False,
        ).clone()
        wrench[:, :3] *= self._force_scale[:, None]
        wrench[:, 2] += self._force_bias
        q_command = q_all.clone()

        for env_id in range(self._num_envs_local):
            measured_pose, tcp_mm, rotation = self._fk(env_id, q[env_id])
            if int(self._calibration_step[env_id]) < self._calibration_steps:
                # Static FT zeroing requires a stationary joint target. Do not
                # start the approach or accumulate removal during calibration.
                q_command[env_id, :6] = self.robot.data.default_joint_pos[env_id, :6]
                self._calibration_step[env_id] += 1
                self._approach_start[env_id] = measured_pose
                self._record_visualization(env_id, measured_pose, 0.0)
                self._print_debug(env_id, "CALIBRATION", measured_pose)
                continue
            approach = int(self._approach_step[env_id]) < self._approach_steps
            self.polishing_active[env_id] = not approach
            normal = rotation[:, 2]
            measured_normal_force = float(torch.dot(normal, wrench[env_id, :3]))
            abs_force = abs(measured_normal_force)
            force_delta = (abs_force - float(self._previous_force[env_id])) / self._step_dt_local
            self.force_derivative_n_s[env_id] = force_delta
            self._previous_force[env_id] = abs_force
            self.current_abs_fz[env_id] = abs_force

            tangent_distance = 0.0
            if bool(self._previous_tcp_valid[env_id]):
                delta = tcp_mm - self._previous_tcp_mm[env_id]
                tangent = delta - normal * torch.dot(delta, normal)
                tangent_distance = min(float(torch.linalg.norm(tangent)), 5.0)
            self._previous_tcp_mm[env_id] = tcp_mm
            self._previous_tcp_valid[env_id] = True
            actual_speed = tangent_distance / self._step_dt_local
            self.current_sliding_velocity_mm_s[env_id] = actual_speed
            previous_mrr = float(self.current_mrr_n_mm_s[env_id])
            in_contact = not approach and abs_force >= self.int_cfg.contact_force_n
            actual_mrr = abs_force * actual_speed if in_contact else 0.0
            self.prev_mrr_n_mm_s[env_id] = previous_mrr
            self.current_mrr_n_mm_s[env_id] = actual_mrr
            self.current_mrr_delta_n_mm_s[env_id] = actual_mrr - previous_mrr
            removal = abs_force * tangent_distance if in_contact else 0.0
            self.realized_removal_step[env_id] = removal
            self.cumulative_removal[env_id] += removal

            physical_index = self._nearest_physical_index(env_id, tcp_mm)
            self.physical_path_index[env_id] = physical_index
            physical_progress = float(self.arc_mm[physical_index]) / max(self.path_length_mm, 1.0e-6)
            bin_index = min(int(physical_progress * self.int_cfg.surface_bins), self.int_cfg.surface_bins - 1)
            if removal > 0.0:
                self.surface_removal_by_index[env_id, bin_index] += removal
                self.surface_visit_counts[env_id, bin_index] += 1.0
                self.surface_last_index[env_id] = max(int(self.surface_last_index[env_id]), bin_index)

            if approach:
                t = float(self._approach_step[env_id] + 1) / self._approach_steps
                t = t * t * (3.0 - 2.0 * t)
                reference_pose = torch.lerp(self._approach_start[env_id], self.traj_positions[0], t)
                desired_force = torch.zeros(3, device=self.device)
                # Keep the shared admittance history aligned while the robot
                # is travelling to the first polishing waypoint.
                self.force_controllers[env_id].reset(
                    measured_pose.detach().cpu().double().tolist()
                )
                self._approach_step[env_id] += 1
                command_pose = reference_pose
            else:
                reference_pose, desired_force, target_index = self._trajectory_at(
                    float(self.path_cursor_mm[env_id])
                )
                self.path_index[env_id] = target_index
                self.current_target_index[env_id] = target_index
                target_force = abs(float(desired_force[2]))
                tracking_error = float(torch.linalg.norm(tcp_mm - reference_pose[:3]))
                self.current_path_tracking_error_mm[env_id] = tracking_error
                self.force_error_n[env_id] = abs_force - target_force

                speed = self.int_cfg.nominal_speed_mm_s * (
                    1.0 + self.int_cfg.residual_speed_fraction * float(self._filtered_action[env_id])
                )
                speed = min(self.int_cfg.max_speed_mm_s, max(self.int_cfg.min_speed_mm_s, speed))
                overload = target_force > 0.0 and abs_force > self.int_cfg.force_overload_ratio * target_force
                tracking_stop = tracking_error > self.int_cfg.tracking_stop_mm
                shield = overload or tracking_stop
                self.safety_shield_active[env_id] = shield
                if shield:
                    speed = 0.0
                self.commanded_speed_mm_s[env_id] = speed
                distance_delta = speed * self._step_dt_local
                self.current_index_delta[env_id] = distance_delta
                self.path_cursor_mm[env_id] = min(
                    self.path_length_mm, float(self.path_cursor_mm[env_id]) + distance_delta
                )
                self.path_done[env_id] = (
                    float(self.path_cursor_mm[env_id]) >= self.path_length_mm - 1.0e-6
                )

                output = self.force_controllers[env_id].step(
                    measured_pose.detach().cpu().double().tolist(),
                    reference_pose.detach().cpu().double().tolist(),
                    desired_force.detach().cpu().double().tolist(),
                    wrench[env_id].detach().cpu().double().tolist(),
                    rotation.reshape(-1).detach().cpu().double().tolist(),
                )
                command_pose = torch.tensor(output[:6], device=self.device, dtype=torch.float32)

                visited = self.surface_removal_by_index[env_id, : max(1, bin_index + 1)]
                positive = visited[visited > 0.0]
                deficit = 0.0
                if positive.numel() > 1:
                    deficit = float((positive.mean() - visited[bin_index]) / positive.mean().clamp_min(1.0e-6))
                tangent_change = 0.0
                if 0 < target_index < self.traj_length - 2:
                    before = self.traj_positions[target_index, :3] - self.traj_positions[target_index - 1, :3]
                    after = self.traj_positions[target_index + 1, :3] - self.traj_positions[target_index, :3]
                    tangent_change = float(1.0 - torch.dot(before, after) /
                                           (torch.linalg.norm(before) * torch.linalg.norm(after)).clamp_min(1.0e-6))
                progress = float(self.path_cursor_mm[env_id]) / max(self.path_length_mm, 1.0e-6)
                values = (
                    float(self.force_error_n[env_id]) / max(target_force, 1.0),
                    abs_force / max(target_force, 1.0),
                    force_delta / 100.0,
                    actual_speed / max(self.int_cfg.max_speed_mm_s, 1.0),
                    tracking_error / max(self.int_cfg.tracking_stop_mm, 1.0),
                    float(self._filtered_action[env_id]),
                    float(self._raw_actions[env_id, 0] - self._filtered_action[env_id]),
                    progress,
                    deficit,
                    tangent_change,
                    1.0 if abs_force >= self.int_cfg.contact_force_n else 0.0,
                    1.0 if shield else 0.0,
                )
                self.policy_state[env_id] = torch.tensor(values, device=self.device)

            seed = self._previous_q_command[env_id] if self._previous_q_valid[env_id] else q[env_id]
            q_next = self._command_ik(env_id, seed, command_pose)
            self._previous_q_command[env_id] = q_next
            self._previous_q_valid[env_id] = True
            q_command[env_id, :6] = q_next
            self._record_visualization(env_id, measured_pose, measured_normal_force)
            # All values are the existing controller snapshot, before this
            # physics step. Logging never reads FT again or changes a target.
            self._print_debug(
                env_id, "APPROACH" if approach else "POLISH", measured_pose,
                reference_pose=reference_pose, command_pose=command_pose,
                wrench=wrench[env_id], desired_force=desired_force,
                normal_force=measured_normal_force,
            )

        self.robot.set_joint_position_target(q_command)
        if self.int_cfg.joint_gravity_compensation:
            self.robot.set_joint_effort_target(
                self.robot.root_physx_view.get_gravity_compensation_forces()
            )


AdmittanceControlActionCfg.class_type = AdmittanceControlAction
