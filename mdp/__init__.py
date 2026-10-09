"""Lazy MDP exports: approximate removal training needs no Isaac process."""
from importlib import import_module

_EXPORTS = {'spatial_to_rotmat': 'action', 'rotmat_to_spatial': 'action', 'ActionIntegrationCfg': 'action', 'AdmittanceControlActionCfg': 'action', 'AdmittanceControlAction': 'action', 'get_hdf5_trajectory_length': 'observation', 'get_action_term': 'observation', 'adaptive_velocity_observation': 'observation', 'get_ee_idx': 'observation', 'get_current_pose_and_velocity': 'observation', 'get_current_fz': 'observation', 'get_path_sliding_metrics': 'observation', 'get_current_target_pose_for_reward': 'observation', 'get_surface_uniformity_reward_value': 'observation', 'get_ee_pose': 'observation', 'get_current_pose_and_force': 'observation', 'load_hdf5_trajectory': 'observation', 'load_hdf5_positions': 'observation', 'get_hdf5_target_positions': 'observation', 'get_hdf5_target_forces': 'observation', 'get_hdf5_target_pose_force': 'observation', 'get_joint_velocities': 'observation', 'get_velocity_adjusted_target_positions': 'observation', 'get_camera_distance': 'observation', 'get_camera_normals': 'observation', 'get_processed_polishing_target': 'observation', 'realized_removal_reward': 'rewards', 'removal_rate_tracking_penalty': 'rewards', 'force_tracking_reward': 'rewards', 'force_overshoot_penalty': 'rewards', 'spatial_uniformity_reward': 'rewards', 'removal_variation_penalty': 'rewards', 'action_rate_penalty': 'rewards', 'command_acceleration_penalty': 'rewards', 'command_jerk_penalty': 'rewards', 'safety_shield_penalty': 'rewards', 'completion_quality_reward': 'rewards', 'completion_rate_quality_reward': 'rewards', 'trajectory_finished': 'terminations', 'control_failed': 'terminations', 'polishing_timeout': 'terminations'}
__all__ = list(_EXPORTS)

def __getattr__(name):
    if name in _EXPORTS:
        value = getattr(import_module('.' + _EXPORTS[name], __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(name)
