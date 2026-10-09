"""Frozen fixed-force PPO process contract, separate from legacy profile optimization."""
from ..paths import REFERENCE
from ..utils.removal_metrics import sha

CONTRACT = {'action_mapping': ['6 + 6*a0 mm/s'],
 'action_names': ['feed'],
 'actor_kind': 'neural_ppo',
 'backend': 'ppo_fixed_force_spatial_v1',
 'control_period_s': 0.008,
 'deployment_compatible_with_variable_force_runtime': False,
 'depth_units': 'h/K, uncalibrated',
 'dwell': 'continuous residence modulation by forward feed, no periodic pauses or reversals',
 'entry_reference_footprint_mm': 18.0,
 'entry_seconds': 2.0,
 'feed_action_slew_mm_s2': 18.0,
 'force_bounds_n': [20.0, 20.0],
 'forward_speed_bounds_mm_s': [0.0, 12.0],
 'hard_force_stop_n': 100.0,
 'local_return': False,
 'minimum_mode6_volume_ratio': 0.5,
 'moving_speed_bounds_mm_s': [1.5, 12.0],
 'network': '16 observations plus 16 Fourier progress features; actor and critic each 128x128 '
            'tanh',
 'observation_names': ['progress',
                       'force_over_20',
                       'speed_over_6',
                       'centered_nominal_depth_deficit',
                       'centered_predicted_final_deficit',
                       'behind_minus_current_deficit',
                       'local_residence_over_5s',
                       'tracking_error_over_10mm',
                       'elapsed_over_nominal',
                       'reverse_budget_fraction',
                       'contact',
                       'shield',
                       'target_force_over_20',
                       'rate_error',
                       'return_allowed',
                       'remaining_nominal_depth_over_target'],
 'policy_backend': 'pytorch_ppo',
 'policy_controls_force': False,
 'policy_period_s': 0.08,
 'profile_interpolation': None,
 'requires_independent_isaac_validation': True,
 'rotation_applied_to': 'removal estimator only; URDF dynamics unchanged',
 'rotation_contribution': True,
 'rpm_is_assumed_not_measured': True,
 'schema_version': 1,
 'second_complete_pass': False,
 'source_path_sha256': '9f4397cea47faa4cb89c5ad4d81afbcb3860988839adf4cc52fcb284449c046e',
 'speed_acceleration_limit_mm_s2': 16.0,
 'speed_jerk_limit_mm_s3': 160.0,
 'speed_mapping': {'center_mm_s': 6.0, 'scale_mm_s': 6.0},
 'spindle_rpm': 1000.0,
 'static_dwell_removal': 'p * omega * radius * dt',
 'target_force_n': 20.0,
 'teacher_profile_used': False,
 'tracking_compensation': {'gain': 0.6, 'limit_mm': 4.0, 'tau_s': 0.08},
 'training_environment': 'reference-path rotary-removal and digital-command-response simulator; '
                         'not Isaac physics',
 'training_method': 'PPO clipped stochastic policy gradient with GAE and learned value function',
 'velocity_model': 'rotating_disk',
 'volume_constraint_scopes': ['processing_roi', 'full_roi', 'whole_grid']}
CONTRACT['source_path_sha256'] = sha(REFERENCE)

def check_ppo_or_baseline(contract):
    if contract.get('backend') != CONTRACT['backend']:
        from nrs_rl.tasks.model_based.policies.fixed_force_refinement import check_contract as check_baseline_contract
        return check_baseline_contract(contract)
    for key, value in CONTRACT.items():
        if contract.get(key) != value:
            raise ValueError(f'PPO contract mismatch: {key}')
    return 1.5, 12.

