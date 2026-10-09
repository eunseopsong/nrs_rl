"""Bounds, anti-chatter gates and configuration compatibility for joint search."""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

import copy
import unittest

from nrs_rl.tasks.manager_based.nrs_rl.model_based.training.train_tcp_control import assess
from nrs_rl.tasks.manager_based.nrs_rl.model_based.training.train_tcp_removal import SCHEDULER_CONTRACT, _validate_scheduler_contract
from nrs_rl.tasks.manager_based.nrs_rl.mdp.runtime.ppo_policy_node import _validate_tangential_config


class TcpControlTests(unittest.TestCase):
    def baseline(self):
        return dict(completed=True,processing_samples=14581,processing_rate_cv=.21,
            force_peak_n=29.,force_rmse_n=1.5,force_mean_n=20.,cross_track_rmse_mm=2.,
            endpoint_tangential_error_mm=7.5,measured_path_length_mm=640.,
            fault_fraction=0.,shield_fraction=0.,preston_surface=dict(
                roi_cells=1000,roi_area_mm2=250.,spatial_depth_cv=.52,
                roi_volume_over_k=12800.,tcp_full_footprint_volume_over_k=12830.,zero_depth_fraction=.1))

    def candidate(self):
        value=self.baseline();value.update(processing_rate_cv=.19,cross_track_rmse_mm=.6,
            endpoint_tangential_error_mm=.5,measured_path_length_mm=690.)
        value['preston_surface'].update(spatial_depth_cv=.4,roi_volume_over_k=13800.,
            tcp_full_footprint_volume_over_k=13830.,zero_depth_fraction=.01)
        return value

    def test_recovered_coverage_can_raise_volume_at_unchanged_force(self):
        result=assess(self.candidate(),[self.baseline()],700.,validation=True)
        self.assertTrue(result['passed'],result)
        self.assertGreater(result['comparisons'][0]['roi_volume_gain'],.05)

    def test_extra_chatter_force_missing_endpoint_and_time_are_rejected(self):
        baseline=self.baseline()
        for key,value in [('measured_path_length_mm',725.),('force_mean_n',22.),('force_peak_n',33.),
                          ('endpoint_tangential_error_mm',8.),('processing_samples',16000),
                          ('shield_fraction',.001),('cross_track_rmse_mm',2.1)]:
            candidate=self.candidate();candidate[key]=value
            self.assertFalse(assess(candidate,[baseline],700.,validation=True)['passed'],key)
        candidate=self.candidate();candidate['preston_surface']['tcp_full_footprint_volume_over_k']=14500.
        self.assertFalse(assess(candidate,[baseline],700.,validation=True)['passed'])

    def test_calibrated_checkpoint_requires_matching_controller_configuration(self):
        config={'damping_ns_m':4000.,'stiffness_n_m':8000.}
        contract={**SCHEDULER_CONTRACT,'tangential_controller':config}
        with self.assertRaisesRegex(ValueError,'matching tangential'):
            _validate_scheduler_contract(contract)
        with self.assertRaisesRegex(ValueError,'matching tangential'):
            _validate_scheduler_contract(contract,{**config,'stiffness_n_m':4000.})
        _validate_scheduler_contract(contract,config)
        _validate_scheduler_contract(SCHEDULER_CONTRACT)
        with self.assertRaisesRegex(ValueError,'matching tangential'):
            _validate_scheduler_contract(SCHEDULER_CONTRACT,config)

    def test_malformed_controller_config_is_rejected(self):
        for config in [None, {}, {'damping_ns_m': 4000., 'stiffness_n_m': float('nan')},
                       {'damping_ns_m': 0., 'stiffness_n_m': 8000.},
                       {'damping_ns_m': 4000., 'stiffness_n_m': 17000.},
                       {'damping_ns_m': 4000., 'stiffness_n_m': 8000., 'extra': 1}]:
            with self.assertRaises(ValueError):
                _validate_tangential_config(config)


if __name__=='__main__':unittest.main()
