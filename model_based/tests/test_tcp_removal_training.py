"""Deployment and anti-shortcut checks for the RPM-free episode objective."""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch
from nrs_rl.tasks.manager_based.nrs_rl.model_based.training.train_tcp_removal import assess, TcpRemovalActor, SCHEDULER_CONTRACT, _build_agent


class TcpRemovalTests(unittest.TestCase):
    def baseline(self):
        return dict(completed=True,processing_samples=1000,processing_rate_cv=.3,
            force_peak_n=25.,force_rmse_n=2.,cross_track_rmse_mm=2.,
            endpoint_tangential_error_mm=5.,command_jerk_rms_mm_s3=10.,
            fault_fraction=0.,shield_fraction=0.,
            preston_surface=dict(roi_cells=100,roi_area_mm2=25.,spatial_depth_cv=.5,
                                 roi_volume_over_k=1000.,zero_depth_fraction=.1))

    def improved(self):
        value=self.baseline(); value['processing_rate_cv']=.25
        value['preston_surface']['spatial_depth_cv']=.45
        return value

    def test_frozen_gate_rejects_volume_time_force_and_missing_area_shortcuts(self):
        baseline=self.baseline()
        self.assertTrue(assess(self.improved(),[baseline],gate=True)['passed'])
        for key,bad in [('processing_samples',1100),('force_peak_n',30.),('force_rmse_n',3.),
                        ('cross_track_rmse_mm',3.),('endpoint_tangential_error_mm',7.),
                        ('shield_fraction',.001),('completed',False)]:
            candidate=self.improved(); candidate[key]=bad
            self.assertFalse(assess(candidate,[baseline],gate=True)['passed'],key)
        for key,bad in [('roi_volume_over_k',800.),('zero_depth_fraction',.1001)]:
            candidate=self.improved(); candidate['preston_surface'][key]=bad
            self.assertFalse(assess(candidate,[baseline],gate=True)['passed'],key)
        candidate=self.improved(); candidate['preston_surface']['roi_cells']=99
        with self.assertRaises(ValueError): assess(candidate,[baseline])

    def test_search_cannot_use_only_the_easier_duplicate_baseline(self):
        baseline=self.baseline(); tougher=copy.deepcopy(baseline)
        tougher['preston_surface']['spatial_depth_cv']=.4
        result=assess(self.improved(),[baseline,tougher],gate=True)
        self.assertFalse(result['passed'])
        self.assertLess(result['comparisons'][1]['spatial_cv_gain'],0.)

    def test_causal_action_direction_contact_gate_and_torchscript_transfer(self):
        actor=TcpRemovalActor([.5,.1,.3,.2,.06,.1])
        obs=torch.zeros(101,14); obs[:,1]=1.; obs[:,10]=1.
        obs[:,9]=torch.linspace(0,1,101)
        self.assertTrue(torch.all(actor(obs).squeeze().diff()<=0))
        self.assertGreater(float(actor(obs)[0]),float(actor(obs)[-1]))
        torch.testing.assert_close(actor(obs),torch.jit.script(actor)(obs))
        obs[:,11]=1.
        torch.testing.assert_close(actor(obs),torch.zeros(101,1))
        obs[:,11]=0.; obs[:,10]=0.
        torch.testing.assert_close(actor(obs),torch.zeros(101,1))

    def test_runtime_roundtrip_and_rotation_contract_rejection(self):
        contract={**SCHEDULER_CONTRACT,'schema_version':3,'observation_size':14,
            'control_period_s':.008,'policy_action_repeat':1,'policy_period_s':.008,
            'policy_backend':'torchscript_tcp_removal','objective':'preston_tcp_spatial_removal',
            'spindle_rpm':None,'rotation_contribution':False,'physics_tool_diameter_mm':30.}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/'params').mkdir(); (root/'checkpoints').mkdir()
            sidecar=root/'params/velocity_policy.json'; sidecar.write_text(json.dumps(contract))
            path=root/'checkpoints/actor.pt'; actor=TcpRemovalActor([.2,.1,.1,.1,.05,.1])
            torch.jit.save(torch.jit.script(actor),str(path),_extra_files={'velocity_policy.json':json.dumps(contract)})
            agent,_=_build_agent(str(path))
            torch.testing.assert_close(actor(torch.zeros(3,14)),agent.act(torch.zeros(3,14))[0])
            contract['spindle_rpm']=1000.; sidecar.write_text(json.dumps(contract))
            with self.assertRaisesRegex(ValueError,'exclude spindle'): _build_agent(str(path))


if __name__=='__main__': unittest.main()
