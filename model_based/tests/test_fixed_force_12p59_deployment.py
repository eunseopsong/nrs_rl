"""Cross-language control parity and streaming checkpoint/reset tests."""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

import json
import math
from pathlib import Path
import subprocess
import unittest
import numpy as np
from nrs_rl.tasks.manager_based.nrs_rl.model_based.policies.spatial_uniformity import ROOT, REFERENCE, LocalReturn, SignedSpeedLimiter
import importlib.util
_spec=importlib.util.spec_from_file_location('spatial14p24_runtime',ROOT/'deployment/fixed_force_12p59/fixed_force_policy_runtime.py')
_runtime=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_runtime)
FixedForcePolicyRuntime=_runtime.FixedForcePolicyRuntime
from nrs_rl.tasks.manager_based.nrs_rl.utils.tcp_tracking_compensation import TangentTrackingCompensation
from nrs_rl.tasks.manager_based.nrs_rl.model_based.policies.fixed_force_refinement import ForwardSpeedLimiter

CHECKPOINT=ROOT/'deployment/fixed_force_12p59/checkpoints/actor.pt'


class EchoCore:
    def reset(self,pose):pass
    def step(self,measured,reference,force,wrench,rotation):return reference


class PairedSpatialDeploymentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source=ROOT/'deployment/fixed_force_12p59/runtime_source'
        cls.binary=ROOT/'deployment/fixed_force_12p59/build/spatial_parity_probe'
        subprocess.run(['g++','-std=c++17','-O2','-I'+str(source/'include'),
            str(source/'test/spatial_parity_probe.cpp'),str(source/'src/ppo_trajectory_scheduler.cpp'),
            '-o',str(cls.binary)],check=True)
        cls.runtime=FixedForcePolicyRuntime(CHECKPOINT,REFERENCE)

    def test_scalar_actor_bounds(self):
        import torch
        obs=torch.randn(4096,16)*5
        with torch.inference_mode(): actions=self.runtime.actor(obs)
        self.assertEqual(tuple(actions.shape),(4096,1))
        self.assertTrue(torch.isfinite(actions).all().item())
        speed=6.+6.*actions
        self.assertTrue(torch.all((speed>=1.5-1e-6)&(speed<=12.+1e-6)).item())

    def test_cpp_matches_python_force_speed_return_and_tracking(self):
        route=LocalReturn(700.);limiter=ForwardSpeedLimiter(12.);tracking=TangentTrackingCompensation(EchoCore(),.6)
        cursor=0.;filtered=0.;target_force=20.;fault=0;inputs=[];expected=[]
        for k in range(18000):
            measured=np.array([cursor-.6,.15*math.sin(k*.01),0.,0.,0.,0.])
            force=20. if k<17500 else 100.1
            a=.5*math.sin(k*.003);b=.5*math.sin(k*.004);c=1. if k%1000<10 else -1.
            enabled=not 5000<=k<5020
            inputs.append([*measured[:3],force,a,int(enabled)])
            before=cursor
            filtered+=np.clip((1.-math.exp(-.008/.08))*(a-filtered),-.024,.024)
            target_force=20.
            fault |= int(force>100.)
            request=np.clip(6.+6.*filtered,0.,12.)
            speed=limiter.step(request,enabled and force<=1.6*target_force and cursor<700.-1e-6 and not fault)
            cursor=route.advance(cursor,speed*.008)
            reference=tracking.step(measured,[before,0.,0.,0.,0.,0.],[0.,0.,target_force],np.zeros(6),np.eye(3))
            expected.append([cursor,speed,target_force,route.reverse_mm,route.frontier,route.events,fault,*reference[:3]])
        text='\n'.join(' '.join(format(x,'.17g') for x in row) for row in inputs)+'\n'
        result=subprocess.run([str(self.binary)],input=text,text=True,capture_output=True,check=True)
        actual=np.array([[float(x) for x in line.split()] for line in result.stdout.splitlines()])
        np.testing.assert_allclose(actual,expected,atol=2e-8,rtol=2e-9)
        self.assertEqual(route.events,0)
        self.assertEqual(route.reverse_mm,0.)
        self.assertTrue(np.all(actual[17500:,1]==0.))

    def test_entry_overshoot_and_100n_boundary(self):
        # The GUI failed on its first sample at 68.9 N. Hold it longer than
        # two seconds to ensure there is no hidden entry timeout, then
        # exercise the exact user-selected hard-stop boundary and latching.
        forces=[68.89791117615832]*300+[99.9,100.]+[20.]*50+[100.1]+[20.]*20
        lines=['0 0 0 '+format(force,'.17g')+' 0 1' for force in forces]
        result=subprocess.run([str(self.binary)],input='\n'.join(lines)+'\n',
            text=True,capture_output=True,check=True)
        actual=np.array([[float(x) for x in line.split()] for line in result.stdout.splitlines()])
        self.assertTrue(np.all(actual[:352,6]==0.))
        self.assertTrue(np.all(actual[:302,0]==0.))
        self.assertGreater(actual[351,0],0.)
        self.assertTrue(np.all(actual[352:,6]==1.))
        np.testing.assert_array_equal(actual[352:,0],np.full(21,actual[351,0]))

    def telemetry(self,epoch,seq):
        x,y,z=self.runtime.geometry.positions[0]
        cursor=max(0.,(seq-1)*.008*6.)
        return [2.,epoch,seq,seq*.008,cursor,self.runtime.geometry.length,cursor,0.,20.,0.,seq*.008,
            0.,0.,6.,.5,x+cursor,y,z,0.,0.,1.,20.,0.]

    def test_streaming_resets_and_policy_cadence(self):
        r=self.runtime;r.reset(0)
        first=r.process(self.telemetry(1,1))
        for seq in range(2,102):r.process(self.telemetry(1,seq))
        self.assertEqual(r.decisions,11)
        self.assertGreater(float(r.surface.depth.sum()),0.)
        self.assertEqual(r.missing_samples,0)
        again=r.process(self.telemetry(2,1))
        np.testing.assert_array_equal(first['observation'],again['observation'])
        np.testing.assert_array_equal(first['action'][2:],again['action'][2:])
        with self.assertRaises(ValueError):r.process(self.telemetry(2,1))
        with self.assertRaises(ValueError):r.process(self.telemetry(1,2))

    def test_wrong_telemetry_and_fault_fail_closed(self):
        r=self.runtime
        for change in ('nan','length','fault','gap','force','reverse','schema'):
            r.reset(0);r.process(self.telemetry(1,1));value=self.telemetry(1,2)
            if change=='nan':value[21]=float('nan')
            elif change=='length':value[5]+=10.
            elif change=='fault':value[12]=4.
            elif change=='force':value[8]=19.
            elif change=='reverse':value[7]=.1
            elif change=='schema':value[0]=1.
            else:value=self.telemetry(1,100)
            with self.assertRaises(ValueError):r.process(value)


class MatchedArmTests(unittest.TestCase):
    def test_mode6_constant_applied_but_actor_inference_retained(self):
        a=FixedForcePolicyRuntime(CHECKPOINT,REFERENCE,5)
        b=FixedForcePolicyRuntime(CHECKPOINT,REFERENCE,6)
        x,y,z=a.geometry.positions[0]
        proposed=[]
        for seq in range(1,32):
            cursor=(seq-1)*.048
            v=[2.,1.,seq,seq*.008,cursor,a.geometry.length,cursor,0.,20.,0.,seq*.008,
               0.,0.,6.,.5,x+cursor,y,z,0.,0.,1.,20.,0.]
            ra=a.process(v);rb=b.process(v)
            np.testing.assert_array_equal(ra['observation'],rb['observation'])
            np.testing.assert_array_equal(ra['actor_action'],rb['actor_action'])
            np.testing.assert_array_equal(rb['action'][2:],[0.])
            np.testing.assert_array_equal(ra['action'][2:],ra['actor_action'])
            proposed.append(rb['actor_action'])
        self.assertGreater(np.max(np.abs(np.array(proposed)[:,:2])),.1)
        self.assertEqual(a.decisions,b.decisions)
        b.reset(0)
        self.assertEqual(b.force_control_mode,6)
        with self.assertRaises(ValueError):FixedForcePolicyRuntime(CHECKPOINT,REFERENCE,7)

    def test_controller_tick_measurement_integral_and_gap_rejection(self):
        import tempfile
        import importlib.util
        spec=importlib.util.spec_from_file_location('paired_measurement_test',ROOT/'deployment/fixed_force_12p59/paired_measurement.py')
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        rec=m.Recording()
        count=401;length=(count-1)*.008*6.
        for seq in range(1,count+1):
            cursor=(seq-1)*.008*6.
            # Duplicate callback simulation timestamps are allowed: the control
            # index still supplies the exact integration period.
            v=[2.,1.,seq,(seq//2)*.016,cursor,length,cursor,0.,20.,0.,seq*.008,
               0.,0.,6.,0.,cursor,0.,0.,0.,0.,1.,20.,float(seq==count)]
            rec.append(v)
        with tempfile.TemporaryDirectory() as folder:
            result=m.summarize(rec.rows,np.array([[0.,0.,0.],[length,0.,0.]]),folder,make_plots=False)
            self.assertTrue(result['valid_completed_run'])
            self.assertEqual(result['virtual_spindle_rpm'],1000.)
            self.assertGreater(result['metrics']['grid_volume_over_k'],20.*length*50.)
            self.assertAlmostEqual(result['processing_metrics']['processing_time_s'],(count-250)*.008,places=8)
        self.assertFalse(rec.append(v))
        broken=m.Recording();broken.append(rec.rows[0])
        with self.assertRaises(ValueError):broken.append(rec.rows[2])
        with self.assertRaises(ValueError):m.Recording().append(rec.rows[1])

if __name__=='__main__':unittest.main()
