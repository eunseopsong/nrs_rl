#!/usr/bin/env python3
"""Check the staged ROS measurement adapter without starting any ROS node."""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

from nrs_rl.tasks.manager_based.nrs_rl.paths import ROOT as _REPO_ROOT
import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT=_REPO_ROOT


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec); sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--node-script',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    sys.path.insert(0,str(args.node_script.parent))
    node=load('tcp_measurement_adapter',args.node_script)
    surface=load('tcp_measurement_surface',ROOT/'source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/utils/preston_surface.py')
    reference=np.loadtxt('/home/eunseop/dev_ws/src/y2_ur10skku_control/Y2RobMotion/txtcmd/cmd_continue9D_preston_d14_20n.txt')[:,:3]
    fake=SimpleNamespace(surface=surface.PlanarPrestonSurface(reference,cell_size_mm=.5),
        get_parameter=lambda name:SimpleNamespace(value={'contact_threshold_N':1.5,'depth_model':'tcp_fixed_roi'}[name]))
    method=node.PolishingRemovalNode.compute_d14_removal
    positions=np.tile(np.r_[reference[100],0.,0.,0.],(5,1))
    forces=np.tile([0.,0.,20.],(5,1)); times=np.arange(5)*.008
    result=method(fake,times,positions,forces)
    assert result.dremoval.sum()==0.,'Stationary TCP contact must not remove material when rotation is excluded'
    positions[:,0]+=np.arange(5)*.04
    result=method(fake,times,positions,forces)
    assert np.isclose(result.dremoval.sum(),20.*.16,rtol=1e-10)
    # Pure normal displacement cannot create tangential removal.
    positions[:,0]=positions[0,0];positions[:,2]+=np.arange(5)*.04
    assert method(fake,times,positions,forces).dremoval.sum()==0.
    try: method(fake,np.zeros(5),positions,forces)
    except ValueError: pass
    else: raise AssertionError('Nonmonotonic measurement clock accepted')
    # Under unchanged force and spatial path, feed retiming alone cancels in
    # F*v*dt. A rollout must demonstrate its claimed contact/tracking benefit.
    positions=np.tile(np.r_[reference[100],0.,0.,0.],(5,1));positions[:,0]+=np.arange(5)*.04
    method(fake,times,positions,forces); a=fake.surface.depth.copy()
    method(fake,times*2.,positions,forces);np.testing.assert_allclose(fake.surface.depth,a,atol=1e-12)
    parity={}
    diagnosis=json.loads((ROOT/'logs/experiments/tcp_removal_retrain_20261005/diagnosis/summary.json').read_text())
    for name,record in diagnosis['runs'].items():
        data=np.loadtxt(record['recording'],delimiter=',',skiprows=1)
        result=method(fake,data[:,0],data[:,1:7],data[:,7:10])
        metrics=fake.surface.metrics()
        for field in ('spatial_depth_cv','roi_volume_over_k','zero_depth_fraction'):
            np.testing.assert_allclose(metrics[field],record[field],rtol=1e-10,atol=1e-10)
        parity[name]=metrics
    report={'ok':True,'checks':['stationary zero removal','F*distance conservation','normal motion excluded',
        'bad clock rejected','feed retiming invariance','October 5 reanalysis parity'],'recording_parity':parity}
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False))
    print('TCP_MEASUREMENT_PASS',str(args.output))


if __name__=='__main__': main()
