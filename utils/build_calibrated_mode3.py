#!/usr/bin/env python3
"""Build a separate research extension and prove default-controller parity."""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

from nrs_rl.tasks.manager_based.nrs_rl.paths import ROOT as _REPO_ROOT
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.cpp_extension import load

ROOT=_REPO_ROOT
OUT=ROOT/'logs/experiments/tcp_removal_retrain_20261005/calibrated_control'
BUILD=OUT/'native';BUILD.mkdir(parents=True,exist_ok=True)
SOURCE=Path('/home/eunseop/dev_ws/src/y2_ur10skku_control/Y2ForceCon')
os.environ['MAX_JOBS']='2'
module=load(name='_tcp_calibrated_mode3',sources=[str(ROOT/'scripts/calibrated_mode3.cpp'),
    *[str(SOURCE/'src'/name) for name in ('admittance_control.cpp','Naf_mdGradi.cpp','mode3_force_control_core.cpp')]],
    extra_include_paths=[str(SOURCE/'include')],extra_cflags=['-O2','-g0','-fvisibility=hidden'],
    build_directory=str(BUILD),with_cuda=False,verbose=True)
sys.path.insert(0,str(ROOT/'source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/y2_control_pybind'))
import y2_control_py
from y2_control_py import config
torch.set_num_threads(1)
args=[config.NAF_MDGRADI_CKPT,.008,config.FORCE_CON_COORDINATE,
      config.FORCE_SWITCH_DESIRED_FORCE_THRESHOLD,config.FORCE_SWITCH_ACTUAL_FORCE_THRESHOLD,
      config.FORCE_SWITCH_PRECONTACT_FORCE_HOLD,config.FORCE_SWITCH_RETURN_TAU]
old=y2_control_py.Mode3ForceController(*args)
new=module.TangentialMode3(*args,6000.,2000.)
pose=[870.,340.,100.,0.,0.,1.57]
old.reset(pose);new.reset(pose);error=0.
for i in range(500):
    reference=[870.+.03*i,340.+np.sin(i/100.),100.,0.,0.,1.57]
    wrench=[3.*np.sin(i/30.),-12.,20.+np.cos(i/20.),0.,0.,0.]
    inputs=[pose,reference,[0.,0.,20.],wrench,[1.,0.,0.,0.,1.,0.,0.,0.,1.]]
    a=np.array(old.step(*inputs));b=np.array(new.step(*inputs))
    error=max(error,float(abs(a-b).max()))
    pose=list(a[:6])
assert error<1e-10,error
for damping,stiffness in [(0.,2000.),(1000.,float('nan')),(7000.,2000.),(1000.,17000.)]:
    try: module.TangentialMode3(*args,damping,stiffness)
    except ValueError: pass
    else: raise AssertionError('Invalid experimental gain accepted')
report={'ok':True,'default_parity_steps':500,'maximum_error':error,
        'invalid_gain_rejected':True,'production_extension_replaced':False,
        'extension':str(BUILD/'_tcp_calibrated_mode3.so')}
(OUT/'native_validation.json').write_text(json.dumps(report,indent=2))
print('CALIBRATED_CORE_PARITY',json.dumps(report),flush=True)
