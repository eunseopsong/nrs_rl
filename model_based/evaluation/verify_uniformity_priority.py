#!/usr/bin/env python3
"""Reconstruct uniformity results independently from the saved control traces."""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
import torch
from nrs_rl.tasks.manager_based.nrs_rl.model_based.policies.spatial_uniformity import PlanarPrestonSurface


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(root):
    result=json.loads((root/'result.json').read_text());checks=[]
    checkpoint=root/'candidate/checkpoints/actor.pt'
    assert sha(checkpoint)==result['checkpoint_sha256']
    extra={'spatial_policy.json':''};actor=torch.jit.load(str(checkpoint),_extra_files=extra).eval()
    contract=json.loads(extra['spatial_policy.json'])
    assert contract==result['runtime_contract']
    assert contract==json.loads((root/'candidate/spatial_policy.json').read_text())
    assert contract['rotation_contribution'] is False and contract['second_complete_pass'] is False
    assert contract['total_removal_is_selection_gate'] is False
    assert contract['hard_force_stop_n']==100.
    assert result['selection_seed'] not in result['heldout_seeds']
    assert result['holdouts_used_for_selection'] is False
    assert result['new_gui_experiment_executed'] is False
    with torch.inference_mode():
        actions=actor(torch.zeros(64,16))
    assert actions.shape==(64,3) and torch.isfinite(actions).all() and actions.abs().max()<=1.
    manifests={}
    for directory in [root/'candidate',Path(result['export_directory'])]:
        manifest=json.loads((directory/'manifest.json').read_text())
        for rel,digest in manifest.items():assert sha(directory/rel)==digest,rel
        manifests[str(directory)]=len(manifest)
    for rel,digest in json.loads((root/'source_manifest.json').read_text()).items():
        assert sha(root/'source_snapshot'/rel)==digest,rel
    for label in ['selection_2101','held_out_2111','held_out_2112']:
        directory=root/label;summary=json.loads((directory/'summary.json').read_text())
        processing=json.loads((directory/'processing_summary.json').read_text())
        assert summary['hard_force_stop_n']==100. and summary['uniformity_only'] is True
        assert summary['rotation_contribution'] is False
        assert summary['checkpoint_sha256'][result['selected']]==result['checkpoint_sha256']
        assert processing['processing_roi_cells']==61277
        with h5py.File(directory/'path.h5') as h:path=h['position'][:,:3]
        surface=PlanarPrestonSurface(path,cell_size_mm=.5)
        roi=surface.roi.copy();arc=np.r_[0.,np.linalg.norm(np.diff(path,axis=0),axis=1).cumsum()]
        for p in path[arc<=18.]:
            rows,cols,_,_,weights,_=surface._footprint((p-surface.origin)@surface.basis.T)
            roi[rows[weights>0],cols[weights>0]]=False
        for index,name in enumerate(summary['names']):
            metrics=processing['candidates'][name];surface.reset()
            with np.load(directory/f'{name}_trace.npz') as trace:
                time=trace['time_s'];keep=time>=2.;force=trace['normal_force_n'];speed=trace['measured_speed_mm_s']
                fault=trace['safety_fault_reason'];contact=(force>=1.5)&(fault==0)
                assert np.allclose(np.diff(time),.008,atol=1.e-10)
                assert np.isfinite(force).all() and np.isfinite(speed).all()
                assert np.allclose(trace['raw_mrr_n_mm_s'],np.where(contact,force*speed,0.))
                target=trace['target_force_n'];low,high=summary['force_bounds_n'][index]
                assert target.min()>=low-1.e-5 and target.max()<=high+1.e-5
                assert np.max(np.abs(np.diff(target)))<=.04001
                assert trace['commanded_speed_mm_s'].min()>=-3.-1.e-5
                assert trace['commanded_speed_mm_s'].max()<=9.+1.e-5
                assert metrics['reverse_mm']<=.05*summary['path_length_mm']+1.e-3
                assert np.isclose(keep.sum()*.008,metrics['time_s'])
                assert np.isclose(force[keep].mean(),metrics['force_mean_n'])
                assert np.isclose(force[~keep].max(),metrics['entry_force_peak_n'])
                assert np.isclose(np.mean(fault[keep]!=0),metrics['fault_fraction'])
                rate=trace['raw_mrr_n_mm_s'][keep]
                assert np.isclose(rate.std()/rate.mean(),metrics['rate_cv'])
                for i in np.flatnonzero(keep):
                    surface.deposit(trace['deposition_midpoint_mm'][i],trace['tangent_velocity_mm_s'][i],
                        float(force[i]),.008,contact=bool(contact[i]))
            with np.load(directory/f'{name}_processing_surface.npz') as saved:
                assert np.array_equal(saved['roi'],roi)
                assert np.allclose(surface.depth,saved['depth_over_k'],atol=1.e-10,rtol=1.e-10)
                values=surface.depth[roi]
                assert np.isclose(values.std()/values.mean(),metrics['spatial_cv'],rtol=1.e-10)
                assert np.isclose(values.sum()*surface.area,metrics['roi_volume_over_k'],rtol=1.e-10)
                assert np.isclose(np.mean(values==0.),metrics['zero_fraction'])
            checks.append({'batch':label,'policy':name,'samples':len(time),'passed':True})
            print('AUDITED',label,name,flush=True)
    audit={'passed':True,'runs':checks,'artifact_manifests':manifests,
        'checkpoint_sha256':sha(checkpoint),'spatial_depth_reconstructed_from_traces':True}
    (root/'verification.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({'passed':True,'runs':len(checks)}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    verify(parser.parse_args().root.resolve())
