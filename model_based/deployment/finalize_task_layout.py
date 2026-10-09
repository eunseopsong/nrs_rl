"""After verification, retire the external model_based path and refresh outputs.

Maintained code lives in this task Git repository. Three deployment Python files
are generated copies of the adjacent templates; actor and control binaries stay
byte-identical. This tool does not launch ROS or move a robot.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from nrs_rl.tasks.manager_based.nrs_rl.paths import ROOT, TASK_ROOT
from nrs_rl.tasks.manager_based.nrs_rl.utils.removal_metrics import sha, save


def finish(study):
    study=Path(study).resolve()
    result=json.loads((study/'final_report.json').read_text())
    assert result['numerical_audit_passed']
    assert json.loads((study/'status.json').read_text())['phase']=='complete'
    old=TASK_ROOT.parent.parent/'model_based'
    destination=TASK_ROOT/'model_based'
    assert old.is_symlink() and old.resolve()==destination.resolve()
    deployment=ROOT/'deployment/fixed_force_12p59'
    before=json.loads((deployment/'deployment.json').read_text())
    for filename,digest in before['pinned_files'].items():
        assert sha(filename)==digest, filename
    actor_hash=sha(before['checkpoint'])
    binary_hashes={k:sha(before[k]) for k in ['motion_binary','command_binary']}
    backup=study/'layout_reorganization/deployment_before_retiring_external_path'
    backup.mkdir(exist_ok=False)
    templates=Path(__file__).resolve().parent/'templates/fixed_force_12p59'
    changed=[]
    old_prefix='source/nrs_rl/nrs_rl/tasks/model_based/'
    new_prefix='source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/model_based/'
    for source in sorted(templates.glob('*.py')):
        target=deployment/source.name
        # Only path literals are allowed to differ from the validated runtime.
        assert source.read_text()==target.read_text().replace(old_prefix,new_prefix)
        shutil.copy2(target,backup/source.name)
        shutil.copy2(source,target)
        changed.append(str(target))
    shutil.copy2(deployment/'deployment.json',backup/'deployment.json')

    def repath(value):
        if isinstance(value,str):return value.replace(old_prefix,new_prefix)
        if isinstance(value,list):return [repath(x) for x in value]
        if isinstance(value,dict):return {repath(k):repath(v) for k,v in value.items()}
        return value

    after=repath(before)
    for filename,digest in after['pinned_files'].items():
        if filename in changed:after['pinned_files'][filename]=sha(filename)
        else:assert sha(filename)==digest, filename
    after['task_repository_layout']={'canonical_model_based':str(destination),
        'updated_at_utc':datetime.now(timezone.utc).isoformat(),
        'actor_and_control_binaries_unchanged':True,'generator':str(Path(__file__).resolve())}
    save(deployment/'deployment.json',after)
    # These are existing compatibility links, not additional source copies.
    links=[]
    for p in (ROOT/'scripts').rglob('*'):
        if p.is_symlink():
            resolved=p.resolve()
            if resolved.is_relative_to(TASK_ROOT):
                previous=os.readlink(p)
                target=os.path.relpath(resolved,p.parent)
                if previous!=target:
                    p.unlink();p.symlink_to(target)
                    links.append({'path':str(p),'before':previous,'after':target})
    old.unlink()
    assert not old.exists() and not old.is_symlink()
    for filename,digest in after['pinned_files'].items():assert sha(filename)==digest, filename
    assert sha(after['checkpoint'])==actor_hash
    assert all(sha(after[k])==v for k,v in binary_hashes.items())
    preflight=subprocess.run([sys.executable,str(deployment/'preflight.py')],
                             check=True,capture_output=True,text=True,cwd=ROOT)
    for p in (ROOT/'scripts').rglob('*'):
        if p.is_symlink():assert p.exists(),p
    record={'complete':True,'external_model_based_removed':True,'canonical_directory':str(destination),
        'checkpoint_sha256_unchanged':actor_hash,'binary_hashes_unchanged':binary_hashes,
        'deployment_templates':str(templates),'refreshed_deployment_outputs':changed,
        'retargeted_existing_script_links':links,'deployment_preflight':preflight.stdout.splitlines(),
        'pinned_files_verified':len(after['pinned_files'])}
    save(study/'layout_reorganization/finalized.json',record)
    print(json.dumps({'complete':True,'external_model_based_removed':True,
        'canonical_directory':str(destination),'checkpoint_sha256_unchanged':actor_hash,
        'pinned_files_verified':record['pinned_files_verified'],
        'retargeted_script_links':len(links),'deployment_preflight':record['deployment_preflight']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True)
    finish(p.parse_args().study)
