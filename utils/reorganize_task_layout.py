"""One-time, audited relocation into the task Git repository; preserve old paths.

This changes module locations and import names, never numeric policy/control code.
Run after saving the active experiment's source snapshot.
"""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil

from nrs_rl.tasks.manager_based.nrs_rl.utils.removal_metrics import save, sha
from nrs_rl.tasks.manager_based.nrs_rl.paths import ROOT, TASK_ROOT

PREFIX = 'nrs_rl.tasks.manager_based.nrs_rl'


def relocate(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    old_model = TASK_ROOT.parent.parent/'model_based'
    assert not old_model.is_symlink() and not (TASK_ROOT/'model_based').exists()
    original_files = {}
    for directory in [TASK_ROOT, old_model]:
        for p in directory.rglob('*'):
            if p.is_file() and p.suffix in {'.py','.yaml','.md','.json'} and not any(
                    x in p.relative_to(directory).parts for x in ['.git','__pycache__','logs','outputs','build']):
                original_files[p] = p.read_bytes()
    # Immutable copies allow exact before/after and rollback without touching Git.
    for p, contents in original_files.items():
        dest=output/'before'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(contents)
    moved = {}

    def move_tree(old, new):
        old_files = [p for p in old.rglob('*') if p.is_file() and not p.is_symlink()]
        new.parent.mkdir(parents=True, exist_ok=True)
        old.rename(new)
        old.symlink_to(os.path.relpath(new,old.parent),target_is_directory=True)
        moved.update({p:new/p.relative_to(old) for p in old_files})

    def move_file(old, new):
        new.parent.mkdir(parents=True,exist_ok=True)
        old.rename(new);old.symlink_to(os.path.relpath(new,old.parent))
        for initial,current in list(moved.items()):
            if current == old:
                moved[initial]=new
                break
        else:moved[old]=new

    move_tree(old_model,TASK_ROOT/'model_based')
    for name in ['agents','evaluation','runtime']:
        move_tree(TASK_ROOT/name,TASK_ROOT/'mdp'/name)
    move_tree(TASK_ROOT/'model_based/reporting',TASK_ROOT/'utils/reporting')
    for filename in ['export_fixed_force_ppo_results.py','export_ppo_refinement_results.py','build_ppo_report.py']:
        move_file(TASK_ROOT/'mdp/evaluation'/filename,TASK_ROOT/'utils/reporting'/filename)
    (TASK_ROOT/'utils/diagnostics').mkdir(exist_ok=True)
    (TASK_ROOT/'utils/diagnostics/__init__.py').write_text('"""Common robot, ROS, and visualization diagnostics."""\n')
    for filename in ['validate_polishing_diagnostics.py','validate_d14_return_measurement.py','validate_d14_ros_bridge.py']:
        move_file(TASK_ROOT/'mdp/evaluation'/filename,TASK_ROOT/'utils/diagnostics'/filename)
    move_tree(TASK_ROOT/'tests',TASK_ROOT/'utils/tests')
    (TASK_ROOT/'mdp/tests').mkdir(exist_ok=True)
    (TASK_ROOT/'mdp/tests/__init__.py').write_text('"""Neural-policy and reinforcement-learning tests."""\n')
    for filename in ['test_fixed_force_ppo.py','test_velocity_actor.py']:
        move_file(TASK_ROOT/'utils/tests'/filename,TASK_ROOT/'mdp/tests'/filename)
    move_file(TASK_ROOT/'analyze_skrl_tensorboard.py',TASK_ROOT/'utils/reporting/analyze_skrl_tensorboard.py')

    def module(p):
        rel=p.relative_to(ROOT/'source/nrs_rl').with_suffix('')
        parts=list(rel.parts)
        if parts[-1]=='__init__':parts.pop()
        return '.'.join(parts)

    module_map={module(old):module(new) for old,new in moved.items() if old.suffix=='.py'}
    def remap(name):
        for old,new in sorted(module_map.items(),key=lambda x:-len(x[0])):
            if name==old or name.startswith(old+'.'):return new+name[len(old):]
        return name

    rewrites=[]
    for original,contents in original_files.items():
        p=moved.get(original,original)
        if p.suffix!='.py' or 'fixed_force_12p59_source' in p.parts:continue
        source=contents.decode()
        tree=ast.parse(source)
        old_package=module(original) if original.name=='__init__.py' else module(original).rsplit('.',1)[0]
        lines=source.splitlines(keepends=True)
        offsets=[0]
        for line in lines:offsets.append(offsets[-1]+len(line))
        replacements=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom) and node.level:
                absolute=importlib.util.resolve_name('.'*node.level+(node.module or ''),old_package)
                names=', '.join(a.name+(' as '+a.asname if a.asname else '') for a in node.names)
                replacements.append((offsets[node.lineno-1]+node.col_offset,
                                     offsets[node.end_lineno-1]+node.end_col_offset,
                                     'from '+remap(absolute)+' import '+names))
        for start,end,text in sorted(replacements,reverse=True):source=source[:start]+text+source[end:]
        for old,new in sorted(module_map.items(),key=lambda x:-len(x[0])):
            source=source.replace(old,new)
        # Package-resource strings and explicit executable source paths.
        source=source.replace("f'{__name__}.mdp.agents:skrl_ppo_cfg.yaml'", "f'{__name__}.mdp.agents:skrl_ppo_cfg.yaml'")
        for name in ['agents','evaluation','runtime']:
            source=source.replace('manager_based/nrs_rl/'+name+'/', 'manager_based/nrs_rl/mdp/'+name+'/')
        source=source.replace("TASK_ROOT / 'mdp/evaluation/", "TASK_ROOT / 'mdp/evaluation/")
        ast.parse(source)
        if source!=p.read_text():
            p.write_text(source);rewrites.append(str(p.relative_to(TASK_ROOT)))
    # Preserve exact bytes of deployment-pinned private dependencies.
    for original,contents in original_files.items():
        if 'fixed_force_12p59_source' in original.parts:
            assert moved.get(original,original).read_bytes()==contents
    manifest={'moved':{str(k.relative_to(ROOT)):str(v.relative_to(ROOT)) for k,v in moved.items()
                       if '__pycache__' not in k.parts},'import_rewrites':rewrites,
              'frozen_runtime_bytes_unchanged':True,'compatibility_links_preserved':True,
              'before_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(c).hexdigest() for p,c in original_files.items()}}
    save(output/'manifest.json',manifest)
    return manifest


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    result=relocate(p.parse_args().output)
    print(json.dumps({'moved_files':len(result['moved']),'rewritten_files':len(result['import_rewrites']),
                      'frozen_runtime_bytes_unchanged':True}))
