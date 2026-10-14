"""Build an isolated binding from the preserved current ROS force controller."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--save', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
saved = json.loads((args.save / 'SAVE.json').read_text())
controller = args.save / 'controller_source/Y2ForceCon'
sources = [controller / 'src' / name for name in (
    'admittance_control.cpp', 'Naf_mdGradi.cpp', 'mode3_force_control_core.cpp')]
for path in sources:
    relative = str(path.relative_to(args.save))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == saved['files_sha256'][relative]
args.output.mkdir(parents=True, exist_ok=True)
binding = Path(__file__).resolve().parents[2] / 'y2_control_pybind/cpp/src/api/calibrated_mode3.cpp'
generated = args.output / 'latest_force_mode3.cpp'
generated.write_text(binding.read_text().replace(
    'PYBIND11_MODULE(_tcp_calibrated_mode3,m)', 'PYBIND11_MODULE(_latest_force_mode3,m)'))
os.environ['MAX_JOBS'] = '2'
module = load(name='_latest_force_mode3', sources=[str(generated), *map(str, sources)],
    extra_include_paths=[str(controller / 'include')], extra_cflags=['-O2', '-g0', '-fvisibility=hidden'],
    build_directory=str(args.output), with_cuda=False, verbose=True)
torch.set_num_threads(1)
model = str(controller / 'src/checkpoints/NAF_MDGradi/NAF_mdGradi_policy_script.pt')
core = module.TangentialMode3(model, .008, 1, .01, 1.5, 15., .2, 6000., 2000.)
pose = [870., 340., 100., 0., 0., 1.57]
core.reset(pose)
rotation = [1., 0., 0., 0., 1., 0., 0., 0., 1.]
for step in range(100):
    output = core.step(pose, [870. + .02 * step, 340., 100., 0., 0., 1.57],
                       [0., 0., 20.], [0., 0., 20., 0., 0., 0.], rotation)
    assert output[12:14] == [6000., 6000.] and output[18:20] == [2000., 2000.]
    assert all(torch.isfinite(torch.tensor(output)))
    pose = output[:6]
report = {'ok': True, 'source_save': str(args.save), 'extension': module.__file__,
    'tangent_damping_ns_m': 6000., 'tangent_stiffness_n_m': 2000., 'tracking_compensation': False,
    'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
    'active_controller_unchanged': True}
(args.output / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
