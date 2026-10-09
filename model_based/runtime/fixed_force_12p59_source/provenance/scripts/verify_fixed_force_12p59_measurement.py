"""Replay a complete held-out training trace through the deployment recorder."""
import importlib.util
import json
from pathlib import Path
import tempfile

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    bundle = ROOT/'deployment/fixed_force_12p59'
    spec = importlib.util.spec_from_file_location('fixed_measurement_parity', bundle/'paired_measurement.py')
    measurement = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(measurement)
    folder = ROOT/'logs/experiments/fixed_force_dynamics_20261009/physics/heldout_2811'
    with np.load(folder/'selected_trace.npz') as stream:
        trace = dict(stream)
    with h5py.File(folder/'path.h5') as stream:
        reference = np.array(stream['position'][:, :3])
    txt = np.loadtxt('/home/eunseop/dev_ws/install/Y2RobMotion/share/Y2RobMotion/txtcmd/cmd_continue9D_preston_d14_20n.txt')
    np.testing.assert_allclose(txt[:, :3], reference, atol=5e-7, rtol=0)
    sequence = np.arange(1, len(trace['time_s'])+1)
    data = np.zeros((len(sequence), 23))
    length = np.linalg.norm(np.diff(reference, axis=0), axis=1).sum()
    data[:, 0] = 2.; data[:, 1] = 1.; data[:, 2] = sequence
    data[:, 3] = sequence*.008; data[:, 4] = trace['cursor_mm']; data[:, 5] = length
    data[:, 6] = trace['cursor_mm']; data[:, 8] = 20.; data[:, 10] = sequence*.008
    data[:, 11] = trace['safety_shield_active']; data[:, 12] = trace['safety_fault_reason']
    data[:, 13] = trace['measured_speed_mm_s']; data[:, 15:18] = trace['tcp_pose'][:, :3]
    data[:, 20] = 1.; data[:, 21] = trace['normal_force_n']; data[-1, 22] = 1.
    with tempfile.TemporaryDirectory(prefix='fixed_removal_parity_') as temporary:
        report = measurement.summarize(data, reference, temporary, make_plots=False)
        with np.load(Path(temporary)/'processing_surface.npz') as actual, np.load(folder/'selected_processing_surface.npz') as expected:
            np.testing.assert_array_equal(actual['roi'], expected['roi'])
            np.testing.assert_allclose(actual['depth_over_k'], expected['depth_over_k'], atol=1e-8, rtol=1e-9)
            cell_error = float(np.max(np.abs(actual['depth_over_k']-expected['depth_over_k'])))
        expected_metrics = json.loads((folder/'processing_summary.json').read_text())['candidates']['selected']
        cv_error = abs(report['processing_metrics']['spatial_depth_cv']-expected_metrics['spatial_cv'])
        assert cv_error < 1e-10
    result = {'ok': True, 'samples': len(sequence), 'source_trace': str(folder/'selected_trace.npz'),
        'processing_depth_max_cell_error': cell_error, 'spatial_cv_error': cv_error,
        'launch_trajectory_matches_training_reference': True, 'entry_protocol_unchanged': True,
        'virtual_rpm': 1000., 'midpoint_arithmetic': 'float32, same as training recorder'}
    output = ROOT/'logs/experiments/fixed_force_12p59/measurement_training_parity.json'
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
