# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

import unittest
import numpy as np

from nrs_rl.tasks.manager_based.nrs_rl.utils.tcp_tracking_compensation import TangentTrackingCompensation


class Core:
    def reset(self, pose):
        self.pose = list(pose)

    def step(self, measured, reference, force, wrench, rotation):
        self.reference = list(reference)
        return list(reference)


class TrackingTests(unittest.TestCase):
    def test_zero_gain_preserves_the_existing_command_exactly(self):
        core = Core()
        wrapper = TangentTrackingCompensation(core, 0.)
        desired = [1., 2., 3., .1, .2, .3]
        self.assertEqual(wrapper.step([0.]*6, desired, [0., 0., 20.], [0.]*6, np.eye(3).ravel()), desired)
        np.testing.assert_array_equal(wrapper.correction, np.zeros(3))

    def test_tangent_projection_preserves_normal_component_and_orientation(self):
        rotation = np.array([[1., 0., 0.], [0., .8, -.6], [0., .6, .8]])
        normal = rotation[:, 2]
        target = np.array([10., 12., 14., .1, .2, .3])
        wrapper = TangentTrackingCompensation(Core(), 1.)
        for _ in range(100):
            command = np.array(wrapper.step([0.]*6, target, [0., 0., 20.], [0.]*6, rotation.ravel()))
            self.assertLessEqual(np.linalg.norm(command[:3]-target[:3]), 4.+1e-12)
            self.assertAlmostEqual((command[:3]-target[:3])@normal, 0., places=12)
            np.testing.assert_array_equal(command[3:], target[3:])

    def test_filter_is_causal_and_reset_clears_compensation(self):
        wrapper = TangentTrackingCompensation(Core(), .5)
        target = [2., 0., 0., 0., 0., 0.]
        command = wrapper.step([0.]*6, target, [0., 0., 20.], [0.]*6, np.eye(3).ravel())
        self.assertAlmostEqual(command[0], 2.+wrapper.alpha)
        wrapper.reset([0.]*6)
        np.testing.assert_array_equal(wrapper.correction, np.zeros(3))

    def test_invalid_gain_and_nonfinite_pose_are_rejected(self):
        for gain in (-.01, 1.01, float('nan')):
            with self.assertRaises(ValueError):
                TangentTrackingCompensation(Core(), gain)
        wrapper = TangentTrackingCompensation(Core(), .5)
        with self.assertRaises(ValueError):
            wrapper.step([float('nan')]*6, [0.]*6, [0., 0., 20.], [0.]*6, np.eye(3).ravel())


if __name__ == '__main__':
    unittest.main()
