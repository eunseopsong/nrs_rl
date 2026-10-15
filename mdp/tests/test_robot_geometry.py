"""Reject the two legacy geometry mismatches before starting Isaac."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from nrs_rl.tasks.manager_based.nrs_rl.mdp.robot_geometry import (
    TRAINING_CONFIG, configure_training_robot)


class RobotGeometryTests(unittest.TestCase):
    def test_current_config_is_selected_instead_of_controller_snapshot_yaml(self):
        with patch.dict(os.environ, {'Y2_ROBOT_CONFIG': '/old/snapshot/robot.yaml'}):
            result = configure_training_robot('/frozen/force/controller')
            self.assertEqual(result['tcp_length_mm'], 146.)
            self.assertEqual(result['robot_kinematics'], 'ur10e')
            self.assertEqual(os.environ['Y2_ROBOT_CONFIG'], str(TRAINING_CONFIG.resolve()))

    def test_rejects_obsolete_tcp_lengths(self):
        for length in (111., 114., 320.):
            with self.subTest(length=length), tempfile.TemporaryDirectory() as temporary:
                values = yaml.safe_load(TRAINING_CONFIG.read_text())
                values['EE2TCP'][2][3] = length
                path = Path(temporary) / 'robot.yaml'
                path.write_text(yaml.safe_dump(values))
                with self.assertRaisesRegex(ValueError, '146 mm'):
                    configure_training_robot('/controller', path)

    def test_rejects_cb3_kinematics_for_the_ur10e_usd(self):
        with tempfile.TemporaryDirectory() as temporary:
            values = yaml.safe_load(TRAINING_CONFIG.read_text())
            values['ROBOT_KINEMATICS'] = 'ur10'
            path = Path(temporary) / 'robot.yaml'
            path.write_text(yaml.safe_dump(values))
            with self.assertRaisesRegex(ValueError, 'UR10e'):
                configure_training_robot('/controller', path)


if __name__ == '__main__':
    unittest.main()
