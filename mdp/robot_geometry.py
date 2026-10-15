"""Explicit robot geometry for current Isaac PPO training and evaluation."""
import hashlib
import os
from pathlib import Path

import yaml

TRAINING_CONFIG = Path(__file__).with_name('config') / 'isaac_robot.yaml'
TCP_LENGTH_MM = 146.0
EE2TCP = [[-1., 0., 0., 0.], [0., 1., 0., 0.],
          [0., 0., -1., TCP_LENGTH_MM], [0., 0., 0., 1.]]


def configure_training_robot(controller_source, config=TRAINING_CONFIG):
    """Select the config before importing any kinematics/action modules."""
    path = Path(config).resolve()
    values = yaml.safe_load(path.read_text())
    if values['EE2TCP'] != EE2TCP:
        raise ValueError('Current Isaac training requires the 146 mm EE2TCP transform')
    if (values['ROBOT_KINEMATICS'], values['CONTROL_PERIOD'],
            values['NUMBER_OF_JOINTS']) != ('ur10e', .008, 6):
        raise ValueError('Expected UR10e matching the unchanged USD, 125 Hz, six joints')
    os.environ['Y2_CONTROL_SOURCE_DIR'] = str(Path(controller_source).resolve())
    os.environ['Y2_ROBOT_CONFIG'] = str(path)
    return {'robot_kinematics': 'ur10e', 'tcp_length_mm': TCP_LENGTH_MM, 'ee2tcp': EE2TCP,
            'config_path': str(path),
            'config_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'reference_frame': '/Robot/wrist_3_link/flange/tool0',
            'contact_prim': '/Robot/spindle_link/collisions/mesh_3/cylinder'}


def audit_scene_geometry(directory):
    """Read the existing USD; never author or save the stage."""
    from pxr import Gf, Usd, UsdGeom
    path = Path(directory) / 'robot.usdc'
    stage = Usd.Stage.Open(str(path))
    cylinder = UsdGeom.Cylinder(stage.GetPrimAtPath(
        '/Robot/spindle_link/collisions/mesh_3/cylinder'))
    tool0 = stage.GetPrimAtPath('/Robot/wrist_3_link/flange/tool0')
    if not cylinder or not tool0 or cylinder.GetAxisAttr().Get() != 'Z':
        raise ValueError('Expected the existing tool0 and Z-axis contact cylinder')
    relative = (UsdGeom.Xformable(cylinder).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                * UsdGeom.Xformable(tool0).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse())
    scale = UsdGeom.GetStageMetersPerUnit(stage) * 1000.
    center = relative.Transform(Gf.Vec3d(0., 0., 0.)) * scale
    end = relative.Transform(Gf.Vec3d(0., 0., cylinder.GetHeightAttr().Get() / 2.)) * scale
    if max(abs(end[i] - (TCP_LENGTH_MM if i == 2 else 0.)) for i in range(3)) > 1.e-3:
        raise ValueError('USD contact face does not match the configured 146 mm TCP')
    # Check the entire kinematic chain, not only the final tool translation.
    import numpy as np
    from ..y2_control_pybind.y2_control_py._y2_control_pybind import RobotKinematics
    names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
             'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
    joints = [stage.GetPrimAtPath('/Robot/joints/' + name) for name in names]
    q = np.deg2rad([joint.GetAttribute('state:angular:physics:position').Get()
                   for joint in joints])
    base = UsdGeom.Xformable(stage.GetPrimAtPath('/Robot/base_link_inertia'))
    in_base = (UsdGeom.Xformable(cylinder).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
               * base.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse())
    actual = np.asarray(in_base.Transform(Gf.Vec3d(0., 0., cylinder.GetHeightAttr().Get()/2.))) * scale
    predicted = np.asarray(RobotKinematics('ur10e', .008, EE2TCP).forward_kinematics(q.tolist()))[:3, 3]
    error = float(np.linalg.norm(actual - predicted))
    if error > .02:
        raise ValueError(f'UR10e/146 mm FK and USD contact face disagree by {error} mm')
    return {'robot_usd': str(path.resolve()),
            'robot_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'center_in_tool0_mm': list(center), 'contact_face_in_tool0_mm': list(end),
            'height_mm': cylinder.GetHeightAttr().Get() * scale, 'usd_modified': False,
            'ur10e_contact_face_fk_error_mm': error,
            'usd_contact_face_in_base_mm': actual.tolist(),
            'ur10e_contact_face_in_base_mm': predicted.tolist()}


def audit_live_geometry(term):
    """Check measured articulation poses against the kinematics actually in use."""
    import torch
    from isaaclab.utils.math import matrix_from_quat
    from ..y2_control_pybind.y2_control_py import config
    if config.ROBOT_KINEMATICS != 'ur10e' or config.EE2TCP != EE2TCP:
        raise ValueError('Runtime kinematics did not load the UR10e / 146 mm configuration')
    body = term.robot.body_names.index('spindle_link')
    rotations = matrix_from_quat(term.robot.data.body_quat_w[:, body])
    position = term.robot.data.body_pos_w[:, body] - term._env.scene.env_origins
    actual = position * 1000. + rotations[:, :, 2] * TCP_LENGTH_MM
    predicted = torch.stack([term._fk(i, q[:6])[1]
                             for i, q in enumerate(term.robot.data.joint_pos)])
    errors = torch.linalg.norm(actual - predicted, dim=1)
    if float(errors.max()) > .05:
        raise ValueError(f'Live USD contact face and FK differ: {errors.cpu().tolist()} mm')
    return {'contact_face_fk_error_mm': errors.cpu().tolist(),
            'maximum_error_mm': float(errors.max()), 'tolerance_mm': .05,
            'runtime_robot_kinematics': config.ROBOT_KINEMATICS,
            'runtime_tcp_length_mm': config.TCP_LENGTH_MM}
