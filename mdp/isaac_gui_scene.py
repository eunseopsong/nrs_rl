"""Load the frozen on-disk GUI geometry for separate headless training/evaluation."""
import json
from pathlib import Path
from ..utils.removal_metrics import sha


def configure_scene(cfg, directory, *, randomize):
    directory = Path(directory).resolve()
    profile = json.loads((directory / 'manifest.json').read_text())
    for name, digest in profile['files_sha256'].items():
        if sha(directory / name) != digest:
            raise ValueError('GUI scene snapshot changed: ' + name)
    cfg.sim.dt = profile['physics_dt_s']
    cfg.decimation = 1
    cfg.sim.render_interval = 1
    cfg.scene.robot.spawn.usd_path = str(directory / 'robot.usdc')
    cfg.scene.robot.spawn.rigid_props.disable_gravity = False
    cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = False
    cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count = 32
    cfg.scene.robot.spawn.articulation_props.solver_velocity_iteration_count = 1
    actuator = cfg.scene.robot.actuators['ur10_arm']
    # Let PhysX read the authored USD drive units. Copying the numerical USD
    # angular gains into Isaac's radian-valued config silently changes them.
    actuator.stiffness = None
    actuator.damping = None
    actuator.effort_limit_sim = None
    cfg.scene.workpiece.spawn.usd_path = str(directory / 'workpiece.usdc')
    integration = cfg.actions.arm_action.integration
    integration.joint_gravity_compensation = profile['gravity_compensation']
    integration.physical_tool_collision_relpath = profile['physical_collision_relpath']
    integration.physical_tool_diameter_mm = profile['physical_collision_pad_diameter_mm']
    if not randomize:
        integration.force_scale_range = (1., 1.)
        integration.force_bias_range_n = (0., 0.)
        integration.max_action_delay_steps = 0
    return profile
