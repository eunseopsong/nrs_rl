"""Open the unchanged production USD in a visible GUI for one ROS trial."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
args = p.parse_args()
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=True)
source = Path('/home/eunseop/isaac/isaac_save/ur10e_v6.usd')
expected = '7314969427068562bf9cd8f1e9fa09962b8ae06b48b9a33af103ade0cbb30248'
assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
assert os.environ['ROS_DOMAIN_ID'] == '181'

from isaacsim import SimulationApp
app = SimulationApp({'headless': False, 'width': 1280, 'height': 720,
                     'window_width': 1440, 'window_height': 900,
                     'renderer': 'RaytracedLighting'})
import carb
import numpy as np
import omni.usd
from pxr import UsdGeom, Usd, Gf
from isaacsim.core.api import SimulationContext
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.utils.viewports import set_camera_view
from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file

settings = carb.settings.get_settings()
settings.set('/app/omni.graph.scriptnode/opt_in', True)
enable_extension('isaacsim.ros2.bridge')
enable_extension('omni.graph.scriptnode')
app.update()
context = omni.usd.get_context()
context.open_stage(str(source))
while context.get_stage_loading_status()[2] > 0:
    app.update()
stage = context.get_stage()
stage.SetEditTarget(stage.GetSessionLayer())
# Resolve the same external sensor script without relying on the working directory.
sensor = stage.GetPrimAtPath('/World/ActionGraphFT/script_node')
sensor.GetAttribute('inputs:scriptPath').Set(str(source.parent/'ft_sensor/ft_sensor.py'))
settings.set('/app/omni.graph.scriptnode/opt_in', True)
sim = SimulationContext(physics_dt=.008, rendering_dt=0., stage_units_in_meters=1.,
    physics_prim_path='/World/Robot/physicsScene', device='cpu')
sim.initialize_physics()
sim.play()
robot = SingleArticulation('/World/Robot')
robot.initialize()
set_camera_view(eye=np.array([1.75, 1.8, 1.1]),
                target=np.array([.65, .25, .35]))

def state():
    cylinder = UsdGeom.Cylinder(stage.GetPrimAtPath('/World/Robot/spindle_link/collisions/mesh_3/cylinder'))
    world = UsdGeom.Xformable(cylinder.GetPrim()).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    point = world.Transform(Gf.Vec3d(0., 0., cylinder.GetHeightAttr().Get()/2.))
    base = UsdGeom.Xformable(stage.GetPrimAtPath('/World/Robot/base_link')).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    local = base.GetInverse().Transform(point)
    return {'simulation_time_s': float(sim.current_time),
        'joint_names': list(robot.dof_names),
        'joint_positions_rad': robot.get_joint_positions().tolist(),
        'joint_velocities_rad_s': robot.get_joint_velocities().tolist(),
        'contact_face_world_mm': [float(v)*1000. for v in point],
        'contact_face_dh_base_mm': [-float(local[0])*1000., -float(local[1])*1000., float(local[2])*1000.]}

def write(name, value):
    temp = out/(name+'.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(out/name)

write('scene_provenance.json', {'source': str(source), 'source_sha256': expected,
    'headless': False, 'ros_domain': 181, 'physics_dt_s': .008,
    'render_every_physics_steps': 4, 'source_usd_modified': False,
    'sensor_script': str(source.parent/'ft_sensor/ft_sensor.py'),
    'sensor_script_sha256': hashlib.sha256((source.parent/'ft_sensor/ft_sensor.py').read_bytes()).hexdigest(),
    'initial_state': state()})
frame = 0
captures = []
started = time.monotonic()
try:
    while app.is_running() and not (out/'stop.json').exists():
        tick = time.monotonic()
        sim.step(render=False)
        if frame % 4 == 0:
            sim.render()
        frame += 1
        if frame == 500:
            write('ready.json', state())
            captures.append(capture_viewport_to_file(get_active_viewport(), str(out/'gui_initial.png')))
        if frame % 125 == 0:
            write('scene_status.json', {**state(), 'wall_seconds': time.monotonic()-started})
        request = out/'capture.json'
        if request.exists():
            filename = json.loads(request.read_text())['filename']
            assert Path(filename).name == filename
            captures.append(capture_viewport_to_file(get_active_viewport(), str(out/filename)))
            request.unlink()
        # ROS transport and control remain asynchronous, as in interactive GUI use.
        time.sleep(max(0., .008-(time.monotonic()-tick)))
        if time.monotonic()-started > 1200.:
            raise TimeoutError('GUI trial exceeded 20 minutes')
    write('scene_final.json', state())
finally:
    sim.stop()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
    app.close()
