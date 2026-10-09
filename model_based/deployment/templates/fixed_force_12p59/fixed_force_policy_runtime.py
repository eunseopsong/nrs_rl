"""ROS-independent streaming adapter using the exact trained spatial features."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/model_based/runtime/fixed_force_12p59_source'))
from nrs_fixed_force_12p59.spatial_uniformity import OBSERVATIONS
from nrs_fixed_force_12p59.fixed_force_velocity import FixedForceGeometry
from nrs_fixed_force_12p59.fixed_force_refinement import check_contract

CHECKPOINT_SHA256 = '6d1dc47bc915b393880d2e6e380f44fbcb82e542de0fa12e4f02eb47c97147df'
TELEMETRY = ['schema','epoch','sequence','simulation_time_s','cursor_mm','path_length_mm',
    'frontier_mm','reverse_mm','target_force_n','return_allowed','elapsed_s','shield','fault',
    'actual_speed_mm_s','tracking_error_mm','tcp_x_mm','tcp_y_mm','tcp_z_mm',
    'normal_x','normal_y','normal_z','normal_force_n','done']


class FixedForcePolicyRuntime:
    def __init__(self, checkpoint, reference, force_control_mode=5):
        if force_control_mode not in (5,6):
            raise ValueError("Expected Mode 5 or matched Mode 6")
        self.force_control_mode = force_control_mode
        torch.set_num_threads(1)
        checkpoint, reference = Path(checkpoint), Path(reference)
        self.sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if self.sha256 != CHECKPOINT_SHA256:
            raise ValueError('This runtime requires the frozen fixed20_virtual1000_12p59 checkpoint')
        extra = {'fixed_force_policy.json': ''}
        self.actor = torch.jit.load(str(checkpoint), map_location='cpu', _extra_files=extra).eval()
        self.contract = json.loads(extra['fixed_force_policy.json'])
        c = self.contract
        minimum, maximum = check_contract(c)
        if (minimum != 1.5 or maximum != 12. or c['target_force_n'] != 20.
                or c['actor_kind'] != 'time_preview_residence'
                or c['source_path_sha256'] != hashlib.sha256(reference.read_bytes()).hexdigest()):
            raise ValueError('Fixed-force checkpoint/runtime/reference mismatch')
        self.geometry = FixedForceGeometry(reference)
        self.surface = self.geometry.surface
        self.reset(0)

    def reset(self, epoch):
        self.epoch = epoch;self.sequence = 0;self.elapsed = 0.;self.previous_position = None
        self.surface.reset();self.residence = np.zeros_like(self.surface.depth)
        self.action = np.array([0.], dtype=np.float32)
        self.actor_action = self.action.copy()
        self.observation = np.zeros(16, dtype=np.float32)
        self.samples = 0;self.decisions = 0;self.missing_samples = 0

    def process(self, values):
        v = np.asarray(values, dtype=float)
        if v.shape != (23,) or not np.isfinite(v).all() or v[0] != 2.:
            raise ValueError('Expected finite 23D fixed-force telemetry schema 2')
        epoch, sequence = int(v[1]), int(v[2])
        if epoch < 1 or sequence < 1 or v[1] != epoch or v[2] != sequence:
            raise ValueError('Invalid trajectory epoch/sequence')
        if epoch < self.epoch:
            raise ValueError('Old trajectory telemetry rejected')
        if epoch != self.epoch:
            self.reset(epoch)
        if sequence <= self.sequence:
            raise ValueError('Repeated or reordered telemetry rejected')
        if abs(v[5]-self.geometry.length) > .05:
            raise ValueError('Runtime path length differs from trained path')
        if v[12] != 0:
            raise ValueError(f'Controller latched fault {v[12]}')
        if v[8] != 20. or v[7] != 0. or v[9] != 0.:
            raise ValueError('Force target or forward-only contract violated')
        if self.sequence == 0 and (sequence != 1 or abs(v[10]-.008)>1.e-8):
            raise ValueError('Missed trajectory start; restart launch')
        if self.sequence and v[4] < self.cursor-1.e-6:
            raise ValueError('Backward cursor forbidden')
        dt = .008 if self.sequence == 0 else v[10]-self.elapsed
        gap = 0 if self.sequence == 0 else sequence-self.sequence-1
        if not 0 < dt <= .25 or abs(dt/.008-round(dt/.008))>1.e-3:
            raise ValueError('Missing or inconsistent simulation ticks')
        if gap > 0:
            # Recover travelled distance across short transport gaps; mark
            # them in the log rather than pretending all samples arrived.
            self.missing_samples += gap
        position = v[15:18].astype(np.float32)
        previous = self.previous_position
        self.previous_position = position.copy()
        delta = np.zeros(3) if previous is None else position-previous
        delta[2] = 0.  # This frozen checkpoint/reference is planar XY.
        norm = np.linalg.norm(delta)
        speed = v[13] if gap == 0 else norm/dt
        velocity = delta/max(norm,1.e-12)*np.float32(speed)
        midpoint = position-.5*delta
        contact = v[21] >= 1.5
        self.surface.deposit(midpoint, velocity, v[21], dt, contact=contact)
        if contact:
            center = (midpoint-self.surface.origin) @ self.surface.basis.T
            rows, cols, _, _, weights, _ = self.surface._footprint(center)
            self.residence[rows,cols] += dt*(weights>0)
        g = self.geometry
        features = g.features(self.surface.depth.ravel()[g.roi_flat][None,:],
            self.residence.ravel()[g.roi_flat][None,:], np.array([v[4]]),np.array([v[6]]),
            np.array([v[21]]),np.array([v[13]]),np.array([v[10]]),np.array([v[7]]),
            np.array([v[8]]),np.array([v[9]]),tracking=np.array([v[14]]),shield=np.array([v[11]]))
        self.observation = features[0]
        # Decisions every ten control ticks; republish the held decision on
        # every NEW measurement, never with a free-running wall heartbeat.
        decision = self.sequence == 0 or (sequence-1)//10 != (self.sequence-1)//10
        if decision:
            with torch.inference_mode():
                self.action = self.actor(torch.from_numpy(features))[0].numpy().copy()
            if self.action.shape != (1,) or not np.isfinite(self.action).all() or np.max(np.abs(self.action))>1.000001:
                raise ValueError('Invalid spatial actor output')
            self.actor_action = np.clip(self.action,-1.,1.)
            # Run identical feature extraction/inference in both modes. Only
            # the applied output is replaced in the matched constant arm.
            self.action = self.actor_action.copy() if self.force_control_mode == 5 else np.array([0.],dtype=np.float32)
            self.decisions += 1
        self.sequence = sequence;self.elapsed = v[10];self.cursor = v[4];self.samples += 1
        return {'action':[epoch,sequence,*self.action.tolist()], 'observation':self.observation.tolist(),
                'decision':decision,'missing_samples':self.missing_samples,
                'actor_action':self.actor_action.tolist()}
