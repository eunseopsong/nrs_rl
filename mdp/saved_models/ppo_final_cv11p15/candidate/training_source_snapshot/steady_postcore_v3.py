"""Study-private post-core tangent correction; the shared core is never modified.

The gate controls only this extra correction. It is not a replacement contact
transition. All core arguments, call counts, internal history and reset calls
remain original. Hard gates return the entire original output. After initial
qualification, loss of the steady-force band only decays the existing offset.
"""
from dataclasses import asdict, dataclass
import math
import numpy as np


@dataclass(frozen=True)
class SteadyConfig:
    gain: float = .2
    limit_mm: float = 1.5
    tau_s: float = .08
    dt_s: float = .008
    first_force_s: float = 2.
    stable_s: float = .5
    stable_min_n: float = 18.
    stable_max_n: float = 22.
    contact_n: float = 1.5
    desired_threshold_n: float = .01
    overload_ratio: float = 1.6
    tracking_stop_mm: float = 10.
    release_epsilon_mm: float = 1.e-6

    def __post_init__(self):
        expected = SteadyConfig.__dataclass_fields__
        for name, value in asdict(self).items():
            if not math.isfinite(value):
                raise ValueError('Nonfinite experimental parameter: ' + name)
            if name == 'gain':
                if value not in (0., .2):
                    raise ValueError('Only frozen comparison gains0 and0.2 are supported')
            elif value != expected[name].default:
                raise ValueError('Frozen study parameter changed: ' + name)


REASONS = {0: 'active', 1: 'first_force_interval', 2: 'force_not_stable',
           3: 'force_request_or_release', 4: 'contact_loss', 5: 'approach',
           6: 'existing_fault', 7: 'progression_disabled', 8: 'current_tick_shield',
           9: 'invalid_geometry', 10: 'zero_gain'}


class SteadyPostCore:
    def __init__(self, core, gain=.2):
        self.core = core
        self.config = SteadyConfig(gain=float(gain))
        self.context = None
        self.correction = np.zeros(3)
        self.epoch_ticks = self.stable_ticks = 0
        self.total_steps = self.total_active_steps = self.total_resets = 0
        self.maximum_correction_mm = self.maximum_normal_component_mm = 0.
        self.maximum_disable_drop_mm = 0.
        self.soft_release_steps = 0
        self.soft_release_step_mm = self.soft_projection_adjustment_mm = 0.
        self.maximum_soft_release_step_mm = 0.
        self.reason_counts = {reason: 0 for reason in REASONS.values()}
        self.raw_last = None
        self.gate_force_n = self.gate_tracking_mm = 0.
        self.last = self._record(5, 0., 0.)

    def bind_context(self, supplier):
        """Read-only current action/episode context; not an actuator callback."""
        self.context = supplier

    def _record(self, reason, normal_component, disable_drop):
        return {'active': reason == 0, 'eligible': reason == 0,
                'applied': bool(reason == 0 or reason == 2 and self.soft_release_step_mm > 0.),
                'releasing': bool(reason == 2 and self.soft_release_step_mm > 0.),
                'soft_releasing': bool(reason == 2 and self.soft_release_step_mm > 0.),
                'soft_release_step_mm': self.soft_release_step_mm,
                'soft_projection_adjustment_mm': self.soft_projection_adjustment_mm,
                'reason': reason,
                'force_epoch_s': self.epoch_ticks * self.config.dt_s,
                'stable_force_s': self.stable_ticks * self.config.dt_s,
                'correction_mm': self.correction.copy(),
                'gate_force_n': self.gate_force_n,
                'gate_tracking_mm': self.gate_tracking_mm,
                'normal_component_mm': float(normal_component),
                'disable_drop_mm': float(disable_drop)}

    def reset(self, pose):
        # Forward exactly the caller's original reset, once; no rebase or setter.
        result = self.core.reset(pose)
        self.correction.fill(0.)
        self.epoch_ticks = self.stable_ticks = 0
        self.total_resets += 1
        self.raw_last = None
        self.gate_force_n = self.gate_tracking_mm = 0.
        self.soft_release_step_mm = self.soft_projection_adjustment_mm = 0.
        self.last = self._record(5, 0., 0.)
        return result

    def telemetry(self):
        return {key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in self.last.items()}

    def summary(self):
        return {'config': asdict(self.config), 'total_core_steps': self.total_steps,
                'active_steps': self.total_active_steps, 'existing_resets_forwarded': self.total_resets,
                'maximum_correction_mm': self.maximum_correction_mm,
                'maximum_normal_component_mm': self.maximum_normal_component_mm,
                'maximum_disable_drop_mm': self.maximum_disable_drop_mm,
                'soft_release_steps': self.soft_release_steps,
                'maximum_soft_release_step_mm': self.maximum_soft_release_step_mm,
                'reason_counts': dict(self.reason_counts),
                'core_arguments_modified': False, 'core_history_written': False}

    def step(self, measured, reference, force, wrench, rotation):
        # This call precedes wrapper arithmetic and receives the same objects.
        raw = self.core.step(measured, reference, force, wrench, rotation)
        self.raw_last = tuple(raw)
        self.total_steps += 1
        self.soft_release_step_mm = self.soft_projection_adjustment_mm = 0.
        self.epoch_ticks += 1
        cfg = self.config
        context = self.context() if self.context is not None else {'progression_allowed': False}
        pose, target = np.asarray(measured, dtype=float), np.asarray(reference, dtype=float)
        requested_force = np.asarray(force, dtype=float)
        measured_wrench = np.asarray(wrench, dtype=float)
        matrix = np.asarray(rotation, dtype=float).reshape(3, 3)
        raw_normal = matrix[:, 2]
        geometry_ok = (np.isfinite(pose).all() and np.isfinite(target).all()
                       and np.isfinite(requested_force).all() and np.isfinite(measured_wrench).all()
                       and np.isfinite(matrix).all()
                       and np.allclose(matrix.T @ matrix, np.eye(3), atol=1.e-5, rtol=0.)
                       and np.isclose(np.linalg.det(matrix), 1., atol=1.e-5, rtol=0.))
        # action.py promotes an originally float32 rotation to double for the
        # native call. Normalize only our private projection basis; never the
        # native inputs or the original controller's force measurement.
        normal = raw_normal / np.linalg.norm(raw_normal) if geometry_ok else raw_normal
        fn = abs(float(context.get('normal_force_n', raw_normal @ measured_wrench[:3]))) if geometry_ok else float('nan')
        tracking = float(context.get('tracking_error_mm', np.linalg.norm(target[:3] - pose[:3]))) if geometry_ok else float('inf')
        geometry_ok = geometry_ok and math.isfinite(fn) and math.isfinite(tracking)
        self.gate_force_n, self.gate_tracking_mm = fn, tracking
        normal_request = abs(float(requested_force[2]))
        fixed_request = (np.all(requested_force[:2] == 0.) and requested_force[2] == 20.)
        current_shield = (normal_request > 0. and fn > cfg.overload_ratio * normal_request
                          or tracking > cfg.tracking_stop_mm)
        reason = 0
        if not geometry_ok:
            reason = 9
        elif context.get('fault', False):
            reason = 6
        elif context.get('approach', True):
            reason = 5
        elif not context.get('progression_allowed', False):
            reason = 7
        elif not fixed_request or normal_request <= cfg.desired_threshold_n:
            reason = 3
        elif fn <= cfg.contact_n:
            reason = 4
        elif current_shield:
            reason = 8
        if reason in (3, 4, 5, 6, 7, 9):
            self.epoch_ticks = 0
        if reason == 0 and cfg.stable_min_n <= fn <= cfg.stable_max_n:
            self.stable_ticks += 1
        else:
            self.stable_ticks = 0
        if reason == 0 and self.epoch_ticks * cfg.dt_s <= cfg.first_force_s:
            reason = 1
        if reason == 0 and self.stable_ticks * cfg.dt_s + 1.e-12 < cfg.stable_s:
            reason = 2
        if cfg.gain == 0.:
            reason = 10
        soft_release = reason == 2 and bool(np.any(self.correction != 0.))
        if reason and not soft_release:
            drop = float(np.linalg.norm(self.correction))
            self.maximum_disable_drop_mm = max(self.maximum_disable_drop_mm, drop)
            self.correction.fill(0.)
            self.last = self._record(reason, 0., drop)
            self.reason_counts[REASONS[reason]] += 1
            return raw
        old_correction = self.correction.copy()
        if soft_release:
            # Never track a new error while unqualified. Only decay the offset
            # previously earned through the unchanged initial steady gate.
            self.correction *= math.exp(-cfg.dt_s / cfg.tau_s)
        else:
            error = target[:3] - pose[:3]
            tangent = error - normal * float(error @ normal)
            desired = cfg.gain * tangent
            desired *= min(1., cfg.limit_mm / max(float(np.linalg.norm(desired)), 1.e-12))
            self.correction += -math.expm1(-cfg.dt_s / cfg.tau_s) * (desired - self.correction)
        before_projection = self.correction.copy()
        self.correction -= normal * float(self.correction @ normal)
        self.correction *= min(1., cfg.limit_mm / max(float(np.linalg.norm(self.correction)), 1.e-12))
        if soft_release:
            self.soft_projection_adjustment_mm = float(np.linalg.norm(self.correction - before_projection))
            if np.linalg.norm(self.correction) <= cfg.release_epsilon_mm:
                self.correction.fill(0.)
            self.soft_release_step_mm = float(np.linalg.norm(self.correction - old_correction))
            self.maximum_soft_release_step_mm = max(self.maximum_soft_release_step_mm, self.soft_release_step_mm)
            self.soft_release_steps += 1
        output = list(raw)
        for axis in range(3):
            output[axis] += float(self.correction[axis])
        self.total_active_steps += int(reason == 0)
        magnitude = float(np.linalg.norm(self.correction))
        normal_component = abs(float(self.correction @ normal))
        self.maximum_correction_mm = max(self.maximum_correction_mm, magnitude)
        self.maximum_normal_component_mm = max(self.maximum_normal_component_mm, normal_component)
        self.reason_counts[REASONS[reason]] += 1
        self.last = self._record(reason, normal_component, 0.)
        return output if self.last['applied'] else raw


def experimental_contract(gain):
    config = SteadyConfig(gain=float(gain))
    return {'schema': 'study_private_steady_postcore_v3', 'experimental': True,
        'controller_variant': 'unchanged_core' if gain == 0. else 'steady_postcore_gain0p2_soft_release',
        'normal_force_core': 'original latest force Mode3ForceControlCore',
        'position_to_force_logic_changed': False, 'core_initial_mdk_changed': False,
        'correction_position': 'after original core step, before original IK and guards',
        'correction_coordinates': 'instantaneous TCP tangent plane; not unqualified baseXY',
        'projection_basis': 'private unit TCP normal after proper-rotation validation; original core inputs untouched',
        'gate_measurement': 'same-tick action force/tracking snapshot; raw input arithmetic only for standalone replay',
        'disabled_output': 'hard gates and zero correction return original entire core output object',
        'soft_release': 'reason2 only: old correction decays with tau0.08, no new error correction; epsilon1e-6mm',
        'soft_release_step_bound': 'old_norm*(1-exp(-dt/tau))+epsilon plus measured tangent reprojection adjustment',
        'telemetry_semantics': 'eligible/active qualify new tracking; releasing/soft_releasing decay old tracking; applied=active|releasing',
        'core_input_or_history_changed': False,
        'gate_scope': 'extra correction eligibility only; original mode/contact switches remain unchanged',
        'variant_applied_identically_to_all_candidates': True,
        'baseline_comparison': 'PPO versus constant Mode6 under this same controller variant',
        'deployment_status': 'not installed or connected to ROS', 'config': asdict(config)}
