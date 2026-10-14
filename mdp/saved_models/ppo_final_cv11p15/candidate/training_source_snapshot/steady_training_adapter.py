"""Study-private PPO environment adapter for the frozen gain-0.2 controller."""
from contextlib import contextmanager
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys

STUDY = Path(__file__).resolve().parent
ROOT = Path('/home/eunseop/nrs_rl')
PROTOCOL = STUDY / 'protocol.json'
SETTINGS = json.loads(PROTOCOL.read_text())
# PARENT remains the identical physical controller source. Learning resumes a
# separate saved actor/critic/Adam bundle; these two provenances must not mix.
PARENT = Path(SETTINGS['controller_parent_bundle'])
LEARNING_PARENT = Path(SETTINGS['reference_bundle'])
MANIFEST = STUDY / 'training_adapter_manifest.json'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_frozen_inputs():
    manifest = read(MANIFEST)
    for path, digest in manifest['frozen_source_sha256'].items():
        if sha(path) != digest:
            raise ValueError('Frozen training adapter input changed: ' + path)
    for name, digest in read(PARENT / 'manifest.json').items():
        if sha(PARENT / name) != digest:
            raise ValueError('Saved 10.52% bundle changed: ' + name)
    for name, digest in read(LEARNING_PARENT.parent / 'SAVE.json')['files_sha256'].items():
        if sha(LEARNING_PARENT.parent / name) != digest:
            raise ValueError('Saved 11.01% learning input changed: ' + name)
    return manifest


def controller_metadata():
    return {'controller_experiment': read(PARENT / 'runtime_requirement.json')['controller_contract'],
            'requires_experimental_steady_v3_gain': .2, 'ros_deployment_validated': False,
            'private_controller_training': True,
            'training_controller_scope': 'PPO actor/critic trained with fixed private v3 gain0.2; original core and transition untouched',
            'controller_source_sha256': verify_frozen_inputs()['frozen_source_sha256'],
            'training_adapter_manifest': str(MANIFEST), 'training_adapter_manifest_sha256': sha(MANIFEST),
            'controller_parent_bundle': str(PARENT),
            'learning_parent_bundle': str(LEARNING_PARENT),
            'learning_parent_actor': SETTINGS['reference_actor'],
            'learning_parent_actor_sha256': SETTINGS['reference_actor_sha256'],
            'learning_parent_training_state': SETTINGS['reference_training_state'],
            'learning_parent_training_state_sha256': SETTINGS['reference_training_state_sha256'],
            'original_cv_reward_unchanged': True,
            'volume_constraint_location': SETTINGS['volume_constraint_location']}


def validate_environment_args(native, controller_snapshot, gui_scene, nominal_sensors, reward_cell_mm):
    verify_frozen_inputs()
    parent = read(PARENT / 'protocol.json')
    for value, key in [(native, 'native_dir'), (controller_snapshot, 'controller_source'), (gui_scene, 'scene_profile')]:
        if value is None or Path(value).resolve() != Path(parent[key]).resolve():
            raise ValueError('Use frozen steady-controller input: ' + key)
    if not nominal_sensors or reward_cell_mm != .5:
        raise ValueError('Steady training requires nominal sensors and 0.5mm reward geometry')


def prepare_training_args(args):
    validate_environment_args(args.native_dir, args.control_snapshot, args.gui_scene, args.nominal_sensors, args.reward_cell_mm)
    if args.reset_critic_output or args.critic_warmup_updates != 0 or not args.critic_potential_residual:
        raise ValueError('Exact continuation requires residual critic, no output reset and no warmup')
    if args.exploration_log_std is not None:
        raise ValueError('Preserve the saved learned log_std exactly')
    if (Path(args.resume).resolve() != Path(SETTINGS['reference_training_state']).resolve()
            or sha(args.resume) != SETTINGS['reference_training_state_sha256']):
        raise ValueError('This final attempt must resume the saved 11.01% training state')
    for key, expected in SETTINGS['training_common'].items():
        if key != 'candidate_rollout_batches' and getattr(args, key) != expected:
            raise ValueError('Frozen learning argument mismatch: ' + key)
    for key, expected in [('seed', SETTINGS['training_seed']),
                          ('exploration_rho', SETTINGS['exploration_rho']),
                          ('entropy_coefficient', .0002), ('device', 'cpu')]:
        if getattr(args, key) != expected:
            raise ValueError('Frozen continuation argument mismatch: ' + key)
    if (args.reward_forecast is None
            or Path(args.reward_forecast).resolve() != Path(SETTINGS['reward_forecast']).resolve()
            or sha(args.reward_forecast) != SETTINGS['reward_forecast_sha256']):
        raise ValueError('Preserve the original same-controller CV forecast')
    import torch
    parent = torch.load(args.resume, map_location='cpu', weights_only=False)
    protocol = parent['protocol']
    if (protocol.get('critic_potential_residual') is not True
            or protocol.get('controller_experiment') != SETTINGS['controller_contract']
            or protocol.get('requires_experimental_steady_v3_gain') != .2
            or protocol.get('reward_cell_mm') != args.reward_cell_mm
            or protocol.get('nominal_sensors') is not True):
        raise ValueError('Parent critic must have the same controller and residual-value semantics')


def exact_tree_equal(left, right):
    """Compare actual model/Adam tensors and scalar settings, without mutation."""
    import torch
    if isinstance(left, torch.Tensor):
        return (isinstance(right, torch.Tensor) and left.dtype == right.dtype
                and left.shape == right.shape and torch.equal(left.cpu(), right.cpu()))
    if isinstance(left, dict):
        return (isinstance(right, dict) and left.keys() == right.keys()
                and all(exact_tree_equal(left[key], right[key]) for key in left))
    if isinstance(left, (list, tuple)):
        return (type(left) is type(right) and len(left) == len(right)
                and all(exact_tree_equal(a, b) for a, b in zip(left, right)))
    return type(left) is type(right) and left == right


def snapshot_initial_training_state(model, optimizer, output, args, parent):
    """Audit and save the actual initialized state before export/rollout/training."""
    import torch
    saved = torch.load(args.resume, map_location='cpu', weights_only=False)
    current = model.state_dict()
    checks = {prefix.rstrip('.') + '_exact':
              exact_tree_equal({k: v for k, v in current.items() if k.startswith(prefix)},
                               {k: v for k, v in saved['model'].items() if k.startswith(prefix)})
              for prefix in ['actor.', 'critic.', 'log_std']}
    checks['optimizer_exact'] = exact_tree_equal(optimizer.state_dict(), saved['optimizer'])
    checks['model_exact'] = exact_tree_equal(current, saved['model'])
    if not all(checks.values()):
        raise ValueError('Initial saved-state continuation mismatch: ' + str(checks))
    raw = Path(output) / 'checkpoints/initial_training_state.pt'
    torch.save({'model': current, 'optimizer': optimizer.state_dict(), 'update': 0,
                'parent_training': parent, 'parent_protocol': saved['protocol'],
                'torch_rng_state': torch.get_rng_state()}, raw)
    audit = {'passed': True, **checks, 'parent_training_state': str(Path(args.resume).resolve()),
             'parent_training_state_sha256': sha(args.resume),
             'initial_training_state': str(raw), 'initial_training_state_sha256': sha(raw),
             'log_std': float(model.log_std.detach()), 'critic_reset': False, 'warmup_updates': 0,
             'parent_rollout_batches': int(saved['update']),
             'new_rollout_rng': True, 'scope': 'Actual initialization before any PPO optimizer step'}
    (Path(output) / 'initial_resume_audit.json').write_text(json.dumps(audit, indent=2) + '\n')


def training_metadata(args):
    proof = Path(args.output).resolve() / 'initial_resume_audit.json'
    initial = read(proof)
    return {**controller_metadata(), 'policy_scope': 'Neural PPO actor and critic only; fixed experimental v3 gain0.2 is not tuned',
            'initial_resume_audit': str(proof), 'initial_resume_audit_sha256': sha(proof),
            'initial_training_state': initial['initial_training_state'],
            'initial_training_state_sha256': initial['initial_training_state_sha256'],
            'training_resume_contract': SETTINGS['training_resume_contract'],
            'critic_controller_reinitialization': {'output_reset': args.reset_critic_output,
                'warmup_updates': args.critic_warmup_updates, 'residual_initial_bias': None,
                'reason': 'Same fixed v3 dynamics and residual critic semantics; exact continuation'}}


def decorate_actor_export(original):
    def export(model, path, metadata):
        # Extra contract metadata does not change network tensors or action mapping.
        return original(model, path, {**metadata, **controller_metadata()})
    return export


def snapshot_private_sources(destination):
    for name in ['steady_training_adapter.py', 'train_steady_v3_ppo.py', 'training_adapter_manifest.json']:
        shutil.copy2(STUDY / name, destination / name)
    shutil.copy2(PARENT / 'steady_postcore_v3.py', destination / 'steady_postcore_v3.py')
    shutil.copy2(STUDY / 'original_train_latest_force_ppo.py', destination / 'original_train_latest_force_ppo.py')
    shutil.copy2(PROTOCOL, destination / 'continuation_study_protocol.json')


def live_context(term, index):
    # Same action snapshots and decision order as the frozen v3 evaluation probe.
    # In training, completed episodes reset immediately. path_done is cleared by
    # the original action reset; no run-global ended latch may leak into the next episode.
    return {'fault': bool(term.safety_terminated[index]) or bool(term.safety_fault_reason[index]),
            'approach': bool(term._approach_step[index] < term._approach_steps),
            'progression_allowed': bool(term.polishing_active[index]) and not bool(term.path_done[index]),
            'normal_force_n': float(term.current_abs_fz[index]),
            'tracking_error_mm': float(term.current_path_tracking_error_mm[index])}


@contextmanager
def wrapped_native_factory(native_module, wrapper_type, term_supplier, controllers):
    """Change only construction; restore the module symbol even if construction fails."""
    original = native_module.TangentialMode3
    def construct(*args, **kwargs):
        index = len(controllers)
        controller = wrapper_type(original(*args, **kwargs), .2)
        controller.bind_context(lambda index=index: live_context(term_supplier(), index))
        controllers.append(controller)
        return controller
    native_module.TangentialMode3 = construct
    try:
        yield
    finally:
        native_module.TangentialMode3 = original


def make_env_class(base_class):
    class IsaacSteadyForceEnv(base_class):
        def __init__(self, count, seed, native, controller_snapshot, output, reward_cell_mm=.5,
                     gui_scene=None, reward_forecast=None, nominal_sensors=False):
            validate_environment_args(native, controller_snapshot, gui_scene, nominal_sensors, reward_cell_mm)
            sys.path.insert(0, str(Path(native).resolve()))
            sys.path.insert(0, str(PARENT))
            native_module = importlib.import_module('_latest_force_mode3')
            wrapper_type = importlib.import_module('steady_postcore_v3').SteadyPostCore
            if Path(sys.modules['steady_postcore_v3'].__file__).resolve() != (PARENT / 'steady_postcore_v3.py').resolve():
                raise ValueError('Another steady wrapper module was already imported')
            self.steady_controllers = []
            # Callbacks are bound at wrapper construction, before base env.reset.
            # Base assigns self.term before that reset. Reset forwards exactly once.
            with wrapped_native_factory(native_module, wrapper_type, lambda: self.term, self.steady_controllers):
                super().__init__(count, seed, native, controller_snapshot, output, reward_cell_mm,
                                 gui_scene, reward_forecast, nominal_sensors)
            if len(self.steady_controllers) != count or any(a is not b for a, b in
                    zip(self.steady_controllers, self.term.force_controllers)):
                raise ValueError('Not every training controller was wrapped exactly once')
            cfg = self.term.int_cfg
            if (cfg.force_overload_ratio, cfg.tracking_stop_mm, cfg.contact_force_n) != (1.6, 10., 1.5):
                raise ValueError('Original same-tick guard thresholds changed')
            if cfg.spindle_rpm is not None:
                raise ValueError('Physical spindle rotation must remain disabled')
            (Path(output) / 'controller_experiment.json').write_text(json.dumps(controller_metadata(), indent=2) + '\n')

        def close(self):
            try:
                return super().close()
            finally:
                verify_frozen_inputs()
    return IsaacSteadyForceEnv
