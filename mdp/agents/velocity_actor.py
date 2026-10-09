"""Portable deterministic SB3 actor, with its observation normalization embedded.

The deployed worker needs Torch only. The complete training state remains in
the SB3 zip and VecNormalize pickle; this artifact is inference-only.
"""
from __future__ import annotations
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))


import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


class NormalizedVelocityActor(torch.nn.Module):
    def __init__(self, policy, normalizer):
        super().__init__()
        self.register_buffer("obs_mean", torch.as_tensor(normalizer.obs_rms.mean.copy(), dtype=torch.float64))
        self.register_buffer("obs_var", torch.as_tensor(normalizer.obs_rms.var.copy(), dtype=torch.float64))
        self.register_buffer("obs_count", torch.tensor(float(normalizer.obs_rms.count), dtype=torch.float64))
        self.epsilon = float(normalizer.epsilon)
        self.clip_obs = float(normalizer.clip_obs)
        self.actor = copy.deepcopy(policy.mlp_extractor.policy_net).cpu().eval()
        self.action = copy.deepcopy(policy.action_net).cpu().eval()

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        # SB3 normalizes with float64 running statistics, then casts to float32.
        state = (observation.double() - self.obs_mean) / torch.sqrt(self.obs_var + self.epsilon)
        state = state.clamp(-self.clip_obs, self.clip_obs).float()
        return self.action(self.actor(state))


def export_actor(policy, normalizer, path, contract, probe_observations):
    """Save a raw-mean actor and verify it against the actual training policy."""
    from stable_baselines3.common.torch_layers import FlattenExtractor
    if policy.squash_output or not isinstance(policy.features_extractor, FlattenExtractor):
        raise ValueError("Export supports unsquashed vector-input MLP policies only")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    observations = np.asarray(probe_observations, dtype=np.float32)
    with torch.inference_mode():
        normalized = torch.as_tensor(normalizer.normalize_obs(observations.copy()), device=policy.device)
        expected = policy.get_distribution(normalized).get_actions(deterministic=True).cpu()
        module = torch.jit.script(NormalizedVelocityActor(policy, normalizer).eval())
        actual = module(torch.as_tensor(observations))
        error = float((actual - expected).abs().max())
        if not bool(torch.isfinite(actual).all()) or error > 2e-6:
            raise ValueError(f"Exported actor differs from training inference: maximum error {error}")
    torch.jit.save(module, str(path), _extra_files={"velocity_policy.json": json.dumps(contract, sort_keys=True)})
    return {"observations": len(observations), "maximum_mean_action_error": error,
            "normalizer_count": float(normalizer.obs_rms.count)}


class TorchscriptVelocityAgent:
    """Small adapter for the existing deterministic evaluation/worker interface."""
    normalizer_name = "EmbeddedVecNormalize"

    def __init__(self, checkpoint, contract):
        torch.set_num_threads(1)
        extra = {"velocity_policy.json": ""}
        self.actor = torch.jit.load(str(checkpoint), map_location="cpu", _extra_files=extra).eval()
        if not extra["velocity_policy.json"]:
            raise ValueError("Exported actor has no embedded velocity contract")
        embedded = json.loads(extra["velocity_policy.json"])
        if any(contract.get(key) != value for key, value in embedded.items()):
            raise ValueError("Exported actor and sidecar velocity contracts differ")
        if embedded.get("policy_backend") not in ("torchscript_sb3_ppo", "torchscript_preston_depth", "torchscript_tcp_removal"):
            raise ValueError("Unexpected exported actor backend")
        if embedded['policy_backend'] in ('torchscript_preston_depth', 'torchscript_tcp_removal'):
            self.normalizer_name = 'PhysicalScaling'
        self.velocity_contract = contract
        self._state_preprocessor = SimpleNamespace(current_count=self.actor.obs_count)

    def act(self, observations, timestep=0, timesteps=0):
        mean = self.actor(observations)
        return mean, None, {"mean_actions": mean}
