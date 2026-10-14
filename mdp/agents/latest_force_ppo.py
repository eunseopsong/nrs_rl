"""PPO contract for the preserved merged force controller; legacy saves stay valid."""
import copy
import json
import torch
from .ppo_network import PPOActorCritic, DeterministicPPOActor
from ..removal_config import CONTRACT as LEGACY, check_ppo_or_baseline
from ...utils.removal_metrics import sha

CONTRACT = copy.deepcopy(LEGACY)
CONTRACT.update(
    backend='ppo_latest_force_spatial_v1',
    tracking_compensation={'gain': 0., 'limit_mm': 4., 'tau_s': .08},
    training_environment='Isaac physics with measured TCP rotary removal and preserved merged force control',
    force_control_profile={'tangent_mass_kg': 2., 'tangent_damping_ns_m': 6000.,
        'tangent_stiffness_n_m': 2000., 'tracking_compensation': False,
        'mode3_core_sha256': '7ed0a2714ec7ba973564931b2a70de80e1887869fbcd4e732082f82b86036f04'})

def check_contract(contract):
    if contract.get('backend') != CONTRACT['backend']:
        return check_ppo_or_baseline(contract)
    for key, value in CONTRACT.items():
        if contract.get(key) != value:
            raise ValueError('Latest-force PPO contract mismatch: ' + key)
    return 1.5, 12.

def export_actor(model, path, metadata):
    contract = {**CONTRACT, **metadata}
    check_contract(contract)
    actor = DeterministicPPOActor(model.actor)
    scripted = torch.jit.script(actor)
    torch.jit.save(scripted, str(path), _extra_files={'fixed_force_policy.json': json.dumps(contract)})
    return sha(path)
