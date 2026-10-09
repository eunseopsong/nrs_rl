"""Public PPO API. Policy, environment and reward implementations are separated."""
if not __package__:
    import sys
    from pathlib import Path
    _root = next(p / 'source/nrs_rl' for p in Path(__file__).resolve().parents
                 if (p / 'source/nrs_rl/nrs_rl').is_dir())
    sys.path.insert(0, str(_root))
from nrs_rl.tasks.manager_based.nrs_rl.agents.ppo_network import PolicyNetwork, PPOActorCritic, DeterministicPPOActor, export_actor
from nrs_rl.tasks.manager_based.nrs_rl.mdp.removal_env import RemovalEnv
from nrs_rl.tasks.manager_based.nrs_rl.mdp.removal_config import CONTRACT, check_ppo_or_baseline
from nrs_rl.tasks.manager_based.nrs_rl.utils.removal_metrics import sha

__all__ = ['PolicyNetwork', 'PPOActorCritic', 'DeterministicPPOActor', 'export_actor',
           'RemovalEnv', 'CONTRACT', 'check_ppo_or_baseline', 'sha']
