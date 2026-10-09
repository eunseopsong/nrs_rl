"""Polishing task: neural RL, model-based policies, and shared process utilities."""

def register_envs():
    import gymnasium as gym
    task_id = 'Template-Nrs-Rl-v0'
    if task_id not in gym.registry:
        gym.register(id=task_id, entry_point='isaaclab.envs:ManagerBasedRLEnv',
            disable_env_checker=True, kwargs={
                'env_cfg_entry_point': f'{__name__}.mdp.env_cfg:NrsRlEnvCfg',
                'skrl_cfg_entry_point': f'{__name__}.mdp.agents:skrl_ppo_cfg.yaml'})
