"""Optional action holds with PPO timing expressed in 125 Hz control ticks."""

import copy
import math

import gymnasium as gym
import torch


def ppo_config_for_action_repeat(config: dict, repeat: int) -> dict:
    """Convert a 125 Hz PPO configuration to decisions spanning ``repeat`` ticks.

    Gamma/lambda preserve their physical time constants. Sample and logging
    counts round up to whole decisions; disabled logging intervals stay disabled.
    The input remains unchanged so repeated configuration cannot compound scales.
    """
    if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 1:
        raise ValueError("policy action repeat must be a positive integer")
    result = copy.deepcopy(config)
    agent = result["agent"]
    for name in ("discount_factor", "lambda"):
        agent[name] = agent[name] ** repeat
    for owner, name in ((agent, "rollouts"), (result["trainer"], "timesteps")):
        if isinstance(owner[name], bool) or not isinstance(owner[name], int) or owner[name] < 1:
            raise ValueError(f"{name} must be a positive number of control ticks")
        owner[name] = math.ceil(owner[name] / repeat)
    for name in ("write_interval", "checkpoint_interval"):
        value = agent.get("experiment", {}).get(name)
        if isinstance(value, int) and value > 0:
            agent["experiment"][name] = math.ceil(value / repeat)
    return result


class ControlActionRepeat(gym.Wrapper):
    """Aggregate every raw control reward, including stops and contact entry.

    Isaac autoresets individual environments inside step(). Once one finishes,
    its remaining rewards in this decision are masked and its command is zero
    during the new calibration phase. Healthy environments run the full hold.
    The returned observation belongs to the actual current autoreset state.
    """

    def __init__(self, env, repeat: int, decision_discount: float):
        super().__init__(env)
        if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 1:
            raise ValueError("policy action repeat must be a positive integer")
        self.repeat = repeat
        self.control_discount = decision_discount ** (1.0 / repeat)
        self.decisions = 0
        self.completed_episodes = 0
        self.terminated_episodes = 0
        self.truncated_episodes = 0
        self.path_completions = 0
        self.control_failures = 0

    def step(self, actions):
        total_reward = finished = any_terminated = any_truncated = None
        decision_info = {}
        held_action = actions.clone()
        for tick in range(self.repeat):
            if finished is not None:
                held_action[finished] = 0.0
            observation, reward, terminated, truncated, info = self.env.step(held_action)
            if total_reward is None:
                total_reward = torch.zeros_like(reward)
                finished = torch.zeros_like(terminated, dtype=torch.bool)
                any_terminated = torch.zeros_like(finished)
                any_truncated = torch.zeros_like(finished)
            active = ~finished
            total_reward += self.control_discount ** tick * reward * active.to(reward.dtype)
            any_terminated |= terminated & active
            any_truncated |= truncated & active
            new_done = (terminated | truncated) & active
            if new_done.any():
                self.completed_episodes += int(new_done.sum())
                self.terminated_episodes += int((terminated & active).sum())
                self.truncated_episodes += int((truncated & active).sum())
                manager = getattr(self.unwrapped, "termination_manager", None)
                if manager is not None:
                    self.path_completions += int((manager.get_term("trajectory_finished") & new_done).sum())
                    self.control_failures += int((manager.get_term("control_failed") & new_done).sum())
                decision_info.update(info)
            finished |= new_done
        self.decisions += 1
        # Preserve an early substep's episode log across the remaining ticks.
        info = {**info, **decision_info}
        return observation, total_reward, any_terminated, any_truncated, info

    def coverage(self):
        return {
            "policy_decisions_per_environment": self.decisions,
            "control_steps_per_environment": self.decisions * self.repeat,
            "policy_action_repeat": self.repeat,
            "finished_episodes": self.completed_episodes,
            "terminated_episodes": self.terminated_episodes,
            "truncated_episodes": self.truncated_episodes,
            "path_completions": self.path_completions,
            "control_failures": self.control_failures,
        }
