"""Spatial final-depth CV potential and Mode 6 removal-floor penalties."""
import torch

def predicted_cv2(env):
    remaining = env.remaining[env.index(env.cursor)]
    remaining = torch.where((env.cursor >= env.length)[:, None], torch.zeros_like(remaining), remaining)
    values = (env.processing+remaining)[:, env.mask]
    return values.square().mean(1)/values.mean(1).clamp_min(1.e-6).square()-1.


def step_reward(previous_potential, potential, action, previous_action, dt=.08):
    return 100.*(previous_potential-potential)-1.e-5*(action-previous_action).square()-.0001*dt


def terminal_penalty(done, completed, volume_ratio, full_ratio):
    violation = torch.maximum((.5-volume_ratio).clamp_min(0.), (.5-full_ratio).clamp_min(0.))
    return done*(100.*violation.square()+20.*(~completed).float())
