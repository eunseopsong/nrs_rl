"""16D measured/contact/geometry/residence features for the process environment."""
import torch

def removal_observation(env):
    index = env.index(env.cursor)
    ids, weights = env.indices[index], env.weights[index]
    norm = weights.sum(-1)
    tail = env.remaining[index[:, None], ids]
    local = lambda x: (x*weights).sum(-1)/norm
    deficit = 1.-local(env.depth[env.batch, ids]+tail)/env.target
    back = env.index((env.cursor-5.).clamp_min(0.))
    backids, bw = env.indices[back], env.weights[back]
    backpred = env.depth[env.batch, backids]+env.remaining[index[:, None], backids]
    backdeficit = 1.-(backpred*bw).sum(-1)/bw.sum(-1)/env.target
    obs = torch.zeros((env.count, 16), device=env.device)
    obs[:, 0] = env.cursor/env.length
    obs[:, 1] = env.force/20.
    obs[:, 2] = env.speed/6.
    obs[:, 3] = 1.-local(env.nominal[ids])/env.target-env.path_deficit
    obs[:, 4] = deficit-env.path_deficit
    obs[:, 5] = backdeficit-deficit
    obs[:, 6] = local(env.residence[env.batch, ids])/5.
    obs[:, 8] = env.elapsed/(env.length/6.)
    obs[:, 10] = env.force >= 1.5
    obs[:, 12] = 1.
    obs[:, 13] = env.force*env.speed/120.-1.
    obs[:, 15] = local(tail)/env.target
    return obs.clamp(-5., 5.)

