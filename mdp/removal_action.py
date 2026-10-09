"""Forward feed authority: ten digital 8 ms updates per policy decision."""
import math


def advance_feed(env, action):
    if action.shape != (env.count,):
        raise ValueError('Expected one feed action per environment')
    action = action.clamp(-.75, 1.)
    previous = env.cursor.clone()
    # Delay is measured in control ticks, so history updates below.
    for _ in range(10):
        env.history[:, 1:] = env.history[:, :-1].clone()
        env.history[:, 0] = action
        delayed = env.history.gather(1, env.delay[:, None]).squeeze(1)
        env.filtered += ((1.-math.exp(-.1))*(delayed-env.filtered)).clamp(-.024, .024)
        request = (6.+6.*env.filtered).clamp(0., 12.)
        env.slew += (request-env.slew).clamp(-.128, .128)
        env.speed += .04*(env.slew-env.speed)
        env.cursor = (env.cursor+env.speed*.008).clamp_max(env.length)
    return action, previous
