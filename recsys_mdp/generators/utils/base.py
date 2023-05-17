import numpy as np
from numpy.random import Generator


def sample_int(rng: Generator):
    return rng.integers(100_000_000)


def sample_rng(rng: Generator):
    return np.random.default_rng(sample_int(rng))


def exp_sum(ema, decay, val):
    """Returns new exponential moving average (EMA) adding next value."""
    return ema * decay + val


def lin_sum(x, lr, y):
    """Returns linear sum."""
    return x + lr * (y - x)


def update_exp_trace(traces, tr, decay, val=1., with_reset=False):
    """Updates exponential trace."""
    traces *= decay
    if with_reset:
        traces[tr] = val
    else:
        traces[tr] += val
