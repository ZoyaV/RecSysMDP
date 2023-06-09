from __future__ import annotations

import numpy as np
import torch.cuda
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


def get_cuda_device(cuda_device: int | None) -> int | None:
    if cuda_device is not None:
        import torch.cuda
        cuda_available = torch.cuda.is_available()
        print(f'CUDA available: {cuda_available}; device: {cuda_device}')
        if not cuda_available:
            cuda_device = None

    if isinstance(cuda_device, bool):
        cuda_device = 0
    return cuda_device


def load_checkpoint(model, model_name, step=-1):
    if step == -1:
        path = f'checkpoints/{model_name}/{model_name}.pt'
    else:
        path = f'checkpoints/{model_name}/{model_name}_{step}.pt'
    model.load_model(path)
