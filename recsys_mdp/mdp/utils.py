import numpy as np
from d3rlpy.dataset import MDPDataset


def to_d3rlpy_form_ND(
        original_states, original_rewards, original_actions,
        original_termates, discrete = False
):
    # TODO: rework via np.stack
    full_states = []
    full_rewards = []
    full_actions = []
    full_termates = []
    for i, episode in enumerate(original_states):
        if isinstance(episode, np.ndarray):
            episode = episode.tolist()
        full_states += episode
        full_rewards += original_rewards[i].tolist()
        full_actions += original_actions[i].tolist()
        full_termates += original_termates[i].tolist()

    states = np.asarray(full_states)
    rewards = np.asarray(full_rewards).reshape(-1, 1)
    actions = np.asarray(full_actions).reshape(-1, 1)
    termates = np.asarray(full_termates).reshape(-1, 1)

    dataset = MDPDataset(
        observations=states,
        actions=actions,
        rewards=rewards,
        terminals=termates,
        discrete_action = discrete
    )
    return dataset


def isnone(x, default):
    """Return x if it's not None, or default value instead."""
    return x if x is not None else default
