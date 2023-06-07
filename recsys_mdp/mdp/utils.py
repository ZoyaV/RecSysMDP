import numpy as np
from d3rlpy.dataset import MDPDataset


def to_d3rlpy_mdp_dataset(
        states, rewards, actions, terminates, discrete_action: bool
):
    print(f'{len(states)=} | {states[0].shape=}')
    states = np.concatenate(states, axis=0)
    # full_states = []
    # full_rewards = []
    # full_actions = []
    # full_termates = []
    # for i, episode in enumerate(original_states):
    #     full_states += episode
    #     full_rewards += original_rewards[i].tolist()
    #     full_actions += original_actions[i].tolist()
    #     full_termates += original_termates[i].tolist()

    print(f'{states.shape=}')

    rewards = np.asarray(full_rewards).reshape(-1, 1)
    actions = np.asarray(full_actions).reshape(-1, 1)
    termates = np.asarray(full_termates).reshape(-1, 1)

    dataset = MDPDataset(
        observations=states,
        actions=actions,
        rewards=rewards,
        terminals=termates,
        discrete_action = discrete_action
    )
    return dataset


def isnone(x, default):
    """Return x if it's not None, or default value instead."""
    return x if x is not None else default
