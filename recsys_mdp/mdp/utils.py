import os
import numpy as np
from d3rlpy.dataset import MDPDataset


def to_d3rlpy_form_ND(original_states, original_rewards, original_actions,
                      original_termates, discrete = False):
    full_states = []
    full_rewards = []
    full_actions = []
    full_termates = []
    for i,episode in enumerate(original_states):
        if isinstance(episode, np.ndarray):
            episode = episode.tolist()
        full_states += episode
        full_rewards += original_rewards[i].tolist()
        full_actions += original_actions[i].tolist()
        full_termates += original_termates[i].tolist()

    states = np.asarray(full_states)
    states = states #(states).astype(int)
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

    np.random.shuffle(dataset.episodes)
    return dataset


def make_datasets(dataframe, col_mapping, test_part,
                  data_directory = "./data", data_name = "ml_100k_first100",
                  framestask = 5, emb_size = 8):

    user_mapping, inv_user_mapping = random_embeddings(dataframe[col_mapping['user_col_name']],emb_size = emb_size)
    items_mapping, _ = random_embeddings(dataframe[col_mapping['item_col_name']],emb_size = emb_size)

    ts = sorted(list(set(dataframe[col_mapping['timestamp_col_name']])))
    test_count = int(len(ts)*test_part)
    treshold =ts[-test_count]
    train_df = dataframe[dataframe[col_mapping['timestamp_col_name']]<treshold]
    test_df = dataframe[dataframe[col_mapping['timestamp_col_name']]>=treshold]

    print(treshold)
    print(len(train_df))
    print(len(test_df))

    os.makedirs(data_directory, exist_ok=True)
    mdp_train = RecSysMDP(load_from_file=False, dataframe=train_df, data_mapping=col_mapping,
                          framestack = framestask, emb_size = emb_size,  user_mapping = user_mapping,
                          item_mapping = items_mapping, use_user_embedding = True, inv_user_mapping = inv_user_mapping)
    mdp_train.create_mdp()
    dataset_name = data_name + "_train"
    path = os.path.join(data_directory, dataset_name)
    mdp_train.save(path)

    mdp_test = RecSysMDP(load_from_file=False, dataframe=test_df, data_mapping=col_mapping,
                         framestack = framestask, emb_size = emb_size,  user_mapping = user_mapping,
                         item_mapping = items_mapping, use_user_embedding = True, inv_user_mapping = inv_user_mapping)
    mdp_test.create_mdp()
    dataset_name = data_name + "_test"
    path = os.path.join(data_directory, dataset_name)
    mdp_test.save(path)

    return mdp_train, mdp_test


def isnone(x, default):
    """Return x if it's not None, or default value instead."""
    return x if x is not None else default
