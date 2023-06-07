import numpy as np
from d3rlpy.dataset import MDPDataset

from recsys_mdp.mdp.base import USER_ID_COL, OBSERVATION_COL
from recsys_mdp.mdp.episode_splitting import to_episode_ranges


class MDPFormer:
    def __init__(self, reward_function, action_function, episode_splitter):
        self.reward_function = reward_function
        self.action_function = action_function
        self.split_condition = episode_splitter

    def make_mdp(self, log_df, discrete_action: bool):
        observations, rewards, actions, terminates = [], [], [], []

        # as GroupBy preserves the order within each group, we can safely omit sorting
        for user_id, user_log_df in log_df.groupby(USER_ID_COL):
            episode_split_indices = self.split_condition(user_log_df)
            for ep_start_ind, ep_end_ind in to_episode_ranges(user_log_df, episode_split_indices):
                episode = log_df.iloc[ep_start_ind:ep_end_ind]

                # list of (episode_len, obs_size)
                observations.append(np.stack(episode[OBSERVATION_COL].values, axis=0))
                # list of (episode_len,)
                actions.append(self.get_episode_action(episode))
                rewards.append(self.get_episode_reward(episode))
                terminates.append(self.get_episode_terminates(episode))

        # shape: (n_interactions, obs_size)
        observations = np.concatenate(observations, axis=0)
        # shape: (n_interactions, 1)
        actions = np.expand_dims(np.concatenate(actions, axis=0), axis=1)
        rewards = np.expand_dims(np.concatenate(rewards, axis=0), axis=1)
        terminates = np.expand_dims(np.concatenate(terminates, axis=0), axis=1)

        dataset = MDPDataset(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminals=terminates,
            discrete_action=discrete_action
        )
        return dataset

    def get_episode_action(self, df):
        return self.action_function(df)

    def get_episode_reward(self, df):
        return self.reward_function(df)

    # noinspection PyMethodMayBeStatic
    def get_episode_terminates(self, df):
        terminates = np.zeros(df.shape[0], dtype=int)
        terminates[-1] = 1
        return terminates
