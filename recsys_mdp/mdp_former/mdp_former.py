import pickle

import numpy as np
import pandas as pd

from recsys_mdp.mdp_former.acting import continuous_relevance_action
from recsys_mdp.mdp_former.episode_splitting import split_by_time, to_episode_ranges
from recsys_mdp.mdp_former.rewarding import monotony_reward
from recsys_mdp.mdp_former.utils import isnone

DEFAULT_HISTORY_KEYS = ['framestack', 'user_id']


class MDPFormer:
    def __init__(
            self, load_from_file: bool = False, path: str = None,
            dataframe=None, data_mapping=None,
            framestack=5, reward_function=monotony_reward,
            action_function=continuous_relevance_action,
            history_keys=None, episode_splitter=None
    ):
        self.framestack = framestack
        self.reward_function = reward_function
        self.action_function = action_function
        self.history_keys = isnone(history_keys, DEFAULT_HISTORY_KEYS)

        self.condition = isnone(episode_splitter, split_by_time)

        self.states = None
        self.rewards = None
        self.actions = None
        self.terminations = None

        if not load_from_file:
            # Pandas-DataFrame object
            self.dataframe = dataframe
            # Dict of keys for mdp {'reward':'score', 'user_col_name':...}
            self.data_mapping = data_mapping

            self.user_col_name = self.data_mapping['user_col_name']
            self.item_col_name = self.data_mapping['item_col_name']
            self.reward_col_name = self.data_mapping['reward_col_name']
            self.timestamp_col_name = self.data_mapping['timestamp_col_name']
        else:
            self.__load(path)

    def get_episode_action(self, df):
        return self.action_function(df, self.data_mapping)

    def get_episode_reward(self, df):
        return self.reward_function(df, self.data_mapping)

    # noinspection PyMethodMayBeStatic
    def get_episode_terminates(self, df):
        terminates = np.zeros(df.shape[0])
        terminates[-1] = 1
        return terminates

    def _get_user_episodes(self, user_log: pd.DataFrame):
        """
        Transforms the user's log to MDP trajectories aka episodes
        :return:
        """
        states, rewards, actions, terminations = [], [], [], []

        episode_split_indices = self.condition(user_log, self.data_mapping)
        for ep_start_ind, ep_end_ind in to_episode_ranges(user_log, episode_split_indices):
            episode = user_log.iloc[ep_start_ind:ep_end_ind]
            if len(episode) < self.framestack:
                # throw out too short trajectories
                continue
            rewards.append(self.get_episode_reward(episode))
            states.append(episode['history'].values.tolist())
            actions.append(self.get_episode_action(episode))
            terminations.append(self.get_episode_terminates(episode))

        return states, rewards, actions, terminations

    def make_interaction(self, relevance, user, item, ts, obs_prev, relevance2reward = False):
        """

        :param reward: reward for predictef item
        :param user: current user
        :param item: predicted item
        :param ts: timestamp
        :param obs_prev: last framestack
        :return:
        """

        history = []
        history_size = (len(obs_prev) - 1) //2 # len(items stack) + len(scorers stack) + 1(item)
        history += obs_prev[:history_size].copy()  # items history
        history += obs_prev[history_size:history_size*2].copy()  # scorers history
        history += [obs_prev[-1]]  # user id
        if relevance2reward:
            rewards_df = pd.DataFrame({self.data_mapping['reward_col_name']: [relevance]})
            relevance = self.reward_function(rewards_df, self.data_mapping)[0]
        interaction = {self.user_col_name: user,
                       'history': history,
                       self.reward_col_name: relevance,
                       self.item_col_name: item,
                       self.timestamp_col_name: ts}

        obs_prev[:history_size - 1] = obs_prev[1:history_size]
        obs_prev[history_size - 1] = item
        #  print(obs[:framestack_size])
        obs_prev[history_size:history_size * 2 - 1] = obs_prev[history_size + 1:history_size * 2]
        obs_prev[history_size * 2 - 1] = relevance

        return interaction, obs_prev
    def _interaction_history(self, user_df):
        """
        :param user_df:
        :return: return history with self.framestack size for each (user-item) interaction
        """
        interactions = []
        framestack_queue = []
        scorers_queue = []

        # Fill first framestack_size items to history
        for index, row in user_df.iterrows():
            framestack_queue.append(row[self.item_col_name])
            scorers_queue.append(row[self.reward_col_name])
            if len(framestack_queue) >= self.framestack:
                break
        # Make interaction history for each user-item interaction
        t = 0

        obs_prev = []

        obs_prev += framestack_queue.copy() #items history
        obs_prev += scorers_queue.copy() #scorers history
        obs_prev += user_df[:1][self.user_col_name].values.tolist()  # user id
        for index, row in user_df.iterrows():
            t += 1
            if t < self.framestack: continue

            user = row[self.user_col_name]
            relevance = row[self.reward_col_name]
            item = row[self.item_col_name]
            ts = row[self.timestamp_col_name]

            interaction, obs_prev = self.make_interaction(relevance, user,
                                                          item, ts, obs_prev)
            interactions.append(interaction)

        df = pd.DataFrame(interactions)
        return df

    def create_mdp(self):
        """
        convert dataset to MDP
        :return:
        """
        users = list(set(self.dataframe[self.user_col_name]))
        full_states, full_rewards, full_actions, full_terminates = [], [], [], []
        state_tale = []
        for user in users:
            user_df = self.dataframe[self.dataframe[self.user_col_name] == user].sort_values(
                self.timestamp_col_name
            )
            if user_df.shape[0] < self.framestack:
                continue
            interaction_history = self._interaction_history(user_df)
            states, rewards, actions, terminates = self._get_user_episodes(interaction_history)
            if len(states) < 1:
                continue
            state_tale.append(states[-1][-1])
            full_states += states
            full_rewards += rewards
            full_actions += actions
            full_terminates += terminates
        self.states = full_states
        self.rewards = full_rewards
        self.actions = full_actions
        self.terminations = full_terminates
        return full_states, full_rewards, full_actions, full_terminates, state_tale

    def save(self, path):
        data = (self.states, self.rewards, self.actions, self.terminations)
        random_part = np.random.randint(0, 100)
        with open(path + "_%d.pkl" % random_part, 'wb') as f:
            pickle.dump(data, f)
            print("Saved to %s" % path)
        with open(path + "_%d_df.pkl" % random_part, 'wb') as f:
            pickle.dump(self.dataframe, f)
        print("Saved at %s" % (path + "_%d.pkl" % random_part))

    def __load(self, path):
        with open(path + ".pkl", 'rb') as f:
            self.states, self.rewards, self.actions, self.terminations = pickle.load(f)
        with open(path + "_df.pkl", 'rb') as f:
            self.dataframe = pickle.load(f)
        print("Data loaded!")


# Test classes
if __name__ == "__main__":
    pass

