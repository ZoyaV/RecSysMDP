import pickle
import sys

import numpy as np
import pandas as pd

sys.path.append("./recsys_mdp")

from reward_functions import monotony_reward
from action_function import continuous_relevance_action
from episode_split_fucntions import split_by_time
class RecSysMDP():
    def __init__(self, load_from_file=False, path=None,
                 dataframe=None, data_mapping=None,
                 framestack=5, reward_function=monotony_reward,
                 action_function=continuous_relevance_action,
                 history_keys=None, episode_splitter = None):
        super().__init__()

        if history_keys is None:
            history_keys = ['framestack', 'user_id']
        self.framestack = framestack
        self.reward_function = reward_function
        self.action_function = action_function
        self.history_keys = history_keys

        if episode_splitter is not None:
            self.condition = episode_splitter
        else:
            self.condition = split_by_time

        self.states = None
        self.rewards = None
        self.actions = None
        self.termations = None

        if not load_from_file:
            self.dataframe = dataframe  # Pandas-DataFram object
            self.data_mapping = data_mapping  # Dict of keys for mdp {'reward':'score', 'user_col_name':...}

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

    def get_episode_terminates(self, df):
        terminates = np.zeros(df.shape[0])
        terminates[-1] = 1
        return terminates

    def _mdp4user(self, user_df):
        """
        calc mdp trajectories by user
        :return:
        """
        states, rewards, actions, terminations = [], [], [], []
        indx_to_episode_split = self.condition(user_df, self.data_mapping)
        for i, idx in enumerate(indx_to_episode_split):
            start = 0 if i == 0 else indx_to_episode_split[i - 1]
            end = idx
            one_episode = user_df.loc[start:end]
            if len(one_episode) < self.framestack:
                continue
            rewards.append(self.get_episode_reward(one_episode))
            states.append(one_episode['history'].values.tolist())
            actions.append(self.get_episode_action(one_episode))
            terminations.append(self.get_episode_terminates(one_episode))
        return states, rewards, actions, terminations

    def _intaraction_history(self, user_df):
        """
        :param user_df:
        :return: return history with self.framestack size for each (user-item) interaction
        """
        intarections_list = []
        framestask_queue = []

        # Fill first framestack_size items to history
        for index, row in user_df.iterrows():
            framestask_queue.append(row[self.item_col_name])
            if len(framestask_queue) >= self.framestack:
                break
        # Make interaction history for each user-item interaction
        t = 0
        for index, row in user_df.iterrows():
            t += 1
            if t < self.framestack: continue
            history = []
            if 'framestack' in self.history_keys:
                history += framestask_queue.copy()
            if 'user_id' in self.history_keys:
                history += [row[self.user_col_name]]
            interaction = {self.user_col_name: row[self.user_col_name],
                           'history': history,
                           self.reward_col_name: row[self.reward_col_name],
                           self.item_col_name: row[self.item_col_name],
                           self.timestamp_col_name: row[self.timestamp_col_name]}
            framestask_queue.append(row[self.item_col_name])
            framestask_queue.pop(0)
            intarections_list.append(interaction)

        df = pd.DataFrame(intarections_list)
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
            user_df = self.dataframe[self.dataframe[self.user_col_name] == user].sort_values(self.timestamp_col_name)
            if user_df.shape[0] < self.framestack: continue
            interaction_history = self._intaraction_history(user_df)
            states, rewards, actions, terminates = self._mdp4user(interaction_history)
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
        self.termations = full_terminates
        return full_states, full_rewards, full_actions, full_terminates, state_tale

    def save(self, path):
        data = (self.states, self.rewards, self.actions, self.termations)
        random_part = np.random.randint(0, 100)
        with open(path + "_%d.pkl" % random_part, 'wb') as f:
            pickle.dump(data, f)
            print("Saved to %s" % path)
        with open(path + "_%d_df.pkl" % random_part, 'wb') as f:
            pickle.dump(self.dataframe, f)
        print("Saved at %s" % (path + "_%d.pkl" % random_part))

    def __load(self, path):
        with open(path + ".pkl", 'rb') as f:
            self.states, self.rewards, self.actions, self.termations = pickle.load(f)
        with open(path + "_df.pkl", 'rb') as f:
            self.dataframe = pickle.load(f)
        print("Data loaded!")

# Test classes
if __name__ == "__main__":
    pass

