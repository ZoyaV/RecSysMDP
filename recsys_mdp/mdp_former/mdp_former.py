import pickle

import numpy as np
import pandas as pd

from recsys_mdp.mdp_former.acting import continuous_relevance_action
from recsys_mdp.mdp_former.base import USER_ID_COL, TIMESTAMP_COL, ITEM_ID_COL, RATING_COL
from recsys_mdp.mdp_former.episode_splitting import split_by_time, to_episode_ranges
from recsys_mdp.mdp_former.rewarding import monotony_reward
from recsys_mdp.mdp_former.utils import isnone

DEFAULT_HISTORY_KEYS = ['framestack', 'user_id']


class MDPFormer:
    def __init__(
            self, load_from_file: bool = False, path: str = None,
            dataframe: pd.DataFrame = None,
            framestack: int = 5,
            history_keys=None,
            reward_function=monotony_reward,
            action_function=continuous_relevance_action,
            episode_splitter=None,
    ):
        self.framestack = framestack
        self.history_keys = isnone(history_keys, DEFAULT_HISTORY_KEYS)
        self.reward_function = reward_function
        self.action_function = action_function
        self.split_condition = isnone(episode_splitter, split_by_time)

        self.states = None
        self.rewards = None
        self.actions = None
        self.terminations = None
        self.dataframe = dataframe

        if load_from_file:
            self.__load(path)

    def get_episode_action(self, df):
        return self.action_function(df)

    def get_episode_reward(self, df):
        return self.reward_function(df)

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

        episode_split_indices = self.split_condition(user_log)
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

    def make_interaction(self, rating, user, item, ts, obs_prev, relevance2reward=False):
        history = []
        history_size = (len(obs_prev) - 1) // 2 # len(items stack) + len(scorers stack) + 1(item)
        history += obs_prev[:history_size].copy()  # items history
        history += obs_prev[history_size:history_size*2].copy()  # scorers history
        history += [obs_prev[-1]]  # user id
        if relevance2reward:
            # FIXME: smells like kostyl'
            ratings_df = pd.DataFrame({RATING_COL: [rating]})
            reward = self.reward_function(ratings_df)[0]
            rating = reward

        interaction = {
            USER_ID_COL: user,
            RATING_COL: rating,
            ITEM_ID_COL: item,
            TIMESTAMP_COL: ts,
            'history': history,
        }

        obs_prev[:history_size - 1] = obs_prev[1:history_size]
        obs_prev[history_size - 1] = item
        #  print(obs[:framestack_size])
        obs_prev[history_size:history_size * 2 - 1] = obs_prev[history_size + 1:history_size * 2]
        obs_prev[history_size * 2 - 1] = rating

        return interaction, obs_prev

    def _interaction_history(self, user_df):
        """
        :param user_df:
        :return: return history with self.framestack size for each (user-item) interaction
        """
        interactions = []
        framestack_queue = []
        ratings_queue = []

        # Fill first framestack_size items to history
        for index, row in user_df.iterrows():
            framestack_queue.append(row[ITEM_ID_COL])
            ratings_queue.append(row[RATING_COL])
            if len(framestack_queue) >= self.framestack:
                break
        # Make interaction history for each user-item interaction
        t = 0

        obs_prev = []

        obs_prev += framestack_queue.copy() #items history
        obs_prev += ratings_queue.copy()
        obs_prev += user_df.iloc[:1][USER_ID_COL].values.tolist()  # user id
        for index, row in user_df.iterrows():
            t += 1
            if t < self.framestack:
                continue

            interaction, obs_prev = self.make_interaction(
                row[RATING_COL], row[USER_ID_COL],
                row[ITEM_ID_COL], row[TIMESTAMP_COL], obs_prev
            )
            interactions.append(interaction)

        df = pd.DataFrame(interactions)
        return df

    def create_mdp(self):
        """
        convert dataset to MDP
        :return:
        """
        unique_users = self.dataframe[USER_ID_COL].unique()
        full_states, full_rewards, full_actions, full_terminates = [], [], [], []
        state_tale = []
        for user_id in unique_users:
            user_df = (
                self
                .dataframe[self.dataframe[USER_ID_COL] == user_id]
                .sort_values(TIMESTAMP_COL)
            )

            if user_df.shape[0] < self.framestack:
                # lack of interactions
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