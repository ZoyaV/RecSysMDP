import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from abc import ABC, abstractmethod
import sys
sys.path.append("./recsys_mdp")

from utils import to_d3rlpy_form_ND
from reward_functions import monotony_reward, relevance_based_reward, condition_reward
from action_function import discrete_relevance_action, continuous_relevance_action, next_item_action


class RecSysMDP(ABC):
    def __init__(self, load_from_file=False, path=None,
                 dataframe=None, data_mapping=None,
                 framestack=5, reward_function=monotony_reward,
                 action_function=continuous_relevance_action,
                 history_keys=None):
        super().__init__()

        if history_keys is None:
            history_keys = ['framestack', 'user_id']
        self.framestack = framestack
        self.reward_function = reward_function
        self.action_function = action_function
        self.history_keys = history_keys
        self.state_size = None
        self.rewards_size = None
        self.action_size = None
        self.term_size = None

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
            self.state_size = np.asarray(self.states).shape
            self.rewards_size = np.asarray(self.rewards).shape
            self.action_size = np.asarray(self.actions).shape
            self.term_size = np.asarray(self.termations).shape

    def get_episode_action(self, df):
        return self.action_function(df, self.data_mapping)

    def get_episode_reward(self, df):
        return self.reward_function(df, self.data_mapping)

    def get_episode_terminates(self, df):
        terminates = np.zeros(df.shape[0])
        terminates[-1] = 1
        return terminates

    @abstractmethod
    def _mdp4user(self, user_df):
        """
        calc mdp trajectories by user
        :return:
        """
        pass

    def __intaraction_history(self, user_df):
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
            interaction_history = self.__intaraction_history(user_df)
            states, rewards, actions, terminates = self._mdp4user(interaction_history)
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

class WindowBasedRecSysMDP(RecSysMDP):
    """
    Use for each episode interctions relevant to window_size
    """

    def __init__(self, window_size, step_size, **args):
        self.window_size = window_size
        self.step_size = step_size
        super(WindowBasedRecSysMDP, self).__init__(**args)

    def _mdp4user(self, user_df):
        """
        calc mdp trajectories by user
        :return:
        """
        states = []
        rewars = []
        actions = []
        termations = []

        for i in range(0, len(user_df) - self.window_size, self.step_size):
            one_episode = user_df.loc[i:i + self.window_size]
            rewars.append(self.get_episode_reward(one_episode))
            states.append(one_episode['history'].values.tolist())
            actions.append(self.get_episode_action(one_episode))
            termations.append(self.get_episode_terminates(one_episode))
        return states, rewars, actions, termations


class ConditionBasedRecSysMDP(RecSysMDP):
    def __init__(self, condition, **args):
        self.condition = condition  # function that return terminates
        super(ConditionBasedRecSysMDP, self).__init__(**args)

    def __detect_failuer(self, ts, condition):
        result = (ts[1:].values - ts[:-1].values).astype(int)
        indx = np.where(condition(result))
        return indx[0]

    def _mdp4user(self, user_df):
        """
        calc mdp trajectories by user
        :return:
        """
        states, rewards, actions, terminations = [], [], [], []
        indx_to_episode_split = self.condition(user_df)
        for i, idx in enumerate(indx_to_episode_split):
            start = 0 if i == 0 else indx_to_episode_split[i - 1]
            end = idx
            one_episode = user_df.loc[start:end]
            if len(one_episode) < self.framestack: continue
            rewards.append(self.get_episode_reward(one_episode))
            states.append(one_episode['history'].values.tolist())
            actions.append(self.get_episode_action(one_episode))
            terminations.append(self.get_episode_terminates(one_episode))
        return states, rewards, actions, terminations


class FullUserHistoryBasedRecSysMDP(RecSysMDP):
    def __init__(self, **args):
        super(FullUserHistoryBasedRecSysMDP, self).__init__(**args)

    def _mdp4user(self, user_df):
        """
        calc mdp trajectories by user
        :return:
        """
        states, rewards, actions, terminations = [], [], [], []
        one_episode = user_df
        rewards.append(self.get_episode_reward(one_episode))
        states.append(one_episode['history'].values.tolist())
        actions.append(self.get_episode_action(one_episode))
        terminations.append(self.get_episode_terminates(one_episode))
        return states, rewards, actions, terminations

# Test classes
if __name__ == "__main__":
    import pandas as pd

    col_mapping = {'user_col_name': 'user_idx',
                   'item_col_name': 'item_idx',
                   'reward_col_name': 'rating',
                   'timestamp_col_name': 'ts'}
    emb_size = 8
    framestask = 5
    data = pd.read_csv("./row_data/ratings_rev_top10000_users.csv")[:10000]
    data['rating'] = data['rating'].astype(float)
    data[data['rating'] < 3]['rating'] = -data[data['rating'] < 3]['rating']
    data = data.sort_values(['event_dt'])
    best_users_idx = data['user_id'].value_counts()[:800].index

    data['user_idx'] = data['user_id']
    data['old_idx'] = data['user_id']
    filtered_raitings = data.set_index('user_id')
    filtered_raitings = filtered_raitings.loc[best_users_idx]
    filtered_raitings = filtered_raitings.reset_index(drop=False)

    keys = list(set(filtered_raitings['track_id']))
    item_mapping = dict(zip(keys, list(range(1, len(keys) + 1))))
    filtered_raitings['item_idx'] = filtered_raitings['track_id'].apply(lambda x: item_mapping[x])

    keys = list(set(filtered_raitings['user_idx']))
    user_mapping = dict(zip(keys, list(range(1, len(keys) + 1))))
    filtered_raitings['user_idx'] = filtered_raitings['user_idx'].apply(lambda x: user_mapping[x])

    mdp_train = FullUserHistoryBasedRecSysMDP(load_from_file=False, dataframe=filtered_raitings,
                                              data_mapping=col_mapping, framestack=framestask,
                                              reward_function=monotony_reward,
                                              action_function=continuous_relevance_action)
    states, rewards, actions, termations = mdp_train.create_mdp()
    # states, rewards, actions, termations = [np.asarray(component) for component in mdp]

    print("FullUserHistoryBasedRecSysMDP, monotony_reward, continuous_relevance_action")
    print("States representation: ", states[0][:3])
    # print("States shape: ", states.shape)
    # print("Termations count: ", np.sum(termations.sum()))
    print("Action example: ", actions[0][:5])
    print("Reward example: ", rewards[0][:5])
    print()
    # try:
    to_d3rlpy_form_ND(states, rewards, actions, termations)
    print("Success transform to d3rlpy")
    #  except Exception as e:
    #  print(e)

    print("------------------------------------------------------------------")

    mdp_train = WindowBasedRecSysMDP(load_from_file=False, dataframe=filtered_raitings,
                                     data_mapping=col_mapping, framestack=framestask,
                                     reward_function=monotony_reward,
                                     action_function=continuous_relevance_action,
                                     window_size=10, step_size=1)
    states, rewards, actions, termations = mdp_train.create_mdp()
    #  states, rewards, actions, termations = [np.asarray(component) for component in mdp]

    print("WindowBasedRecSysMDP, monotony_reward, continuous_relevance_action")
    print("States representation: ", states[0][:3])
    # print("Termations count: ", np. sum(termations.sum()))
    print("Action example: ", actions[0][:3])
    print("Reward example: ", rewards[0][:3])
    print()
    try:
        to_d3rlpy_form_ND(states, rewards, actions, termations)
        print("Success transform to d3rlpy")
    except Exception as e:
        print(e)
    print("------------------------------------------------------------------")


    def split_by_time(df):
        ts = df['ts']
        condition_music = lambda A: A > 0
        result = (ts[1:].values - ts[:-1].values).astype(int)
        indx = np.where(condition_music(result))
        return indx[0]


    mdp_train = ConditionBasedRecSysMDP(load_from_file=False, dataframe=filtered_raitings,
                                        data_mapping=col_mapping, framestack=framestask,
                                        reward_function=monotony_reward,
                                        action_function=continuous_relevance_action, condition=split_by_time)
    states, rewards, actions, termations = mdp_train.create_mdp()

    print("ConditionBasedRecSysMDP, monotony_reward, continuous_relevance_action")
    print("States representation: ", states[0][:5])
    print("Action example: ", actions[0][:5])
    print("Reward example: ", rewards[0][:5])

    try:
        to_d3rlpy_form_ND(states, rewards, actions, termations)
        print("Success transform to d3rlpy")
    except Exception as e:
        print(e)

    # from utils import to_d3rlpy_form_ND
    #
    # dataset_train = to_d3rlpy_form_ND(mdp_train.states, mdp_train.rewards,
    #                                mdp_train.actions, mdp_train.termations, 4)
    # print(mdp_train.states.shape)
