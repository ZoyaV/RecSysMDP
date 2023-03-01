import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from abc import ABC, abstractmethod

from recsys_mdp.utils import to_d3rlpy_form_ND
from recsys_mdp.reward_functions import monotony_reward, relevance_based_reward, condition_reward
from recsys_mdp.action_function import discrete_relevance_action, continuous_relevance_action, next_item_action


class RecSysMDP(ABC):
    def __init__(self, load_from_file=False, path=None,
                 dataframe=None, data_mapping=None,
                 framestack=5, reward_function=monotony_reward,
                 action_function=continuous_relevance_action,
                 history_keys=['framestack', 'user_id']):
        super().__init__()

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
            #  print(framestask_queue)
            framestask_queue.append(row[self.item_col_name])
            framestask_queue.pop(0)
            intarections_list.append(interaction)
        return pd.DataFrame(intarections_list)

    def create_mdp(self):
        """
        convert dataset to MDP
        :return:
        """
        users = list(set(self.dataframe[self.user_col_name]))
        full_states, full_rewards, full_actions, full_terminates = [], [], [], []
        for user in users:
            user_df = self.dataframe[self.dataframe[self.user_col_name] == user].sort_values(self.timestamp_col_name)
            if user_df.shape[0] < self.framestack: continue
            interaction_history = self.__intaraction_history(user_df)
            states, rewards, actions, terminates = self._mdp4user(interaction_history)
            full_states += states
            full_rewards += rewards
            full_actions += actions
            full_terminates += terminates
        self.states = full_states
        self.rewards = full_rewards
        self.actions = full_actions
        self.termations = full_terminates
        return (full_states, full_rewards, full_actions, full_terminates)

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
        no_op_framestack = np.ones((self.framestack, self.emb_size))
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
        self.condition = condition  # finction that return terminates
        super(ConditionBasedRecSysMDP, self).__init__(**args)

    def __detect_failuer(self, ts, condition):
        result = (ts[1:].values - ts[:-1].values).astype(int)
        indx = np.where(condition(result))
        # print(indx)
        return indx[0]

    def _mdp4user(self, user_df):
        """
        calc mdp trajectories by user
        :return:
        """
        states = []
        rewars = []
        actions = []
        termations = []
        # condition_music = lambda A: A > 0  # в эпизоды попадает только неразрывный по дню сеанс
        indx_to_episode_split = self.condition(user_df)
        for i, idx in enumerate(indx_to_episode_split):
            start = 0 if i == 0 else indx_to_episode_split[i - 1]
            end = idx
            one_episode = user_df.loc[start:end]
            if len(one_episode) < self.framestack: continue
            rewars.append(self.get_episode_reward(one_episode))
            states.append(one_episode['history'].values.tolist())
            actions.append(self.get_episode_action(one_episode))
            termations.append(self.get_episode_terminates(one_episode))
        return states, rewars, actions, termations


class FullUserHistoryBasedRecSysMDP(RecSysMDP):
    def __init__(self, **args):
        super(FullUserHistoryBasedRecSysMDP, self).__init__(**args)

    def _mdp4user(self, user_df):
        """
        calc mdp trajectories by user
        :return:
        """
        states = []
        rewars = []
        actions = []
        termations = []

        one_episode = user_df
        rewars.append(self.get_episode_reward(one_episode))
        states.append(one_episode['history'].values.tolist())
        actions.append(self.get_episode_action(one_episode))
        termations.append(self.get_episode_terminates(one_episode))
        return states, rewars, actions, termations


if __name__ == "__main__":
    from embedddings import random_embeddings
    import pandas as pd

    col_mapping = {'user_col_name': 'user_idx',
                   'item_col_name': 'item_id',
                   'reward_col_name': 'rating',
                   'timestamp_col_name': 'ts'}
    emb_size = 8
    framestask = 5
    data = pd.read_csv("clickstream_1ku.csv")
    data['ts'] = data['timestamp'].apply(pd.to_datetime)
    data = data.sort_values(['ts'])
    best_users_idx = data['user_id'].value_counts().index
    data['user_idx'] = data['user_id']
    filtered_raitings = data.set_index('user_id')
    filtered_raitings = filtered_raitings.loc[best_users_idx]
    filtered_raitings = filtered_raitings.reset_index(drop=False)

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
    print("Action example: ", actions[0][:5])
    print("Reward example: ", rewards[0][:5])
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
    # states, rewards, actions, termations = [np.asarray(component) for component in mdp]

    print("ConditionBasedRecSysMDP, monotony_reward, continuous_relevance_action")
    print("States representation: ", states[0][:3])
    #  print("States shape: ", states.shape)
    # print("Termations count: ", np.sum(termations.sum()))
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
