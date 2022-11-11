import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

class RecSysMDP():
    def __init__(self, load_from_file = False, path = None, dataframe = None, data_mapping = None,
                 rolling_size = 20, framestack = 5, emb_size = 64,
                 user_mapping = None, item_mapping = None,
                 use_user_embedding = False, inv_user_mapping = None):
        self.state_size = None
        self.rewards_size = None
        self.action_size = None
        self.term_size = None

        self.states = None
        self.rewards = None
        self.actions = None
        self.termations = None

        self.rolling_size = rolling_size
        self.framestack = framestack
        self.user_mapping = user_mapping
        self.item_mapping = item_mapping
        self.use_user_embedding = use_user_embedding
        self.inv_user_mapping = inv_user_mapping

        if not load_from_file:
            self.dataframe = dataframe #Pandas-DataFram object
            self.data_mapping = data_mapping #Dict of keys for mdp {'reward':'score', 'user_col_name':...}
            self.emb_size = emb_size
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


    def _sates4episode(self, episode_logs):
        """
        inp:
         - logs for one episode
        :return:
         - stacked states for one episode with size
         (N - count of states,
         framestack + 1(if use_user_embedding) - size of framestack,
         EMB_S - size of embedding

        """
        no_op_framestack = np.zeros((self.framestack, self.emb_size))
        chooses = np.asarray([self.item_mapping[item_idx] for item_idx in episode_logs[self.user_col_name]])
        chooses = np.append(no_op_framestack, chooses, axis=0)
        states = []
        for i in range(len(episode_logs)):
            state = chooses[i: i + self.framestack]
            if self.use_user_embedding:
                user_ebd = self.user_mapping[episode_logs[self.user_col_name].values[0]].reshape(1, -1)
                state = np.append(state, user_ebd, axis=0)
            states.append(state)
        return states

    def _mdp4user(self, user_df):
        """
        calc mdp trajectories by user
        :return:
        """
        no_op_framestack = np.zeros((self.framestack, self.emb_size))
        states = []
        rewars = []
        actions = []
        termations = []
        for i in range(len(user_df) - self.rolling_size):
            logs = user_df[i:i + self.rolling_size]
            states_one_episode = self._sates4episode(logs)
            rewards_one_episode = logs[self.reward_col_name]
            actions_one_episode = logs[self.item_col_name]
            states_one_episode = np.asarray(states_one_episode[self.framestack:])
            rewards_one_episode = rewards_one_episode[self.framestack:]
            actions_one_episode = actions_one_episode[self.framestack:]
            ###
            termations_one_episode = np.zeros_like(actions_one_episode)
            termations_one_episode[-1] = 1

            rewars.append(rewards_one_episode)
            states.append(states_one_episode)
            actions.append(actions_one_episode)
            termations.append(termations_one_episode)
        return np.asarray(states), np.asarray(rewars), np.asarray(actions), np.asarray(termations)

    def save(self, path):
        data =  (self.states, self.rewards,  self.actions, self.termations)
        random_part = np.random.randint(0,100)
        with open(path + "_%d.pkl"%random_part, 'wb') as f:
            pickle.dump(data, f)
            print("Saved to %s"%path)
        with open(path + "_%d_user_embeddings.pkl"%random_part, 'wb') as f:
            pickle.dump(self.user_mapping, f)
        with open(path + "_%d_item_embeddings.pkl"%random_part, 'wb') as f:
            pickle.dump(self.item_mapping, f)
        with open(path + "_%d_df.pkl"%random_part, 'wb') as f:
            pickle.dump(self.dataframe, f)
        with open(path + "_%d_inv_user_mapping.pkl"%random_part, 'wb') as f:
            pickle.dump(self.inv_user_mapping, f)

        print("Saved at %s"%(path + "_%d.pkl"%random_part))

    def __load(self, path):
        with open(path + ".pkl", 'rb') as f:
            self.states, self.rewards,  self.actions, self.termations = pickle.load(f)
        with open(path + "_user_embeddings.pkl", 'rb') as f:
            self.user_mapping = pickle.load(f)
        with open(path + "_item_embeddings.pkl", 'rb') as f:
            self.item_mapping = pickle.load(f)
        with open(path + "_df.pkl", 'rb') as f:
            self.dataframe = pickle.load(f)
        with open(path + "_inv_user_mapping.pkl", 'rb') as f:
            self.inv_user_mapping = pickle.load(f)

        print("Data loaded!")

    def create_mdp(self):
        """
        convert dataset to MDP
        :return:
        """
        users = list(set(self.dataframe[self.user_col_name]))
        full_states = []
        full_rewards = []
        full_actions = []
        full_termates = []
        for user in users:
            user_df = self.dataframe[self.dataframe[self.user_col_name] == user].sort_values(self.timestamp_col_name)
            states, rewards, actions, termates = self._mdp4user(user_df)

            full_states += states.tolist()
            full_rewards += rewards.tolist()
            full_actions += actions.tolist()
            full_termates += termates.tolist()

        self.state_size = np.asarray(full_states).shape
        self.rewards_size = np.asarray(full_rewards).shape
        self.action_size = np.asarray(full_actions).shape
        self.term_size = np.asarray(full_termates).size

        self.states = full_states
        self.rewards = full_rewards
        self.actions = full_actions
        self.termations = full_termates

        return (full_states, full_rewards, full_actions, full_termates)




