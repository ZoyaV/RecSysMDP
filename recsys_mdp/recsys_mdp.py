
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from abc import ABC, abstractmethod
class RecSysMDP(ABC):
    def __init__(self, load_from_file = False, path = None, dataframe = None, data_mapping = None,
                 framestack = 5, emb_size = 64,
                 user_mapping = None, item_mapping = None,
                 use_user_embedding = False, inv_user_mapping = None ):
        super().__init__()
        self.state_size = None
        self.rewards_size = None
        self.action_size = None
        self.term_size = None

        self.states = None
        self.rewards = None
        self.actions = None
        self.termations = None

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


    @abstractmethod
    def _sates4episode(self, episode_logs):
        pass

    @abstractmethod
    def _mdp4user(self, user_df):
        """
        calc mdp trajectories by user
        :return:
        """
        pass

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




class SplitNRecSysMDP(RecSysMDP):
    def __init__(self,rolling_size, **args):
        self.rolling_size = rolling_size
        super.__init__(**args)

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
        chooses = np.asarray([self.item_mapping[item_idx] for item_idx in episode_logs[self.item_col_name]])
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
        no_op_framestack = np.ones((self.framestack, self.emb_size))
        states = []
        rewars = []
        actions = []
        termations = []

        mult = 1 / (self.rolling_size - self.framestack)
        M = [mult * i for i in range(self.rolling_size - self.framestack)][::-1]
        M = np.asarray(M)
        # print(M)
        for i in range(0, len(user_df) - self.rolling_size, 1):
            logs = user_df[i:i + self.rolling_size]
            states_one_episode = self._sates4episode(logs)
            rewards_one_episode = logs[self.reward_col_name].copy()

            actions_one_episode = logs[self.item_col_name]
            states_one_episode = np.asarray(states_one_episode[self.framestack:])

            new_rewards_one_episode = rewards_one_episode[self.framestack:]
            new_rewards_one_episode[new_rewards_one_episode <= 3] = -(
            new_rewards_one_episode[new_rewards_one_episode <= 3])
            new_rewards_one_episode *= M
            actions_one_episode = actions_one_episode[self.framestack:]
            ###
            termations_one_episode = np.zeros_like(actions_one_episode)
            termations_one_episode[-1] = 1

            rewars.append(new_rewards_one_episode)
            states.append(states_one_episode)
            actions.append(actions_one_episode)
            termations.append(termations_one_episode)
        return np.asarray(states), np.asarray(rewars), np.asarray(actions), np.asarray(termations)


class SplitByFailureRecSysMDP(RecSysMDP):
    def __init__(self, **args):
     #   self.rolling_size = rolling_size
        super().__init__(**args)
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
        no_op_framestack = np.ones((self.framestack, self.emb_size))
        chooses = np.asarray([self.item_mapping[item_idx] for item_idx in episode_logs[self.item_col_name]])
        good_examples = np.where(episode_logs[self.reward_col_name] > 3)
       # print(np.where(good_examples))
      #  print(good_examples)
        bad_examples = np.where(episode_logs[self.reward_col_name] < 3)
      #  print(bad_examples)
     #   exit()
        chooses = np.append(no_op_framestack, chooses, axis=0)
        states = []
        actions = []
        rewards = []
        termations = []
        for i in range( self.framestack, len(episode_logs)):
            state = chooses[i: i + self.framestack].reshape(1, self.framestack, self.emb_size)

            filter_good = list(good_examples[self.framestack:i+self.framestack])
            filter_bad = list(bad_examples[self.framestack:i + self.framestack])
            good_hist = chooses[filter_good][-self.framestack:].reshape(1, -1)
            bad_hist = chooses[filter_bad][-self.framestack:].reshape(1, -1)
           # print(good_examples[self.framestack:i+self.framestack])
           # print(chooses[[]])
           # print(chooses[good_examples[self.framestack:i+self.framestack]])
         #   print(good_hist)
        #    print(bad_hist.shape)
          #  exit()
           # print(state.shape)
            try:
               state = np.append(state, good_hist.reshape(1, self.framestack, self.emb_size), axis=0)
               print("Good success!")
            except:
               good_hist = np.ones_like(state).reshape(1, self.framestack, self.emb_size)
               state = np.append(state, good_hist, axis=0)

          #  print(state.shape)
          #   try:
          #       state = np.append(state, bad_hist.reshape(1, self.framestack, self.emb_size), axis=0)
          #       print("Bad success!")
          #   except:
          #      bad_hist = np.ones_like(state[:1]).reshape(1, self.framestack, self.emb_size)
          #      state = np.append(state, bad_hist, axis=0)

            if self.use_user_embedding:
                user_ebd = np.ones((1,self.framestack, self.emb_size))
                user_map =  self.user_mapping[episode_logs[self.user_col_name].values[0]].reshape(1, -1)
                # print(user_map[0].shape)
                # print(user_ebd.shape)
                # print( user_ebd[0,:,0].shape)
                user_ebd[0,0,:] = user_map[0]
                # print(user_ebd.shape)
                # print(state.shape)
                state = np.append(state, user_ebd, axis=0)

           # print(state.shape)
        #    exit()
          #  print(episode_logs[self.item_col_name])
            actions.append(episode_logs[self.item_col_name].values[i])
            rewards.append(0.1)#episode_logs[self.reward_col_name].values[i])
            termations.append(0)
          #  print(termations)

           # print(termations)
            states.append(state)
        termations[-1] = 1
        rewards[-1] = -1
        return states, actions, rewards, termations

    def __detect_failuer(self, ts, condition):
        result = (ts[1:].values-ts[:-1].values).astype(int)
        indx = np.where(condition(result))
        #print(indx)
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
        condition_music = lambda A: A > 0  # ?? ?????????????? ???????????????? ???????????? ?????????????????????? ???? ?????? ??????????
        indx_to_episode_split = self.__detect_failuer(user_df[self.timestamp_col_name],condition_music)

        for i,idx in enumerate(indx_to_episode_split):
            start = 0 if i==0 else indx_to_episode_split[i-1]
            end = idx
            logs = user_df[start:end]
           # print(len(logs))
            if len(logs)<=self.framestack: break
            states_one_episode, actions_one_episode,rewards_one_episode, termations_one_episode = self._sates4episode(logs)

            rewars.append(rewards_one_episode)
            states.append(states_one_episode)
            actions.append(actions_one_episode)
            termations.append(termations_one_episode)
        return np.asarray(states), np.asarray(rewars), np.asarray(actions), np.asarray(termations)

if __name__ == "__main__":
    from embedddings import random_embeddings
    import pandas as pd

    col_mapping = {'user_col_name': 'user_idx',
                   'item_col_name': 'item_idx',
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

    keys = list(set(filtered_raitings['item_id']))
    item_mapping = dict(zip(keys, list(range(1, len(keys) + 1))))
    filtered_raitings['item_idx'] = filtered_raitings['item_id'].apply(lambda x: item_mapping[x])

    keys = list(set(filtered_raitings['user_idx']))
    user_mapping = dict(zip(keys, list(range(1, len(keys) + 1))))
    filtered_raitings['user_idx'] = filtered_raitings['user_idx'].apply(lambda x: user_mapping[x])

   # print(filtered_raitings[:5])
    user_mapping, inv_user_mapping = random_embeddings(filtered_raitings[col_mapping['user_col_name']], emb_size=emb_size)
    items_mapping, _ = random_embeddings(filtered_raitings[col_mapping['item_col_name']], emb_size=emb_size)

    mdp_train = SplitByFailureRecSysMDP(load_from_file=False, dataframe=filtered_raitings, data_mapping=col_mapping,
                          framestack=framestask, emb_size=emb_size, user_mapping=user_mapping,
                          item_mapping=items_mapping, use_user_embedding=True, inv_user_mapping=inv_user_mapping)
    mdp_train.create_mdp()
    from utils import to_d3rlpy_form_ND

    dataset_train = to_d3rlpy_form_ND(mdp_train.states, mdp_train.rewards,
                                   mdp_train.actions, mdp_train.termations, 4)
    #print(mdp_train.states.shape)