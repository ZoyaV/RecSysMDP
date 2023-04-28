from __future__ import annotations

from typing import TYPE_CHECKING

import gym
import numpy as np
import pandas as pd
import wandb
from gym.spaces import Box

from replay.models.rl.experiments.utils.metrics import ndcg, mean_average_precision

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run

def original_for_user(df, target, k = 10):
    mask = df['user_id'] == target
    user_relevance = df[mask]
    return user_relevance.sort_values(['rating'])[::-1][:k]


class FakeRecomenderEnv(gym.Env):
    logger: Run | None

    def __init__(self, logger, test_data, top_k):
        self.logger = logger

        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = Box(0,100000, (2,))
        self.log_data = test_data
        self.top_k = top_k
        self.steps = 0
        self.episode_num = 0
        self.total_ndsg = []
        self.total_mape = []
        self.episodes = list(set(self.log_data['user_id']))
        self.total_episodes = 0
        #mask = self.log_data['user_id'] == episodes[episode_num]
        self.current_episode = None
        self.run = 0

    def step(self, action): 
        #print(action)
        self.relevance_hist.append(action)
        done = False
        reward = 0
        ob = (self.current_episode['user_id'].values[self.steps], 
                self.current_episode['item_id'].values[self.steps])
        self.steps += 1
       
        if len(self.current_episode['user_id']) == self.steps:
            # done = True
            #  print(len(self.user_hist), len(self.item_hist), len(self.relevance_hist))
            pred_df = pd.DataFrame({'user_id': self.user_hist, 'item_hist': self.item_hist,
                                    'relevance': self.relevance_hist})
            pred_top_k = pred_df.sort_values(['relevance'])[::-1][:self.top_k]
            ndcg_ = ndcg(self.top_k, pred_top_k['item_hist'].values, self.original['item_id'].values)
            mape_ = mean_average_precision(self.top_k, pred_top_k['item_hist'].values, self.original['item_id'].values)
            if self.logger:
                self.logger.log({"episode": self.total_episodes, "NDCG": ndcg_, "MAP": mape_})
            self.total_ndsg.append(ndcg_)
            self.total_mape.append(mape_)
            ob = []            
            
            if self.episode_num >= len(self.episodes)-1:
                done = True
                if self.logger:
                    wandb.log({
                        "run": self.run,
                        "total_NDCG": np.mean(np.asarray(self.total_ndsg)),
                        "total_MAP": np.mean(np.asarray(self.total_mape))
                    })
                self.total_ndsg = []
                self.total_mape = []
                self.run += 1
                self.episode_num = 0
            ob = self.reset()
        else:
            self.user_hist.append(self.current_episode['user_id'].values[self.steps])
            self.item_hist.append(self.current_episode['item_id'].values[self.steps])    
         
        return np.asarray(ob), reward, done, {}

    def reset(self):
        self.user_hist = []
        self.item_hist = []
        self.relevance_hist = []
        self.total_episodes += 1
        self.episode_num += 1
        #   if self.episode_num == len(self.episodes):
        #      self.episode_num = 0
        
        self.steps = 0 
        try:
            mask = self.log_data['user_id'] == self.episodes[self.episode_num]
        except:
            print(self.episode_num, self.episodes)
        self.current_episode = self.log_data[mask]
        # print(self.current_episode['user_id'])
        self.user_hist.append(self.current_episode['user_id'].values[0])
        self.item_hist.append( self.current_episode['item_id'].values[0])
        self.original = original_for_user(self.log_data, self.current_episode['user_id'].values[0], k = self.top_k)
        obs = self.current_episode['user_id'].values[0], \
                       self.current_episode['item_id'].values[0]
        #  print( np.asarray(obs))
        return np.asarray(obs)


