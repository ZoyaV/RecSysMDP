from typing import List

import numpy as np
from d3rlpy.dataset import Episode
from d3rlpy.metrics.scorer import AlgoProtocol


def collect_top_actions(algo, episode, item_mapping, top_k,
                        emb_size = 64, use_user_emb = False,
                        count_of_actions = -1):
    top_k_preds_by_steps = []
    count_of_actions = len(episode.actions) if count_of_actions == -1 else count_of_actions  # for how steps in os calk TOP-k
    for _ in range(count_of_actions):
        full_predicted_actions = []
        actions = algo.predict([episode.observations[0]])
        action_embedding = item_mapping[actions[0]]

        if use_user_emb:
            idx = episode.observations[0, -emb_size:]
            new_observation = np.append(episode.observations[0, emb_size:-emb_size], action_embedding, axis=0)
            new_observation = np.append(new_observation, idx, axis=0)
        else:
            #TODO: it should work for vector and img obs. Now only image obs like available.
           # new_observation = np.append(episode.observations[0, emb_size:], action_embedding, axis=0)
            new_observation = episode.observations[0].copy()
            #? (3) framestack em_size
            new_observation[0,:-1] = new_observation[0,1:]
            new_observation[0,-1] = action_embedding
        for i in range(top_k):
            full_predicted_actions.append(actions[0])
            actions = algo.predict([new_observation])
          #  print(actions)
            action_embedding = item_mapping[actions[0]]*255
            action_embedding = action_embedding.astype(np.uint8)
            # full_predicted_actions.append(actions[0])
            if use_user_emb:
                new_observation = np.append(new_observation[emb_size:-emb_size], action_embedding, axis=0)
                new_observation = np.append(new_observation, idx, axis=0)
            else:
                #TODO: it should work for vector and img obs. Now only image obs like available.
               # new_observation = np.append(new_observation[:-emb_size], action_embedding, axis=0)
                new_observation[0,:-1] = new_observation[0,1:]
                new_observation[0,-1] = action_embedding
        top_k_preds_by_steps.append(full_predicted_actions)
    return top_k_preds_by_steps

def part_of_positive_negative(prediction, user_log, rating, item_id, tresh):
    positive = set(list(user_log[user_log[rating] >= tresh][item_id]))
    negative = set(list(user_log[user_log[rating] < tresh][item_id]))
    in_positive = 0
    in_negative = 0
    actions = list(set(prediction))
    for action in actions:
        if action in positive:
            in_positive += 1
        if action in negative:
            in_negative += 1
    return in_positive / len(prediction), in_negative/len(prediction)

def hit_rate(positive = True, key_scrapper = lambda obs: obs[-1][-1][0], top_k=10,
                         inv_user_mapping = None, item_mapping=None,
                         original_test_logs=None, reward_tresh=3, emb_size = 64,
                         use_user_emb = False, user_id = 'user_id', rating = 'rating',
                         item_id = 'item_id', logger = None):
    print(inv_user_mapping)
    def scorer(
            algo: AlgoProtocol, episodes: List[Episode]
    ) -> float:
        total_values = []
        action_set = []
        for episode in episodes:
            top_k_preds_by_steps = collect_top_actions(algo, episode, item_mapping, top_k,
                                emb_size = emb_size, use_user_emb=use_user_emb, count_of_actions=1)
            top_k_preds_by_steps = np.asarray(top_k_preds_by_steps).ravel()

            # Get user embedding and translate to user index
          #  print(episode.observations[-1])
            pkey = key_scrapper(episode.observations)
          #  print(inv_user_mapping)
          #  pkey = pkey * 100

            for okey in inv_user_mapping:
               # print(okey, pkey)

                d = np.mean(np.abs(np.asarray(pkey) - np.asarray(okey[0])))
                if d < 0.5:
                    key = okey
         #   print("Found: ")
          #  print(pkey)
            user_idx = inv_user_mapping[tuple(key)]

            # Calculate part of prediction in postive/negative part
            user_log = original_test_logs[original_test_logs[user_id] == user_idx]
            if positive:
                value, _ = part_of_positive_negative(top_k_preds_by_steps, user_log, rating, item_id, reward_tresh)
            else:
                _, value = part_of_positive_negative(top_k_preds_by_steps, user_log, rating, item_id, reward_tresh)
            total_values.append(value)
        if logger:
            if positive:
                logger.log({"evaluate_in_positive": np.mean(total_values)})
            else:
                logger.log({"evaluate_in_negative": np.mean(total_values)})
        return np.mean(total_values)
    return scorer

def episode_hit_rate(top_k=10, item_mapping=None,emb_size = 64,
                         use_user_emb = False, logger = None):
    def scorer(
            algo: AlgoProtocol, episodes: List[Episode]
    ) -> float:
        total_values = []
        action_set = []
        for episode in episodes:
            top_k_preds_by_steps = collect_top_actions(algo, episode, item_mapping, top_k,
                                emb_size = emb_size, use_user_emb=use_user_emb, count_of_actions=1)
            top_k_preds_by_steps = np.asarray(top_k_preds_by_steps).ravel()

            # Get user embedding and translate to user index
            episode_actions = episode.actions
            count = 0
            for pred in top_k_preds_by_steps:
                if pred in episode_actions:
                    count+=1
            value = count/len(episode_actions)
            total_values.append(value)
        if logger:
                logger.log({"pred_in_episode_action": np.mean(total_values)})
        return np.mean(total_values)
    return scorer

def evaluate_covarage(top_k=10, item_mapping=None, reward_tresh=3, logger = None):
    def scorer(
            algo: AlgoProtocol, episodes: List[Episode]
    ) -> float:
        total_values = []
        action_set = []
        action_original_set = []
        for episode in episodes:
            top_k_preds_by_steps = collect_top_actions(algo, episode, item_mapping, top_k, reward_tresh,
                                emb_size=64, use_user_emb=False, count_of_actions=5)
            top_k_preds_by_steps = np.asarray(top_k_preds_by_steps).ravel()
            original_actions = episode.actions
            action_original_set += list(episode.actions)
            action_set += list(top_k_preds_by_steps)
        action_original_set = set(action_original_set)
        action_set = set(action_set)
        intersection = list(action_original_set&action_set)
        value = len(intersection)/len(list(action_original_set))
        if logger:
            logger.log({"coverage":value})
        return len(intersection)/len(list(action_original_set))
    return scorer

