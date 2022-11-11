import numpy as np
import pandas as pd
from d3rlpy.dataset import Episode, TransitionMiniBatch
from typing import Any, Callable, Iterator, List, Optional, Tuple, Union, cast
from d3rlpy.metrics.scorer import AlgoProtocol

def collect_top_actions(algo, episode, item_mapping, top_k,
                        emb_size = 64, use_user_emb = False, count_of_actions = -1):
    top_k_preds_by_steps = []
    count_of_actions = len(episode.actions) if count_of_actions == -1 else count_of_actions
    for _ in range(count_of_actions):
        full_predicted_actions = []
        actions = algo.predict([episode.observations[0]])
        action_embedding = item_mapping[actions[0]]

        if use_user_emb:
            idx = episode.observations[0, -emb_size:]
            new_observation = np.append(episode.observations[0, emb_size:-emb_size], action_embedding, axis=0)
            new_observation = np.append(new_observation, idx, axis=0)
        else:
            new_observation = np.append(episode.observations[0, emb_size:], action_embedding, axis=0)

        for i in range(top_k):
            full_predicted_actions.append(actions[0])
            actions = algo.predict([new_observation])
            action_embedding = item_mapping[actions[0]]
            # full_predicted_actions.append(actions[0])
            if use_user_emb:
                new_observation = np.append(new_observation[emb_size:-emb_size], action_embedding, axis=0)
                new_observation = np.append(new_observation, idx, axis=0)
            else:
                new_observation = np.append(new_observation[:-emb_size], action_embedding, axis=0)
        top_k_preds_by_steps.append(full_predicted_actions)
    # print("top-k len", len(top_k_preds_by_steps))
    # print("total len", np.asarray(top_k_preds_by_steps))
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
    # print(in_positive)
    # print(len(prediction))
    # print("------------------")
    return in_positive / len(prediction), in_negative/len(prediction)

def evaluate_in_positive(top_k=10, inv_user_mapping = None, item_mapping=None,
                         original_test_logs=None, reward_tresh=3, emb_size = 64,
                         framestack = 5, use_user_emb = 1, user_id = 'user_id', rating = 'rating',
                         item_id = 'item_id', logger = None):
    def scorer(
            algo: AlgoProtocol, episodes: List[Episode]
    ) -> float:
        total_values = []
        action_set = []
        for episode in episodes:
           # print(emb_size)
            top_k_preds_by_steps = collect_top_actions(algo, episode, item_mapping, top_k,
                                emb_size = emb_size, use_user_emb=False, count_of_actions=1)
            top_k_preds_by_steps = np.asarray(top_k_preds_by_steps).ravel()
            pkey = episode.observations[-1].reshape(framestack + use_user_emb, emb_size)[-1]
            for okey in inv_user_mapping:
                if np.mean(np.abs(np.asarray(pkey) - np.asarray(okey))) < 0.1:
                    key = okey
            user_idx = inv_user_mapping[tuple(key)]
            user_log = original_test_logs[original_test_logs[user_id] == user_idx]
            positive, _ = part_of_positive_negative(top_k_preds_by_steps, user_log, rating, item_id, reward_tresh)
            total_values.append(positive)
        if logger:
            logger.log({"evaluate_in_positive": np.mean(total_values)})
        return np.mean(total_values)
    return scorer

def evaluate_in_negative(top_k=10, inv_user_mapping = None, item_mapping=None,
                         original_test_logs=None, reward_tresh=3, emb_size = 64,
                         framestack = 10, use_user_emb = 1, logger = None,
                         user_id = 'user_id', rating = 'rating', item_id = 'item_id'):
    def scorer(
            algo: AlgoProtocol, episodes: List[Episode]
    ) -> float:
        total_values = []
        action_set = []
        for episode in episodes:
         #   print(emb_size)
            top_k_preds_by_steps = collect_top_actions(algo, episode, item_mapping, top_k,
                                emb_size = emb_size, use_user_emb=False, count_of_actions=5)
            top_k_preds_by_steps = np.asarray(top_k_preds_by_steps).ravel()

            pkey = episode.observations[-1].reshape(framestack + use_user_emb, emb_size)[-1]
            for okey in inv_user_mapping:
                if np.mean(np.abs(np.asarray(pkey) - np.asarray(okey))) < 0.1:
                    key = okey
            user_idx = inv_user_mapping[tuple(key)]
            user_log = original_test_logs[original_test_logs[user_id] == user_idx]
            _, negative = part_of_positive_negative(top_k_preds_by_steps, user_log, rating, item_id, reward_tresh)
            total_values.append(negative)
        if logger:
            logger.log({"evaluate_in_negative":np.mean(total_values)})
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

