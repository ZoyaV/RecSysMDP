import numpy as np
from metrics.metrics import ndcg, hit_rate
def log_covarage(total_prediction, users_interests, top_k):
    distribution = np.asarray(total_prediction)
    best_items = np.argmax(distribution, axis = 0)
    return len(set(best_items))

def total_ndcg(total_prediction, users_interests, top_k):
    metricv = []
    for i,user in enumerate(total_prediction[0]):
        ratings = total_prediction[1][i]
        interests = np.argsort(ratings)
        undcg = ndcg(top_k, interests, users_interests[user])
        metricv.append(undcg)
    return np.mean(metricv)

def interactive_hit_rates(interaction_result):
    """

    :param interaction_result: arrays of tuple with
    (interactive_items, original_items, users_interests, not_interactive_items)
    :return:
    """
    full_hitrate = []
    hitrate = []
    for interact in interaction_result:
        interactive_items, original_items, \
            users_interests, not_interactive_items = interact

        interactive_hit_rate = hit_rate(interactive_items, original_items)
        interactive_hit_rate_full = hit_rate(interactive_items, users_interests)
        full_hitrate.append(interactive_hit_rate_full)
        hitrate.append(interactive_hit_rate)
    return {'on_episode':np.mean(full_hitrate), 'on_history':np.mean(hitrate)}

def static_hit_rates(interaction_result):
    """

        :param interaction_result: arrays of tuple with
        (interactive_items, original_items, users_interests, not_interactive_items)
        :return:
        """
    full_hitrate = []
    hitrate = []
    for interact in interaction_result:
        interactive_items, original_items, \
            users_interests, not_interactive_items = interact

        interactive_hit_rate = hit_rate(not_interactive_items, original_items)
        interactive_hit_rate_full = hit_rate(not_interactive_items, users_interests)
        full_hitrate.append(interactive_hit_rate_full)
        hitrate.append(interactive_hit_rate)
    return {'on_episode':np.mean(full_hitrate), 'on_history':np.mean(hitrate)}