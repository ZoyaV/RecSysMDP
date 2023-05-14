import numpy as np
from recsys_mdp.metrics.metrics import ndcg, hit_rate
from scipy.stats import pearsonr, spearmanr

def log_covarage(total_prediction, users_interests, top_k):
    coverage_by_user = []
    distribution = np.asarray(total_prediction[1])
    best_items = np.argmax(distribution, axis = 1)

    for i in range(len(total_prediction[0])):
        ratings = total_prediction[1][i]

       # items = np.argmax(ratings)
        #print(items)
        cov = np.sum(ratings > 0.99)/len(total_prediction[1])
        coverage_by_user.append(cov)
    #print(best_items)
    return np.mean(coverage_by_user)

def total_ndcg(total_prediction, users_interests, top_k):
    metricv = []
    for i,user in enumerate(total_prediction[0]):
        ratings = total_prediction[1][i]
        interests = np.argsort(ratings)[::-1]
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
            users_interests, not_interactive_items, _, _ = interact

        interactive_hit_rate = hit_rate(interactive_items, original_items)
        interactive_hit_rate_full = hit_rate(interactive_items, users_interests)
        full_hitrate.append(interactive_hit_rate_full)
        hitrate.append(interactive_hit_rate)
    return {'on_episode':np.mean(hitrate), 'on_history':np.mean(full_hitrate)}

def preference_correlation(interaction_result):
    mean_corr = []
    for interact in interaction_result:
        d1 = interact[-2] + 2
        d2 = interact[-1]
        d1 = np.asarray(d1)
        d2 = np.asarray(d2)
        idx = np.argsort(d1)

      #  print(d1[idx], d2[idx])
        pearson_corr, _ = pearsonr(d1[idx], d2[idx])

      #  print(pearson_corr)
        not_nan_mask = ~np.isnan(pearson_corr)
        if  not_nan_mask:
            mean_corr.append(pearson_corr)
    return {'preference_correlation': np.mean(mean_corr)}
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
            users_interests, not_interactive_items, _, _ = interact

        interactive_hit_rate = hit_rate(not_interactive_items, original_items)
        interactive_hit_rate_full = hit_rate(not_interactive_items, users_interests)
        full_hitrate.append(interactive_hit_rate_full)
        hitrate.append(interactive_hit_rate)
    return {'on_episode':np.mean(hitrate), 'on_history':np.mean(full_hitrate)}