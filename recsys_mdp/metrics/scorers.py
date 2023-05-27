import numpy as np
from recsys_mdp.metrics.metrics import ndcg, hit_rate
from scipy.stats import pearsonr, spearmanr

MAX_EPISODE_LEN = 40
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
    (interactive_items, true_episode_items, users_interests, not_interactive_items)
    :return:
    """
    full_hitrate = []
    hitrate = []
    negative_hitrate = []
    for interact in interaction_result:
        interact = [np.asarray(res) for res in interact]
        interactive_items, true_episode_items, \
            users_interests, not_interactive_items, episode_rewards, _ = interact


        positive_multp = len(true_episode_items[episode_rewards>0]) / MAX_EPISODE_LEN
        interactive_hit_rate = hit_rate(interactive_items[episode_rewards>0],
                                        true_episode_items[episode_rewards>0]) * positive_multp
        hitrate.append(interactive_hit_rate)

        negative_multp = len(true_episode_items[episode_rewards > 0]) / MAX_EPISODE_LEN
        interactive_hit_rate_negative = hit_rate(interactive_items[episode_rewards < 0],
                                        true_episode_items[episode_rewards < 0]) * negative_multp
        negative_hitrate.append(interactive_hit_rate_negative)

        episode_size_multp = len(interactive_items) / MAX_EPISODE_LEN
        interactive_hit_rate_full = hit_rate(interactive_items, users_interests) * episode_size_multp
        full_hitrate.append(interactive_hit_rate_full)

    return {'on_episode_positive':np.mean(hitrate),
            'on_episode_negative':np.mean(negative_hitrate),
            'on_history':np.mean(full_hitrate)}

def static_hit_rates(interaction_result):
    """

        :param interaction_result: arrays of tuple with
        (interactive_items, original_items, users_interests, not_interactive_items)
        :return:
        """
    full_hitrate = []
    hitrate = []
    negative_hitrate = []
    for interact in interaction_result:
        interact = [np.asarray(res) for res in interact]
        interactive_items, original_items, \
            users_interests, not_interactive_items, episode_reward, _ = interact


        posititve_multp = len(original_items[episode_reward>0]) / MAX_EPISODE_LEN
        interactive_hit_rate = hit_rate(not_interactive_items[episode_reward>0], original_items[episode_reward>0]) * posititve_multp
        hitrate.append(interactive_hit_rate )
        negative_multp = len(original_items[episode_reward<0]) / MAX_EPISODE_LEN
        interactive_hit_rate_negative = hit_rate(not_interactive_items[episode_reward<0], original_items[episode_reward<0]) * negative_multp
        negative_hitrate.append(interactive_hit_rate_negative )

        episode_size_multp = len(not_interactive_items) / MAX_EPISODE_LEN
        interactive_hit_rate_full = hit_rate(not_interactive_items, users_interests) * episode_size_multp
        full_hitrate.append(interactive_hit_rate_full)

    return {'on_episode_positive':np.mean(hitrate),
            'on_episode_negative': np.mean(negative_hitrate),
            'on_history':np.mean(full_hitrate)}

def preference_correlation(interaction_result):
    mean_corr = []
    for interact in interaction_result:

        episode_size_multp = len(interact[-1]) / MAX_EPISODE_LEN #count of interactions
        d1 = interact[-2] + 2
        d2 = interact[-1]
      #  print(d1, d2)
        d1 = np.asarray(d1)
        d2 = np.asarray(d2)
        idx = np.argsort(d1)


        pearson_corr, _ = pearsonr(d1[idx], d2[idx])
       # print(pearson_corr * episode_size_multp)

      #  print(pearson_corr)
        not_nan_mask = ~np.isnan(pearson_corr)
        if  not_nan_mask:
            mean_corr.append(pearson_corr * episode_size_multp)
    return {'preference_correlation': np.mean(mean_corr)}