import numpy as np

from recsys_mdp.mdp.base import ITEM_ID_COL, USER_ID_COL
from recsys_mdp.logging.d3rlpy_loggers import (
    base_ndcg, tsne_embeddings, episode_hit_rate, tsne_encoder
)
from recsys_mdp.logging.logger import Logger
from recsys_mdp.metrics.scorers import (
    log_covarage, total_ndcg, interactive_hit_rates, static_hit_rates, preference_correlation
)
from recsys_mdp.logging.visual_logger import tsne, items_distribution


def init_tsne_vis(test_users, test_items):
    scorer = tsne_embeddings(test_users, test_items)
    return scorer


def init_tsne_encoder(test_users, test_items):
    scorer = tsne_encoder(test_users, test_items)
    return scorer


def init_hit_rate(state_tail, test_items, test_users_or, top_k, discrete):
    test_states, users_interests = get_test_observation(state_tail, test_items, test_users_or, discrete)
    scorer = episode_hit_rate(top_k, users_interests)
    return scorer


def get_test_observation(state_tail, test_items, test_users_or, discrete):
    state_tail = np.asarray(state_tail)
    test_states = []
    true_items = dict()
    test_users = np.unique(test_users_or)
    if discrete:
        for user in test_users:
            true_items[user] = test_items[test_users_or == user]
            user_mask = state_tail[:, -1] == user
            try:
                new_obs = state_tail[user_mask][0]
            except:
                continue
            test_states.append(new_obs)
    else:
        for user in test_users:
            user_mask = state_tail[:, -1] == user
            true_items[user] = test_items[test_users_or == user]
            test_items_unique = np.unique(test_items)
            for item in test_items_unique:
                try:
                    new_obs = state_tail[user_mask][0]
                except:
                    continue
                new_obs[0:-2] = new_obs[1:-1]
                new_obs[-2] = item
                test_states.append(new_obs)
    test_states = np.asarray(test_states)
    return test_states, true_items


def make_observation(state_tail, test_users_or, test_items, discrete=True):
    test_states, true_items = get_test_observation(state_tail, test_items, test_users_or, discrete)
    return test_states, true_items


def init_next_step_scorer(state_tail, test_users_or, test_items, top_k, tresh, discrete=True):
    test_states, true_items = get_test_observation(state_tail, test_items, test_users_or, discrete)
    scorer = base_ndcg(test_states, true_items, tresh, top_k, discrete)
    return scorer


def init_scorers(state_tail, test_values, top_k, tresh, metrics, prediction_type):
    scorers = dict()
    return scorers


def init_logger(
        test_mdp, state_tail, data, top_k, tresh, metrics, prediction_type,
        wandb_logger=None
):
    fake_mdp, users_interests = get_test_observation(
        state_tail, data[ITEM_ID_COL].values, data[USER_ID_COL].values, prediction_type
    )

    static = dict()
    interactive = dict()
    visualizations = []

    if 'coverage' in metrics:
        static['coverage'] = log_covarage
    if 'ndcg' in metrics:
        static['ndcg'] = total_ndcg
    if 'ihitrate' in metrics:
        interactive['ihitrate'] = interactive_hit_rates
    if 'stat_hitrate' in metrics:
        interactive['stat_hitrate'] = static_hit_rates
    if 'items_distribution' in metrics:
        visualizations.append(items_distribution)
    if 'tsne' in metrics:
        visualizations.append(tsne)
    if 'PC' in metrics:
        interactive['preference_correlation'] = preference_correlation

    #print(visualizations)
    logger = Logger(
        interactive_mdp=test_mdp, user_interests=users_interests,
        fake_mdp=fake_mdp, top_k=top_k,
        static_scorers=static, interactive_scorers=interactive, visual_loggers=visualizations,
        wandb_logger=wandb_logger
    )

    return logger
