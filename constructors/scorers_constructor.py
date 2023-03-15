import numpy as np
from metrics.metrics import base_ndcg, tsne_embeddings, episode_hit_rate
def init_tsne_vis(test_users, test_items):
    scorer = tsne_embeddings(test_users, test_items)
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

def init_next_step_scorer(state_tail, test_users_or, test_items, \
                          top_k, tresh, discrete = True):
    test_states, true_items = get_test_observation(state_tail, test_items, test_users_or, discrete)
    scorer = base_ndcg(test_states, true_items, tresh,top_k, discrete)
    return scorer

def init_scorers(config, state_tail, test_values, prediction_type):
    top_k = config['experiment']['top_k']
    tresh = config['experiment']['scorer']['tresh']
    scorers = dict()
    if 'rating_scorer' in config['experiment']['scorer']['metrics']:
        rating_scorer = init_next_step_scorer(state_tail, test_values['full_users'], \
                                              test_values['full_items'], top_k, tresh, discrete=prediction_type)
        scorers['rating_scorer'] = rating_scorer

    if 'tsne' in config['experiment']['scorer']['metrics']:
        tsne = init_tsne_vis(test_values['users_unique'], test_values['items_unique'])
        scorers['tsne'] = tsne

    if 'hit_rate' in config['experiment']['scorer']['metrics']:
        hit_rate = init_hit_rate(state_tail, test_values['full_items'], test_values['full_users'], \
                                 top_k, discrete=prediction_type)
        scorers['hit_rate'] = hit_rate
    return scorers