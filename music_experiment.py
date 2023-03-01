from d3rlpy.algos import CQL
import pandas as pd

from encoders import ActorEncoderFactory
from recsys_mdp.recsys_mdp import FullUserHistoryBasedRecSysMDP
from recsys_mdp.utils import to_d3rlpy_form_ND
from recsys_mdp.reward_functions import monotony_reward, relevance_based_reward, condition_reward
from recsys_mdp.action_function import discrete_relevance_action, continuous_relevance_action, next_item_action

from metrics import true_ndcg
import wandb

wandb.init(project="test_refactored", group="sber_zvuk")

if __name__ == "__main__":
    col_mapping = {'user_col_name': 'user_idx',
                   'item_col_name': 'item_idx',
                   'reward_col_name': 'rating',
                   'timestamp_col_name': 'ts'}
    emb_size = 8
    framestask = 5
    data = pd.read_csv("ratings_rev_top10000_users.csv")
    data['rating'] = data['rating'].astype(float)
    data[data['rating'] < 3]['rating'] = -data[data['rating'] < 3]['rating']
    data = data.sort_values(['event_dt'])
    best_users_idx = data['user_id'].value_counts()[:800].index
    user_train_idx = [idx for i, idx in enumerate(best_users_idx) if i % 100 != 0]
    user_test_idx = [idx for i, idx in enumerate(best_users_idx) if i % 100 == 0]

    data['user_idx'] = data['user_id']
    data['old_idx'] = data['user_id']
    filtered_raitings = data.set_index('user_id')
    filtered_raitings = filtered_raitings.loc[best_users_idx]
    filtered_raitings = filtered_raitings.reset_index(drop=False)

    keys = list(set(filtered_raitings['track_id']))
    item_mapping = dict(zip(keys, list(range(1, len(keys) + 1))))
    filtered_raitings['item_idx'] = filtered_raitings['track_id'].apply(lambda x: item_mapping[x])

    keys = list(set(filtered_raitings['user_idx']))
    user_mapping = dict(zip(keys, list(range(1, len(keys) + 1))))
    filtered_raitings['user_idx'] = filtered_raitings['user_idx'].apply(lambda x: user_mapping[x])

    print("Data filterd")
    print(filtered_raitings[:3])
    mdp_train = FullUserHistoryBasedRecSysMDP(load_from_file=False, dataframe=filtered_raitings,
                                                  data_mapping=col_mapping, framestack=framestask,
                                                  reward_function=monotony_reward,
                                                  action_function=continuous_relevance_action)

    states, rewards, actions, termations = mdp_train.create_mdp()
    dataset_train = to_d3rlpy_form_ND(states, rewards, actions, termations)
    print("Dataset actions: ")
    print(dataset_train.actions[:5])
    print("Data generated.")
    model_params=[943, 1682, 8, 16, 5, 256]
    actor_encoder_factory = ActorEncoderFactory(
        *model_params
    )
    print("Encoder generated.")

    part_to_test = 100000
    users = dataset_train.observations[:,-1][:part_to_test]
    observations = dataset_train.observations[:][:part_to_test]
    scorer = true_ndcg(users, observations, top_k=10)

    algo = CQL(use_gpu=True, encoder_factory = actor_encoder_factory, batch_size = 32)
    algo.fit(dataset_train, n_epochs = 32, eval_episodes=dataset_train[:part_to_test], scorers={'ndcg_sorer': scorer})