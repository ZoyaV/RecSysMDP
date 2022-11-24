from recsys_mdp.embedddings import random_embeddings
import pandas as pd
import numpy as np
from d3rlpy.algos import DiscreteCQL, DiscreteSAC

from recsys_mdp.recsys_mdp import SplitByFailureRecSysMDP
from recsys_mdp.utils import to_d3rlpy_form, to_d3rlpy_form_ND
from recsys_mdp.d3rlpy_recsys_metrics import hit_rate
from models import CustomEncoderFactory
import wandb

if __name__ == "__main__":
    col_mapping = {'user_col_name': 'user_idx',
                   'item_col_name': 'item_idx',
                   'reward_col_name': 'rating',
                   'timestamp_col_name': 'ts'}
    emb_size = 10
    framestask = 10
    data = pd.read_csv("clickstream_1ku.csv")
    data['ts'] = data['timestamp'].apply(pd.to_datetime)
    data = data.sort_values(['ts'])
    best_users_idx = data['user_id'].value_counts()[:20].index
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

    print("Prepare data: ")
    print(filtered_raitings[:5])
    user_mapping, inv_user_mapping = random_embeddings(filtered_raitings[col_mapping['user_col_name']], emb_size=emb_size)
    items_mapping, _ = random_embeddings(filtered_raitings[col_mapping['item_col_name']], emb_size=emb_size)

    mdp_train = SplitByFailureRecSysMDP(load_from_file=False, dataframe=filtered_raitings, data_mapping=col_mapping,
                          framestack=framestask, emb_size=emb_size, user_mapping=user_mapping,
                          item_mapping=items_mapping, use_user_embedding=True, inv_user_mapping=inv_user_mapping)
    mdp_train.create_mdp()


    dataset_train = to_d3rlpy_form_ND(mdp_train.states, mdp_train.rewards,
                                   mdp_train.actions, mdp_train.termations, N=3, framestack = framestask, emb_size = emb_size)

    print("Prepare data for d3rlpy: ")

    print(np.min(dataset_train.observations))
    print(np.max(dataset_train.observations))
    print(dataset_train.observations.shape)
    print(dataset_train.terminals[:30])
    print(dataset_train.actions[:30])
    print(dataset_train.observations[:30])
    print(dataset_train.rewards[:30])



    algo = DiscreteCQL(use_gpu=True, encoder_factory = CustomEncoderFactory(512),batch_size = 32)
    part_of_positive = hit_rate(top_k=10, inv_user_mapping=mdp_train.inv_user_mapping,
                                            item_mapping=mdp_train.item_mapping,
                                            original_test_logs=mdp_train.dataframe, reward_tresh=2,
                                            emb_size=emb_size, use_user_emb=False, logger=wandb,
                                            user_id='user_idx', rating='rating', item_id='item_idx')

    wandb.init(project="Right MDP", group="SberZvuk_CQL_with_negative")
    algo.fit(dataset_train, eval_episodes=dataset_train, n_steps_per_epoch=2000,
             n_epochs=50, scorers={'PositiveScore': part_of_positive})
    algo.save_model('CQL_muse.pt')