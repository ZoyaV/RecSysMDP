import wandb
wandb.init(project="Right MDP", group = "MovieLens_CQL_with_negative")

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = 'python'


from d3rlpy.algos import DiscreteCQL

from recsys_mdp.recsys_mdp import RecSysMDP
from recsys_mdp.utils import to_d3rlpy_form
from recsys_mdp.d3rlpy_recsys_metrics import evaluate_in_positive, \
                                            evaluate_in_negative
import rs_datasets
from recsys_mdp.utils import make_datasets


wandb.init(project="Right MDP", group = "MovieLens_CQL_with_negative")

EMBEDDING_SIZE = 8


if __name__ == "__main__":
    col_mapping = {'user_col_name': 'user_idx',
                   'item_col_name': 'item_idx',
                   'reward_col_name': 'rating',
                   'timestamp_col_name': 'timestamp'}

    ml = rs_datasets.MovieLens()
    raitings = ml.ratings
    best_users_idx = raitings['user_id'].value_counts()[200:210].index
    raitings['user_idx'] = raitings['user_id']
    filtered_raitings = raitings.set_index('user_id')
    filtered_raitings = filtered_raitings.loc[best_users_idx]
    filtered_raitings = filtered_raitings.reset_index(drop=False)
    #filtered_raitings['user_id'] = filtered_raitings['user_idx']

    keys = list(set(filtered_raitings['item_id']))
    item_mapping = dict(zip(keys, list(range(1, len(keys)+1))))
    filtered_raitings['item_idx'] = filtered_raitings['item_id'].apply(lambda x: item_mapping[x])

    keys = list(set(filtered_raitings['user_idx']))
    user_mapping = dict(zip(keys, list(range(1, len(keys)+1))))
    filtered_raitings['user_idx'] = filtered_raitings['user_idx'].apply(lambda x: user_mapping[x])

   # print(raitings)

    print(set(filtered_raitings['user_idx'].values))
    print(set(filtered_raitings['item_idx'].values))
    mdp_train, mdp_test = make_datasets(filtered_raitings, col_mapping, 0.2, data_name = "ml_100k_first_1000", framestask=10, emb_size=EMBEDDING_SIZE)

 #  mdp_train = RecSysMDP(load_from_file=True, path='./data/ml_100k_first_1000_train_27')
  # mdp_test = RecSysMDP(load_from_file=True, path='./data/ml_100k_first_1000_test_63')
    #
    dataset_train = to_d3rlpy_form(mdp_train.states, mdp_train.rewards,
                                   mdp_train.actions, mdp_train.termations)

    dataset_test = to_d3rlpy_form(mdp_test.states, mdp_test.rewards,
                                   mdp_test.actions, mdp_test.termations)

    algo = DiscreteCQL(use_gpu=True)
    part_of_positive = evaluate_in_positive(top_k=10, inv_user_mapping=mdp_train.inv_user_mapping,
                                            item_mapping=mdp_train.item_mapping,
                                            original_test_logs=mdp_train.dataframe, reward_tresh=3, emb_size=EMBEDDING_SIZE,
                                            framestack=10, use_user_emb=1, logger=wandb,
                                            user_id='user_idx', rating='rating', item_id='item_idx')

    part_of_negative = evaluate_in_negative(top_k=10, inv_user_mapping=mdp_train.inv_user_mapping,
                                            item_mapping=mdp_train.item_mapping,
                                            original_test_logs=mdp_train.dataframe, reward_tresh=3, emb_size=EMBEDDING_SIZE,
                                            framestack=10, use_user_emb=1, logger=wandb,
                                            user_id='user_idx', rating='rating', item_id='item_idx')

    #print(help(algo.fit))
    algo.fit(dataset_train, eval_episodes=dataset_train[:1000], n_steps_per_epoch=2000, n_epochs=20, scorers={
        'NegativeScore': part_of_negative,
        'PositiveScore': part_of_positive})

    algo.save_model('CQL.pt')



