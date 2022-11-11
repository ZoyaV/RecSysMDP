import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = 'python'


from d3rlpy.algos import DiscreteCQL

from recsys_mdp.recsys_mdp import RecSysMDP
from recsys_mdp.utils import to_d3rlpy_form
from recsys_mdp.d3rlpy_recsys_metrics import evaluate_in_positive, \
                                            evaluate_in_negative
import wandb
wandb.init(project="Right MDP", group = "MovieLens_CQL_with_negative")

EMBEDDING_SIZE = 64


if __name__ == "__main__":
    # col_mapping = {'user_col_name': 'user_id',
    #                'item_col_name': 'item_id',
    #                'reward_col_name': 'rating',
    #                'timestamp_col_name': 'timestamp'}
    #
    # ml = rs_datasets.MovieLens()
    # raitings = ml.ratings
   # make_datasets(raitings[raitings['user_id'] < 100], col_mapping, 0.2)
    mdp_train = RecSysMDP(load_from_file=True, path='./data/ml_100k_first100_train_4')
    mdp_test = RecSysMDP(load_from_file=True, path='./data/ml_100k_first100_test_14')
    #
    dataset_train = to_d3rlpy_form(mdp_train.states, mdp_train.rewards,
                                   mdp_train.actions, mdp_train.termations)

    dataset_test = to_d3rlpy_form(mdp_test.states, mdp_test.rewards,
                                   mdp_test.actions, mdp_test.termations)

    algo = DiscreteCQL(use_gpu=True)
    part_of_positive = evaluate_in_positive(top_k=10, inv_user_mapping=mdp_test.inv_user_mapping,
                                            item_mapping=mdp_test.item_mapping,
                                            original_test_logs=mdp_test.dataframe, reward_tresh=3, emb_size=64,
                                            framestack=5, use_user_emb=1, logger=wandb,
                                            user_id='user_id', rating='rating', item_id='item_id')

    part_of_negative = evaluate_in_negative(top_k=10, inv_user_mapping=mdp_test.inv_user_mapping,
                                            item_mapping=mdp_test.item_mapping,
                                            original_test_logs=mdp_test.dataframe, reward_tresh=3, emb_size=64,
                                            framestack=5, use_user_emb=1, logger=wandb,
                                            user_id='user_id', rating='rating', item_id='item_id')

    algo.fit(dataset_train, eval_episodes=dataset_test, n_epochs=30, scorers={
        'NegativeScore': part_of_negative,
        'PositiveScore': part_of_positive})



