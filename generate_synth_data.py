import argparse
import os
import pickle
from itertools import count
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from d3rlpy.base import LearnableBase

from als_model import ALSRecommender
from recsys_mdp.generators.datasets.synthetic.relevance import similarity
from recsys_mdp.generators.scenarios.mdp_next_item_integration import (
    NextItemEnvironment,
    TypesResolver
)
from recsys_mdp.generators.utils.config import (
    GlobalConfig
)
from recsys_mdp.generators.utils.lazy_imports import lazy_import

wandb = lazy_import('wandb')


def generate_episode(
        env, model, framestack_size=10, user_id = None, log_sat=False, logger=None
):
    orig_user_id = user_id
    user_id = env.reset(user_id=user_id)
    trajectory = []

    # [N last item_ids] + [user_id]
    fake_obs = np.random.randint(0, env.n_items, framestack_size).tolist() + [user_id]
    obs = np.asarray(fake_obs)
    while True:
        try:
            item_id = model.predict(obs.reshape(1, -1))[0]
        except:
            item_id = model.predict(obs[:framestack_size].reshape(1, -1))[0]

        obs[:framestack_size - 1] = obs[1:framestack_size]
        obs[-2] = item_id

        timestamp = env.timestamp

        relevance, terminated = env.step(item_id)
        continuous_relevance, discrete_relevance = relevance
        items_top = env.state.ranked_items(with_satiation=True, discrete=True)
        if log_sat and logger is not None:
            hist = (env.state.satiation, np.arange(len(env.state.satiation)+1))
            histogram = wandb.Histogram(np_histogram=hist)
            logger.log({f'user_{orig_user_id}_satiation': histogram})
        # item_id = random.choice(items_top[:10])

        trajectory.append((
            timestamp,
            user_id, item_id,
            continuous_relevance, discrete_relevance,
            terminated,
            items_top[:10]
        ))
        if terminated:
            break
    return trajectory


def poor_model_from_dataset(dataset_path):
    model = ALSRecommender()
    data = pd.read_csv(dataset_path)
    model.fit(data)
    return model


def generate_dataset(gen_conf, env, model):
    config = gen_conf
    samples = []
    for episode in count():
        samples.extend(generate_episode(env, model))
        if config['episodes_per_epoch'] is not None and episode >= config['episodes_per_epoch']:
            break
        if config['samples_per_epoch'] is not None and len(samples) >= config['samples_per_epoch']:
            break
    return samples


@np.printoptions(3)
def make_user_with_stable_interest(user, print_debug=False):
    embeddings = user.embeddings
    # generate user embeddings near the first item cluster
    item_cluster = 0
    _, user_emb = embeddings.item_embeddings_generator.generate_one(cluster_ind=item_cluster)
    # put user to the first item cluster
    # NB: user cluster index from now on is kinda incorrect!
    user.embeddings.users[user.user_id] = user_emb

    if print_debug:
        # most relevant item
        sorted_tastes = user.ranked_items(with_satiation=False, discrete=False)
        top1_item = sorted_tastes[0]
        top1_item_cluster = user.embeddings.item_cluster_ind[top1_item]
        top1_item_emb = user.embeddings.items[top1_item]
        print(f'User with stable interest =>')
        print(f'    Tastes: {user.tastes}')
        print(f'    Top1:   {top1_item_emb}')
        print(f'    Top1Cluster: {top1_item_cluster}')
        relevance, discr_relevance = user.relevance(top1_item, with_satiation=False, consume=False)
        print(f'    Top1Rel: {round(relevance, 3)} | Top1DiscrRel: {discr_relevance}')

        item_clusters = embeddings.item_clusters
        to_item_clusters_relevance = similarity(user.tastes, item_clusters, metric=user.metric)
        print(f'    ToClusterSims: {to_item_clusters_relevance}')
        print()

    user.satiation[:] = 1
    user.satiation[item_cluster] = 0.5
    user.satiation_speed[:] = 0.0


@np.printoptions(3)
def make_user_with_two_interests(user, print_debug=False):
    embeddings = user.embeddings
    item_clusters = embeddings.item_clusters
    # find two the closest clusters
    pairwise_cluster_sim = np.array([
        similarity(item_cluster, item_clusters, metric='l2')
        for item_cluster in item_clusters
    ])
    # zero out the diagonal as its elements obviously equal 1 and will be among maximal values
    pairwise_cluster_sim[np.diag_indices(item_clusters.shape[0])] = 0

    top1, top2 = np.unravel_index(
        np.argmax(pairwise_cluster_sim),
        shape=pairwise_cluster_sim.shape
    )
    top1_emb, top2_emb = embeddings.item_clusters[top1], embeddings.item_clusters[top2]
    # make user embedding to be almost in the middle between them
    user_emb = .6 * top1_emb + .4 * top2_emb
    # put user tastes to the specified place
    # NB: user cluster index from now on is kinda incorrect!
    user.embeddings.users[user.user_id] = user_emb

    if print_debug:
        print(f'User with two interests =>')
        print(f'    Tastes:   {user.tastes}')
        print(f'    Top1Cl:   {top1_emb}')
        print(f'    Top2Cl:   {top2_emb}')
        print(f'    TopTwoClusters: {top1} and {top2}')

        item_clusters = embeddings.item_clusters
        to_item_clusters_relevance = similarity(user.tastes, item_clusters, metric=user.metric)
        print(f'    ToClusterSims: {to_item_clusters_relevance}')
        print()

    user.satiation[:] = 1
    user.satiation[top1] = 0.5
    user.satiation[top2] = 0.5

    user.satiation_speed[top1] = 0.02
    user.satiation_speed[top2] = 0.04


def make_env_setting(config_path="", env_name='default', dataset_path = None):
    # read configs
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    path_object = Path(config_path)
    config_class = GlobalConfig(
        config=config, config_path=path_object, type_resolver=TypesResolver()
    )
    # make conrollable environment
    env_conf = config['env']
    env: NextItemEnvironment = config_class.resolve_object(
        env_conf, object_type_or_factory=NextItemEnvironment
    )

    ### One of user have one stable interest
    make_user_with_stable_interest(env.states[0], print_debug=True)

    ### One two interest with interaction [need to balance]
    make_user_with_two_interests(env.states[6], print_debug=True)

    # make model
    if dataset_path:
        model = poor_model_from_dataset(dataset_path)
    else:
        model_conf = config['model']
        model: LearnableBase = config_class.resolve_object(
            model_conf | dict(use_gpu=False),
            n_actions=env.n_items
        )
    gen_conf = config['generation']

    # save environment object
    with open(f'environments/{env_name}/env.pkl', 'wb') as f:
        pickle.dump(env, f)

    return gen_conf, env, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config')
    parser.add_argument('--env_name', type=str)
    parser.add_argument('--base_data', type=str, default=None)
    args = parser.parse_args()

    if args.config is None:
        args.config = "recsys_mdp/generators/configs/mdp_next_item_integration.yaml"

    directory = f'environments/{args.env_name}'
    if not os.path.exists(directory):
        os.makedirs(directory)

    gen_conf, env, model = make_env_setting(config_path=args.config, env_name=args.env_name, dataset_path=args.base_data)
    trajectories = generate_dataset(gen_conf, env, model)

    # Define the column names
    column_names = ['timestamp', 'user_idx', 'item_idx', 'relevance_cont', 'relevance_int', 'terminated', 'true_top']

    # Create a DataFrame from the list of tuples
    df = pd.DataFrame(trajectories, columns=column_names)
    file_name = f"{directory}/{args.env_name}.csv"

    # Save the DataFrame to a CSV file
    print(f"Data generated to {file_name}: {df.shape}")
    df.to_csv(file_name, index=False)


if __name__ == "__main__":
    main()
