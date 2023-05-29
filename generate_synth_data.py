import numpy as np
from numpy.random import Generator

from recsys_mdp.models.als_model import ALSRecommender
from recsys_mdp.simulator.relevance import similarity
from recsys_mdp.simulator.user_state import (
    USER_RESET_MODE_DISCONTINUE
)
from recsys_mdp.utils.lazy_imports import lazy_import

wandb = lazy_import('wandb')


def generate_episode(
        env, model, framestack_size=10, user_id=None, log_sat=False, logger=None,
        get_best_for_start=True, use_best=False,
        rng: Generator = None
):
    orig_user_id = user_id
    user_id = env.reset(user_id=user_id)
    trajectory = []

    # Get random items from best for framestack
    # TODO: How it will affect to episode lenghts?
    # TODO: Make framestack making as function

    if get_best_for_start:
        top_framestack = []

        for i in range(framestack_size):
            items_top = env.state.ranked_items(with_satiation=True, discrete=True)
            item_id = rng.choice(items_top[:10])
            top_framestack.append(item_id)
            _, _ = env.step(item_id)
        # add scores as all is best
        # [N last item_ids] + [[5] * N] +  [user_id]
        fake_obs = top_framestack + [5] * framestack_size + [user_id]

    else:
        # [N last item_ids] + [user_id]
        fake_obs = rng.integers(0, env.n_items, framestack_size).tolist() + [user_id]
    obs = np.asarray(fake_obs)
    item_id = 0
    # episode generation
    while True:
        if not use_best:
            try:
                item_id = model.predict(obs.reshape(1, -1))[0]
            except:
                item_id = model.predict(obs[:framestack_size].reshape(1, -1))[0]

        timestamp = env.timestamp
        # log satiation histogram
        if env.timestep % 4 == 0 and log_sat:
            log_satiation(logger, env.state.satiation, orig_user_id)

        relevance, terminated = env.step(item_id)

        continuous_relevance, discrete_relevance = relevance

        # TODO: add addtional function for framestack support

        obs[:framestack_size - 1] = obs[1:framestack_size]
        obs[framestack_size - 1] = item_id
        #  print(obs[:framestack_size])
        obs[framestack_size:framestack_size * 2 - 1] = obs[framestack_size + 1:framestack_size * 2]
        obs[framestack_size * 2 - 1] = discrete_relevance

        # print(obs)
        #   print(obs[framestack_size:framestack_size * 2])
        items_top = env.state.ranked_items(with_satiation=True, discrete=True)

        if use_best:
            item_id = rng.choice(items_top[:10])

        trajectory.append(
            (
                timestamp,
                user_id, item_id,
                continuous_relevance, discrete_relevance,
                terminated,
                items_top[:10]
            )
        )
        if terminated:
            break
    env.reset(user_id, USER_RESET_MODE_DISCONTINUE)
    if log_sat:
        log_satiation(logger, env.state.satiation, orig_user_id)

    return trajectory


def log_satiation(logger, satiation, user_id):
    if logger is None:
        return
    hist = (satiation, np.arange(len(satiation) + 1))
    histogram = wandb.Histogram(np_histogram=hist)
    logger.log({f'user {user_id} satiation': histogram})


def learn_als_model(data):
    model = ALSRecommender()
    model.fit(data)
    return model


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
        relevance, discr_relevance = user.relevance(top1_item, with_satiation=False)
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

    user.satiation_speed[top1] = 0.01
    user.satiation_speed[top2] = 0.04
