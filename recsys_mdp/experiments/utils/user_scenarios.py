import numpy as np

from recsys_mdp.simulator.relevance import similarity


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
