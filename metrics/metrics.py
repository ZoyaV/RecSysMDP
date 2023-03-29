import numpy as np
import math
import wandb
import torch
from sklearn.manifold import TSNE
np.random.seed(42)
def ndcg(k, pred, ground_truth) -> float:
    pred_len = min(k, len(pred))
    ground_truth_len = min(k, len(ground_truth))
    denom = [1 / math.log2(i + 2) for i in range(k)]
    dcg = sum(denom[i] for i in range(pred_len) if pred[i] in ground_truth)
    idcg = sum(denom[:ground_truth_len])
    return dcg / idcg

def hit_rate( pred, ground_truth):
    init_pred_count = len(pred)
    pred = list(set(pred))
    k = 0
    for item in pred:
        if int(item) in ground_truth:
            k+=1
    return k / init_pred_count
def log_user_items_distribution(relevances_for_users, tresholds):
    if epoch % 10 == 0:
        for t in tresholds:
            user_positiove_count = []
            for i in range(len(relevances_for_users)):
                only_positive_prediction = [item for item, rat in relevances_for_users[i] if rat > t]
                user_positiove_count.append(len(only_positive_prediction))
            data = []
            for i in range(len(user_positiove_count)):
                data.append([f"user_{i}", user_positiove_count[i]])
            table = wandb.Table(data=data, columns=["label", "value"])
            wandb.log({f"user_likes_distibution {t}": wandb.plot.bar(table, "label", "value",
                                                                    title=f"User Likes Distibution {t}")})
    return

def log_rating_distribution(total_prediction):
    data = total_prediction.ravel().reshape(-1, 1)
    table = wandb.Table(data=data, columns=["scores"])
    wandb.log({'Scores Histogram': wandb.plot.histogram(table, "scores", title=None)})
    return

def log_items_distribution(total_prediction):
    predicted_element = np.argmax(total_prediction, axis=1)
    data = predicted_element.ravel().reshape(-1, 1)
    wandb.log({'Items Histogram': wandb.Histogram(data)})
    return

epoch = 0

def coverage(observations, discrete = True):
    global epoch
    if discrete:
        oservations_cuda = torch.from_numpy(observations).cpu()
    def metrics(model=None, episodes=None):
        global epoch
        epoch += 1
        if discrete:
            total_prediction = model.predict(observations)
        else:
            raise Exception("Work only with discrete algo!")
        data = total_prediction.ravel().reshape(-1, 1)
        coverage = len(set(total_prediction))

        wandb.log({'Items Histogram': wandb.Histogram(data)})
        wandb.log({"Covarage": coverage})
        return coverage
    return metrics

def compute_ndcg_for_user(oservations, user, mask, total_prediction,
                          users_interests, discrete, top_k=10):
    # Compute NDCG for a specific user
    if not discrete:
        items = oservations[:, -2][mask]
        relevance = total_prediction[mask].ravel()
        item_relevance = list(zip(items, relevance))
    else:
        item_relevance = list(zip(range(0, len(mask)), total_prediction[mask].ravel()))

    # Sort item relevance
    item_relevance = sorted(item_relevance, key=lambda item: item[-1])[::-1]

    # Get top K items
    top_items = [item for item, r in item_relevance[:top_k]]
    top_items = sorted(top_items)

    # Get true top K items
    true_top_items = users_interests[user]
    true_top_items = sorted(true_top_items)

    # Calculate NDCG for user
    ndcg_user = ndcg(top_k, top_items, true_top_items)
    return ndcg_user

def base_ndcg(observations, users_interests, thresh, top_k=10, discrete=True):
    # Convert observations to CUDA if necessary
    if discrete:
        observations_cuda = torch.from_numpy(observations).cpu()

    def metrics(model=None, episodes=None):
        # Calculate total predictions for items
        if discrete:
            with torch.no_grad():
                total_prediction = (model._impl._q_func(observations_cuda)).cpu().detach().numpy()
                total_prediction = (total_prediction - total_prediction.min()) / (total_prediction.max() - total_prediction.min())
        else:
            total_prediction = model.predict(observations)

        # Get unique users
        users = np.unique(observations[:, -1])
        ndcg_tot = []

        # Calculate NDCG for each user
        for user in users:
            mask = observations[:, -1] == user
            ndcg_user = compute_ndcg_for_user(observations, user, mask, total_prediction,
                                              users_interests, discrete, top_k)
            ndcg_tot.append(ndcg_user)

        # Calculate mean and median of NDCG scores
        result_median = np.median(ndcg_tot)
        result_mean = np.mean(ndcg_tot)

        # Log metrics
        wandb.log({"true_NDCG_median": result_median})
        wandb.log({"true_NDCG_mean": result_mean})

        return result_mean
    return metrics

def episode_hit_rate(top_k, users_interests):
    def metrics(model=None, episodes=None):
        interactive_hit_rate_target = []
        static_hit_rate_target = []
        interactive_hit_rate_full_target = []
        static_hit_rate_tot_full_target = []

        for episode in episodes:
            user = int(episode.observations[0][-1])
            not_interactive_items = model.predict(episode.observations)
            original_items = episode.actions
            obs = episode.observations[0]
            interactive_items = []
            for _ in range(top_k):
                new_item = model.predict([obs])[0]
                interactive_items.append(new_item)

                new_obs = obs.copy()
                new_obs[:-3] = new_obs[1:-2]
                new_obs[-2] = new_item
                obs = new_obs.copy()

            # How of sequentialy predicted user-items in realy-user episode
            interactive_hit_rate = hit_rate(interactive_items, original_items)
            interactive_hit_rate_target.append(interactive_hit_rate)

            # How of predicted user-items in realy-user episode
            static_hit_rate = hit_rate(not_interactive_items, original_items)
            static_hit_rate_target.append(static_hit_rate)

            # How of sequentialy predicted user-items in realy-user interests
            interactive_hit_rate_full = hit_rate(interactive_items, users_interests[user])
            interactive_hit_rate_full_target.append(interactive_hit_rate_full)

            # How of  predicted user-items in realy-user interests
            static_hit_rate_tot_full = hit_rate(not_interactive_items, users_interests[user])
            static_hit_rate_tot_full_target.append(static_hit_rate_tot_full)

        wandb.log({"InteractiveHitRate_episode":np.mean(interactive_hit_rate_target)})
        wandb.log({"StaticHitRate_episode": np.mean(static_hit_rate_target)})

        wandb.log({"InteractiveHitRate_full_items": np.mean(interactive_hit_rate_full_target)})
        wandb.log({"StaticHitRate_full_items": np.mean(static_hit_rate_tot_full_target)})
        return 0
    return metrics
def tsne_scatter(vals, name):
    population_tsne = TSNE(n_components=2, random_state=42).fit_transform(vals)
    table = wandb.Table(data=population_tsne, columns=["t-SNE Dimension 1", "t-SNE Dimension 2"])
    wandb.log(
        {f"TSNE_{name}": wandb.plot.scatter(table, "t-SNE Dimension 1", "t-SNE Dimension 2")})
def tsne_embeddings(users, items):
    global epoch
    def metrics(model=None, episodes=None):
        with torch.no_grad():
            users_emb = model._impl._q_func._q_funcs[0]._encoder.state_repr.user_embeddings(torch.from_numpy(users))
            items_emb = model._impl._q_func._q_funcs[0]._encoder.state_repr.item_embeddings(torch.from_numpy(items))
        tsne_scatter(users_emb, "users_emb")
        tsne_scatter(items_emb, "items_emb")
        return 0
    return metrics