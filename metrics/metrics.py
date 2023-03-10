import numpy as np
import math
import wandb
import torch

def ndcg(k, pred, ground_truth) -> float:
    pred_len = min(k, len(pred))
    ground_truth_len = min(k, len(ground_truth))
    denom = [1 / math.log2(i + 2) for i in range(k)]
    dcg = sum(denom[i] for i in range(pred_len) if pred[i] in ground_truth)
    idcg = sum(denom[:ground_truth_len])
    return dcg / idcg



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
    pass

def log_items_distribution(total_prediction):
    predicted_element = np.argmax(total_prediction, axis=1)
    data = predicted_element.ravel().reshape(-1, 1)
    table = wandb.Table(data=data, columns=["items"])
    wandb.log({'Items Histogram': wandb.plot.histogram(table, "items", title=None)})
    pass

epoch = 0
def base_ndcg(oservations, true_reward, tresh, top_k=10, discrete = True):
    global epoch
    if discrete:
        oservations_cuda = torch.from_numpy(oservations).cpu()
        #print(oservations)
    def metrics(model=None, episodes=None):
        global epoch
        epoch += 1
        if discrete:
            with torch.no_grad():
                total_prediction = (model._impl._q_func(oservations_cuda)).cpu().detach().numpy()
                total_prediction = (total_prediction - total_prediction.min())/(total_prediction.max() - total_prediction.min())
        else:
            total_prediction = model.predict(oservations)
        users = np.unique(oservations[:, -1])
        ndcg_tot = []
        relevances_for_users = []
        for user in users:
            mask = oservations[:, -1] == user
            if not discrete:
                items = oservations[:,-2][mask]
                relevance = total_prediction[mask].ravel()
                item_relevance = list(zip(items, relevance))
            else:
                item_relevance = list(zip(range(0, len(mask)), total_prediction[mask].ravel()))

            item_relevance = sorted(item_relevance, key=lambda item: item[-1])[::-1]
            top_items = [item for item, r in item_relevance[:top_k]]
            top_items = sorted(top_items)

            true_top_items = true_reward[user]
            true_top_items = sorted(true_top_items)

            ndcg_user = ndcg(top_k, top_items, true_top_items)
            ndcg_tot.append(ndcg_user)

            relevances_for_users.append(item_relevance)

        tresholds = tresh

        log_user_items_distribution(relevances_for_users, tresholds)
        log_rating_distribution(total_prediction)
        log_items_distribution(total_prediction)

        result_median = np.median(ndcg_tot)
        result_mean = np.mean(ndcg_tot)
        result_std = np.std(ndcg_tot)
        wandb.log({"true_NDCG_median": result_median})
        wandb.log({"true_NDCG_mean": result_mean})
        wandb.log({"true_NDCG_std": result_std})
        return result_mean
    return metrics
