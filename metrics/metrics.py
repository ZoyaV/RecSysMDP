import numpy as np
import math
import wandb


def ndcg(k, pred, ground_truth) -> float:
    pred_len = min(k, len(pred))
    ground_truth_len = min(k, len(ground_truth))
    denom = [1 / math.log2(i + 2) for i in range(k)]
    dcg = sum(denom[i] for i in range(pred_len) if pred[i] in ground_truth)
    idcg = sum(denom[:ground_truth_len])
    return dcg / idcg


epoch = 0
def true_ndcg(users,observations, top_k=10):
    global epoch
    def metrics(model=None, episodes=None):
        global epoch
        metrics_ndcg = []
        user_positiove_count = []
        for episode in episodes:
            current_user = episode.observations[0][-1]
            mask = np.where(users == current_user)
            user_observation = observations[mask]
            items = [value[-2] for value in user_observation]
            predicted_rating = model.predict(user_observation)
            item_ratings = list(zip(items, predicted_rating))
            predicted_top_items = sorted(item_ratings, key=lambda item_rat: item_rat[1])[::-1]
            predicted_top_items = [item for item, rat in predicted_top_items if rat > 0.5]
            #TODO: надо проверить что при генерации оценка идет именно к последнему элементу их фреймстака

            # find original top items
            true_user_items = [value[-2] for value in episode.observations]
            true_item_ratings = episode.rewards.tolist()
            true_item_ratings = list(zip(true_user_items, true_item_ratings))
            original_top_items = sorted(true_item_ratings, key=lambda item_rat: item_rat[1])[::-1]
            original_top_items = [item for item, rat in original_top_items if rat > 0.5]

            only_positive_prediction = [item for item, rat in item_ratings if rat > 0.5]
            user_positiove_count.append(len(only_positive_prediction))

           # print("-----------------------")
           # print(predicted_top_items[:5])
           # print(original_top_items[:5])
           # print("-----------------------")
#
            if len(predicted_top_items) > 0 and len(original_top_items) > 0:
                ndcg_user = ndcg(top_k, predicted_top_items, original_top_items)
            else:
                ndcg_user = 0
            metrics_ndcg.append(ndcg_user)
        # print(metrics_ndcg)
        data = []
        for i in range(len(user_positiove_count)):
            data.append([f"user_{i}", user_positiove_count[i]])
            # wandb.log({f"user_{i}":user_positiove_count[i]}, name = f"epoch_{epoch}")
        table = wandb.Table(data=data, columns=["label", "value"])
        if epoch % 10 == 0:
            wandb.log({f"user_likes_distibution": wandb.plot.bar(table, "label", "value",
                                                                 title=f"User Likes Distibution")})

        result_median = np.median(metrics_ndcg)
        result_mean = np.mean(metrics_ndcg)
        result_std = np.std(metrics_ndcg)
        wandb.log({"NDCG_median": result_median})
        wandb.log({"NDCG_mean": result_mean})
        wandb.log({"NDCG_std": result_std})
        epoch += 1

        return np.mean(metrics_ndcg)
    return metrics
