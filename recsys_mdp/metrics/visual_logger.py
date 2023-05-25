import numpy as np
from sklearn.manifold import TSNE

from recsys_mdp.utils.lazy_imports import lazy_import

wandb = lazy_import('wandb')


def log_user_items_distribution(relevances_for_users, tresholds):
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

def log_items_distribution(total_prediction, discrete = True):
    if not discrete:
        predicted_element = np.argmax(total_prediction, axis=1)
    else:
        predicted_element = total_prediction
    data = predicted_element.ravel().reshape(-1, 1)
    wandb.log({'Items Histogram': wandb.Histogram(data)})
    return

def tsne_scatter(vals, name):
    #print(vals.shape)
    #exit()
    population_tsne = TSNE(n_components=2, random_state=42).fit_transform(vals)
    table = wandb.Table(data=population_tsne, columns=["t-SNE Dimension 1", "t-SNE Dimension 2"])
    wandb.log(
        {f"TSNE_{name}": wandb.plot.scatter(table, "t-SNE Dimension 1", "t-SNE Dimension 2")})


def items_distribution(total_prediction, vals, names, discrete = True):
    if not discrete:
        predicted_element = np.argmax(total_prediction, axis=1)
    else:
        predicted_element = total_prediction
    data = predicted_element.ravel().reshape(-1, 1)
    wandb.log({'Items Histogram': wandb.Histogram(data)})
    return

def tsne(total_prediction, vals, names, discrete = True):

    for i in range(len(vals)):
        population_tsne = TSNE(n_components=2, random_state=42).fit_transform(vals[i])
        table = wandb.Table(data=population_tsne, columns=["t-SNE Dimension 1", "t-SNE Dimension 2"])
        wandb.log(
            {f"TSNE_{names[i]}": wandb.plot.scatter(table, "t-SNE Dimension 1", "t-SNE Dimension 2")})
