import numpy as np

def random_embeddings(df, emb_size):
    mapping = dict()
    inv_mapping = dict()
    users = list(set(df))
    for user in range(1, max(users) + 1):
        new_vector = np.random.normal(0, 1, size=emb_size)
        mapping[user] = new_vector
        inv_mapping[tuple(new_vector)] = user
    return mapping, inv_mapping