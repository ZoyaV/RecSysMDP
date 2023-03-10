import numpy as np

def random_embeddings(df, emb_size):
    mapping = dict()
    inv_mapping = dict()
    users = list(set(df))
    for user in range(1, max(users) + 1):
       # print("aboba")
        if emb_size==1:
            new_vector = [user]
            mapping[user] = np.asarray([user])
        else:
            new_vector = np.random.uniform(0, 1, size=emb_size)
            mapping[user] = new_vector
        inv_mapping[tuple(new_vector)] = user
    #print(mapping)
    #exit()
    return mapping, inv_mapping

# Function for calculating 2 digit