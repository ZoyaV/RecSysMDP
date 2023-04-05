import numpy as np
import implicit
from scipy.sparse import coo_matrix

def als_embeddings(users, items, values, emb_size=8):
    users = users
    items = items
    values = values
    sparse_matrix = coo_matrix((values, (users, items)))
    model = implicit.als.AlternatingLeastSquares(factors=emb_size)
    model.fit(sparse_matrix)
    user_embeddings = model.user_factors
    item_embeddings = model.item_factors

    return user_embeddings, item_embeddings
