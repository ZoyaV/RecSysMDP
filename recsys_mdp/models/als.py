import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares

from recsys_mdp.mdp.base import USER_ID_COL, ITEM_ID_COL, RATING_COL


# TODO: We can merge both together


class AlsEmbeddings:
    model: AlternatingLeastSquares

    def __init__(self, n_dims: int, n_users: int, n_items: int, **model_config):
        self.model = AlternatingLeastSquares(factors=n_dims, **model_config)
        self.data_shape = (n_users + 1, n_items + 1)

    def fit_embeddings_on_dataframe(self, log_df: pd.DataFrame):
        users = log_df[USER_ID_COL].values
        items = log_df[ITEM_ID_COL].values
        rating = log_df[RATING_COL].values

        users, items = self.fit_embeddings(users, items, rating)
        return torch.from_numpy(users), torch.from_numpy(items)

    def fit_embeddings(self, users, items, ratings):
        sparse_matrix = coo_matrix((ratings, (users, items)), shape=self.data_shape)
        self.model.fit(sparse_matrix)
        user_embeddings = self.model.user_factors
        item_embeddings = self.model.item_factors
        return user_embeddings, item_embeddings


class ALSRecommender:
    def __init__(self, seed: int, factors=50, regularization=0.01, iterations=50):
        self.model = AlternatingLeastSquares(
            factors=factors, regularization=regularization, iterations=iterations
        )
        self.user_mapping = {}
        self.item_mapping = {}
        self.rng = np.random.default_rng(seed)

    def fit(self, data):
        # Создание словарей для маппинга индексов пользователей и элементов в их оригинальные ID
        self.user_mapping = {i: user_id for i, user_id in enumerate(data[USER_ID_COL].unique())}
        self.item_mapping = {i: item_id for i, item_id in enumerate(data[ITEM_ID_COL].unique())}

        # Создание обратных словарей
        self.user_mapping_inv = {v: k for k, v in self.user_mapping.items()}
        self.item_mapping_inv = {v: k for k, v in self.item_mapping.items()}

        assert False, f'From Petr: check if I used a correct rating column below and remove assert'
        # noinspection PyUnreachableCode
        # Создание матрицы взаимодействий
        interactions = coo_matrix((
            data[RATING_COL],
            (
                data[USER_ID_COL].map(self.user_mapping_inv),
                data[ITEM_ID_COL].map(self.item_mapping_inv)
            )
        ))

        # Обучение модели
        self.model.fit(interactions)

    def predict(self, interactions):
        #  print(interactions.shape)
        interactions = interactions[0]
        # Последний элемент взаимодействий - это ID пользователя
        user_id = interactions[-1]
        interactions = interactions[:-1]

        # Получение маппинга для заданного пользователя
        #  print(self.user_mapping_inv)
        #  print(user_id)
        user_idx = self.user_mapping_inv[user_id]

        # Предсказание рейтингов для всех элементов
        user_ratings = self.model.user_factors[user_idx] @ self.model.item_factors.T

        # Сортировка рейтингов в порядке убывания
        sorted_ratings = sorted(enumerate(user_ratings), key=lambda x: x[1], reverse=True)

        # Отбор N элементов с наибольшими рейтингами
        top_items = [self.item_mapping[i] for i, _ in sorted_ratings[:5]]

        # Возвращение случайного элемента из списка с наибольшими рейтингами
        return self.rng.choice(top_items, size=1)
