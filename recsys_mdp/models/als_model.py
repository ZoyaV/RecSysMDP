import numpy as np
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares

class ALSRecommender:
    def __init__(self, factors=50, regularization=0.01, iterations=50):
        self.model = AlternatingLeastSquares(factors=factors, regularization=regularization, iterations=iterations)
        self.user_mapping = {}
        self.item_mapping = {}

    def fit(self, data):
        # Создание словарей для маппинга индексов пользователей и элементов в их оригинальные ID
        self.user_mapping = {i: user_id for i, user_id in enumerate(data['user_idx'].unique())}
        self.item_mapping = {i: item_id for i, item_id in enumerate(data['item_idx'].unique())}

        # Создание обратных словарей
        self.user_mapping_inv = {v: k for k, v in self.user_mapping.items()}
        self.item_mapping_inv = {v: k for k, v in self.item_mapping.items()}

        # Создание матрицы взаимодействий
        interactions = coo_matrix(
            (data['relevance_int'],
             (data['user_idx'].map(self.user_mapping_inv), data['item_idx'].map(self.item_mapping_inv)))
        )

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
        return [np.random.choice(top_items)]
