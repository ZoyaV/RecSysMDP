from __future__ import annotations

import numpy as np


def similarity(users: np.ndarray, items: np.ndarray, metric: str) -> np.ndarray | float:
    if metric == 'l2':
        return l2_similarity(users, items)
    elif metric == 'l1':
        return l1_similarity(users, items)
    elif metric == 'cosine':
        return cosine_similarity(users, items)
    raise ValueError(f'Unknown similarity metric: {metric}')


def resolve_similarity_metric(metric: str):
    if metric == 'l2':
        return l2_similarity
    if metric == 'l1':
        return l1_similarity
    if metric == 'cosine':
        return cosine_similarity
    raise ValueError(f'Unknown similarity metric: {metric}')


def l2_similarity(users: np.ndarray, items: np.ndarray) -> np.ndarray | float:
    d = users - items
    avg_sq_d = (d ** 2).mean(axis=-1)
    return 1.0 - np.sqrt(avg_sq_d)


def l1_similarity(users: np.ndarray, items: np.ndarray) -> np.ndarray | float:
    d = users - items
    return 1.0 - np.abs(d).mean(axis=-1)


def cosine_similarity(users: np.ndarray, items: np.ndarray) -> np.ndarray | float:
    dot_product = np.sum(users * items, axis=-1)
    users_norm = np.linalg.norm(users, axis=-1)
    items_norm = np.linalg.norm(items, axis=-1)
    return dot_product / (users_norm * items_norm)
