import numpy as np


def ndcg(k, predictions, ground_truth) -> float:
    # for each rang i => 1 / log2(i + 2)
    denominators = 1.0 / np.log2(np.arange(k) + 2)

    # Discounted Cumulative Gain (DCG): sum of the denominators of items in ground truth
    predictions_denominators = denominators[:min(k, len(predictions))]
    relevance_mask = np.isin(predictions, ground_truth)
    dcg = np.sum(predictions_denominators[relevance_mask])

    # normalize dcg by the maximum possible value — Ideal DCG
    ideal_dcg = np.sum(denominators[:min(k, len(ground_truth))])

    return dcg / ideal_dcg


def mean_average_precision(k, predictions, ground_truth) -> float:
    cumulative_num_elements = np.arange(len(predictions)) + 1

    # count cumulatively the number of relevant items in the predictions
    relevance_mask = np.isin(predictions, ground_truth)
    cumulative_num_relevant_elements = np.cumsum(relevance_mask)

    all_precisions = cumulative_num_relevant_elements / cumulative_num_elements
    relevant_precisions = all_precisions[relevance_mask]

    # Mean Average Precision (MAP): sum of relevant precisions divided by the number
    # of items in ground truth set
    return relevant_precisions.sum() / len(ground_truth)
