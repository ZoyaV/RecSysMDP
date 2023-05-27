import numpy as np
import math
def ndcg(k, pred, ground_truth) -> float:
    pred_len = min(k, len(pred))
    ground_truth_len = min(k, len(ground_truth))
    denom = [1 / math.log2(i + 2) for i in range(k)]
    dcg = sum(denom[i] for i in range(pred_len) if pred[i] in ground_truth)
    idcg = sum(denom[:ground_truth_len])
    return dcg / idcg

def hit_rate( pred, ground_truth):
    init_pred_count = len(pred)
    pred = list(set(pred))
    k = 0
    for item in pred:
        if int(item) in ground_truth:
            k+=1
    if init_pred_count == 0:
        return 0
    return k / init_pred_count

def coverage(pred):
    return len(set(pred))


# TODO: check if a) ndcg2 can replace ndcg; b) MAP is correct


def ndcg2(k, predictions, ground_truth) -> float:
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
