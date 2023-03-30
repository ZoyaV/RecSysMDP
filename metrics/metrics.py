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
    return k / init_pred_count

def coverage(pred):
    return len(set(pred))
