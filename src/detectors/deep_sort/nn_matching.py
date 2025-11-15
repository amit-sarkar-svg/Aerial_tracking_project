# src/trackers/deep_sort/nn_matching.py
import numpy as np
try:
    from scipy.optimize import linear_sum_assignment
    _have_scipy = True
except Exception:
    _have_scipy = False

def iou(bb1, bb2):
    x1 = max(bb1[0], bb2[0])
    y1 = max(bb1[1], bb2[1])
    x2 = min(bb1[2], bb2[2])
    y2 = min(bb1[3], bb2[3])
    w = max(0., x2 - x1)
    h = max(0., y2 - y1)
    inter = w*h
    a = (bb1[2]-bb1[0])*(bb1[3]-bb1[1])
    b = (bb2[2]-bb2[0])*(bb2[3]-bb2[1])
    union = a + b - inter
    if union <= 0:
        return 0.0
    return inter/union

def iou_cost_matrix(tracks, dets):
    cost = np.zeros((len(tracks), len(dets)), dtype=np.float32)
    for i, t in enumerate(tracks):
        for j, d in enumerate(dets):
            cost[i,j] = 1.0 - iou(t, d)
    return cost

def linear_assignment(cost_matrix, thresh=0.7):
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))
    if _have_scipy:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matches = []
        unmatched_a = list(range(cost_matrix.shape[0]))
        unmatched_b = list(range(cost_matrix.shape[1]))
        for r,c in zip(row_ind, col_ind):
            if cost_matrix[r,c] <= thresh:
                matches.append((r,c))
                unmatched_a.remove(r)
                unmatched_b.remove(c)
        return matches, unmatched_a, unmatched_b
    else:
        # greedy matching
        matches = []
        unmatched_a = list(range(cost_matrix.shape[0]))
        unmatched_b = list(range(cost_matrix.shape[1]))
        flat = []
        for i in range(cost_matrix.shape[0]):
            for j in range(cost_matrix.shape[1]):
                flat.append((cost_matrix[i,j], i, j))
        flat.sort()
        used_a=set(); used_b=set()
        for c,i,j in flat:
            if i in used_a or j in used_b:
                continue
            if c <= thresh:
                matches.append((i,j))
                used_a.add(i); used_b.add(j)
        unmatched_a = [i for i in range(cost_matrix.shape[0]) if i not in used_a]
        unmatched_b = [j for j in range(cost_matrix.shape[1]) if j not in used_b]
        return matches, unmatched_a, unmatched_b
