
def linear_annealing(n, total, p_initial, p_final):
    """Linear annealing between p_initial and p_final
    over total steps - computes value at step n"""
    if n >= total:
        return p_final
    else:
        return p_initial - (n * (p_initial - p_final)) / (total)

def onehot_encode(list_of_idxes, num_classes):
    result = np.zeros((len(list_of_idxes), num_classes), dtype=np.float32)
    for i, class_idx in enumerate(list_of_idxes):
        result[i][class_idx] = 1.0
    return result

def none_mask(list_of_items):
    result = np.zeros((len(list_of_items),), dtype=np.float32)
    for i, item in enumerate(list_of_items):
        if item is not None:
            result[i] = 1.
    return result
