def _remove_padding(predictions_padded, targets_padded):
    mask = (targets_padded >= 0).long()
    n = len([out for out in mask.squeeze() if out.all() >= 1])
    outputs = predictions_padded[:n, :]


    return outputs

def _remove_one_hot(targets_padded):
    mask = (targets_padded >= 0).long()
    n = len([out for out in mask.squeeze() if out.all() >= 1])
    targets_padded = targets_padded.squeeze()
    targets = targets_padded[:n, :]
   # print(target)
    _, targets = targets.max(dim=1)  # remove one hot encoding
    return targets