import numpy as np
def get_accuracy(prediction,ground_truth):
    return np.mean(prediction == ground_truth) * 100


def _segment_labels(Yi):
    # print(Yi)
    # print(np.diff(Yi))


    idxs = [0] + (np.nonzero(np.diff(Yi))[0]+1).tolist() + [len(Yi)]

    Yi_split = np.array([Yi[idxs[i]] for i in range(len(idxs)-1)])

    return Yi_split


def _segment_intervals(Yi):

    idxs = [0] + (np.nonzero(np.diff(Yi))[0]+1).tolist() + [len(Yi)]
    intervals = [(idxs[i],idxs[i+1]) for i in range(len(idxs)-1)]

    return intervals

def _get_preds_and_labels(logits_tensor,y_tensor):

    logits = logits_tensor.cpu().detach().numpy()
    preds = np.argmax(logits,axis = 1)

    labels = y_tensor.cpu().detach().numpy()

    return preds,labels