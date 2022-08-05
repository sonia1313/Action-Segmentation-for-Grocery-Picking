import numpy as np
import torch


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


def _get_average_metrics(outputs):
    f1_10_outs = []
    f1_25_outs = []
    f1_50_outs = []
    edit_outs = []
    accuracy_outs = []
    for i, out in enumerate(outputs):
        a, e, f = out
        f1_10_outs.append(f[0])
        f1_25_outs.append(f[1])
        f1_50_outs.append(f[2])

        edit_outs.append(e)
        accuracy_outs.append(a)

    f1_10_mean = np.stack([x for x in f1_10_outs]).mean(0)
    f1_25_mean = np.stack([x for x in f1_25_outs]).mean(0)
    f1_50_mean = np.stack([x for x in f1_50_outs]).mean(0)
    edit_mean = np.stack([x for x in edit_outs]).mean(0)
    accuracy_mean = torch.mean(torch.stack([x for x in accuracy_outs]))

    return f1_10_mean, f1_25_mean, f1_50_mean, edit_mean, accuracy_mean
