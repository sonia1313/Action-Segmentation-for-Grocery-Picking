import numpy as np
from utils.metrics_utils import _segment_intervals, _segment_labels

def f1_score(logits, labels):
    logits = logits.detach().numpy()
    preds = np.argmax(logits,axis = 1)

    labels = labels.detach().numpy()
    #
    # print(preds)
    # print(labels)


    # acc = get_accuracy(preds,labels)
    # print(f"accuracy: {acc}")

    overlap_f1_score, overlap_threshold = get_overlap_f1(preds, labels)

    print(f"f1 score at {overlap_threshold*100} overlap: {overlap_f1_score}")




def get_overlap_f1(prediction,ground_truth, n_classes = 6, overlap = 0.1 ):

    true_intervals = np.array(_segment_intervals(ground_truth))
    true_labels = _segment_labels(ground_truth)

    pred_intervals = np.array(_segment_intervals(prediction))
    pred_labels = _segment_labels(prediction)


    n_true = true_labels.shape[0]
    n_pred = pred_labels.shape[0]

    TP = np.zeros(n_classes,float)
    FP = np.zeros(n_classes,float)


    true_used = np.zeros(n_true,float)

    # print(pred_intervals)
    # print(true_intervals)
    for j in range(n_pred):
        intersection = np.minimum(pred_intervals[j,1], true_intervals[:,1]) - np.maximum(pred_intervals[j,0], true_intervals[:,0])

        union = np.maximum(pred_intervals[j,1], true_intervals[:,1]) - np.minimum(pred_intervals[j,0], true_intervals[:,0])

        IoU = (intersection / union)*(pred_labels[j]==true_labels)

        idx = IoU.argmax()

        #if IoU is higher than overlap thershold and
        # true segment is not already used, then assign as a true positive otherwise false positive

        if IoU[idx] >= overlap and not true_used[idx]:
            TP[pred_labels[j]] += 1
            true_used[idx] = 1
        else:
            FP[pred_labels[j]] += 1

    TP = TP.sum()
    FP = FP.sum()

    #FN are any unused true segement i.e. miss

    FN = n_true - true_used.sum()

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    if (precision + recall == 0.0):
        F1 = 0.0
    else:

        F1 = 2 * (precision * recall) / (precision + recall)

    # If the prec+recall=0, it is a NaN. Set these to 0.
   # F1 = np.nan_to_num(F1)

    return F1*100, overlap

# print(_segment_intervals(test_case_target))
# print(_segment_intervals(test_case_preds))
# print(get_accuracy(test_case_preds,test_case_target))
# print(get_overlap_f1(test_case_preds,test_case_target))

# get_scores()