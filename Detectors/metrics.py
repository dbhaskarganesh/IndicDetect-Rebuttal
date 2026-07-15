import numpy as np
from sklearn.metrics import roc_curve, auc


def get_roc_metrics(real_preds, sample_preds):
    if not real_preds or not sample_preds:
        raise ValueError(
            f"Both classes must have predictions. "
            f"Got {len(real_preds)} real and {len(sample_preds)} sample predictions."
        )

    real_labels = [0] * len(real_preds) + [1] * len(sample_preds)
    predicted_probs = real_preds + sample_preds

    if len(set(predicted_probs)) == 1:
        return 0.5, float(predicted_probs[0]), [[0, 0], [0, 0]], 0.0, 0.0, 0.0, 0.0

    fpr, tpr, thresholds = roc_curve(real_labels, predicted_probs)
    roc_auc = auc(fpr, tpr)

    finite_mask = np.isfinite(thresholds)
    if not np.any(finite_mask):
        return float(roc_auc), 0.0, [[0, 0], [0, 0]], 0.0, 0.0, 0.0, 0.0

    fpr_f = fpr[finite_mask]
    tpr_f = tpr[finite_mask]
    thresholds_f = thresholds[finite_mask]

    optimal_idx = np.argmax(tpr_f - fpr_f)
    optimal_threshold = thresholds_f[optimal_idx]

    return float(roc_auc), float(optimal_threshold), [], 0.0, 0.0, 0.0, 0.0


def get_metrics(real_preds, sample_preds, threshold):
    if not real_preds or not sample_preds:
        raise ValueError(
            f"Both classes must have predictions. "
            f"Got {len(real_preds)} real and {len(sample_preds)} sample predictions."
        )

    all_scores = real_preds + sample_preds
    y_true = [0] * len(real_preds) + [1] * len(sample_preds)
    y_pred = [1 if s >= threshold else 0 for s in all_scores]

    tn = fp = fn = tp = 0
    for t, p in zip(y_true, y_pred):
        if t == 0 and p == 0:
            tn += 1
        elif t == 0 and p == 1:
            fp += 1
        elif t == 1 and p == 0:
            fn += 1
        elif t == 1 and p == 1:
            tp += 1

   
    prec_1 = tp / (tp + fp) if (tp + fp) else 0.0
    rec_1 = tp / (tp + fn) if (tp + fn) else 0.0
    f1_1 = 2 * prec_1 * rec_1 / (prec_1 + rec_1) if (prec_1 + rec_1) else 0.0

    
    prec_0 = tn / (tn + fn) if (tn + fn) else 0.0
    rec_0 = tn / (tn + fp) if (tn + fp) else 0.0
    f1_0 = 2 * prec_0 * rec_0 / (prec_0 + rec_0) if (prec_0 + rec_0) else 0.0

    
    macro_f1 = (f1_0 + f1_1) / 2
    
     
    precision = prec_1 
    recall = rec_1
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0

    return (
        float(threshold),
        [[int(tn), int(fp)], [int(fn), int(tp)]],
        float(precision),
        float(recall),
        float(macro_f1),  # Replaced binary F1 with Macro-F1
        float(accuracy),
    )