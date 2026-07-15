#!/usr/bin/env python3
import logging
import random
import torch
import tqdm
import argparse
import json
import contextlib
import os
import numpy as np
from binoculars_detector import Binoculars
from sklearn.metrics import roc_curve, auc, roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

VALID_LABELS = {"human", "llm", "ai", "machine"}


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_label(label):
    lbl = str(label).strip().lower()
    if lbl in ("ai", "machine"):
        return "llm"
    if lbl in ("human", "llm"):
        return lbl
    return None


def load_and_rename_sentence(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        if "sentence" in item and "text" not in item:
            item["text"] = item.pop("sentence")
        if "text" not in item:
            raise ValueError(f"Missing 'text' or 'sentence' key in entry: {item}")
        if item.get("label") is None:
            raise ValueError(f"Missing 'label' key in entry: {item}")
        lbl = normalize_label(item["label"])
        if lbl is None:
            raise ValueError(f"Invalid label '{item['label']}'. Expected one of {VALID_LABELS}")
        item["label"] = lbl
    return data


def check_data_overlap(threshold_data, test_data, test_name):
    train_texts = {item["text"] for item in threshold_data}
    test_texts = {item["text"] for item in test_data}
    overlap = train_texts & test_texts
    if overlap:
        logging.warning(
            f"DATA LEAKAGE DETECTED: {len(overlap)} overlapping texts between "
            f"threshold data and test file '{test_name}'. Removing from test set."
        )
        test_data = [item for item in test_data if item["text"] not in overlap]
        logging.info(f"Test set size after removing overlaps: {len(test_data)}")
    return test_data


@contextlib.contextmanager
def bf16_autocast(enable=True):
    if enable and torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            yield
    else:
        yield


def compute_roc_and_threshold(human_scores, llm_scores):
    if not human_scores or not llm_scores:
        raise ValueError(
            f"Both classes required. Got {len(human_scores)} human, {len(llm_scores)} llm."
        )

    all_scores = np.concatenate([np.asarray(human_scores), np.asarray(llm_scores)])
    y_true = np.array([0] * len(human_scores) + [1] * len(llm_scores))

    if np.allclose(all_scores, all_scores[0]):
        logging.warning("All scores identical. ROC undefined.")
        return float("nan"), float(all_scores[0])

    fpr, tpr, thresholds = roc_curve(y_true, all_scores)
    roc_auc = float(auc(fpr, tpr))

    finite_mask = np.isfinite(thresholds)
    if not np.any(finite_mask):
        return roc_auc, float(np.median(all_scores))

    best_idx = int(np.nanargmax(tpr[finite_mask] - fpr[finite_mask]))
    optimal_threshold = float(thresholds[finite_mask][best_idx])

    return roc_auc, optimal_threshold


def compute_tpr_at_fpr(human_scores, llm_scores, target_fpr=0.01):
    if not human_scores or not llm_scores:
        return 0.0
    all_scores = np.concatenate([np.asarray(human_scores), np.asarray(llm_scores)])
    y_true = np.array([0] * len(human_scores) + [1] * len(llm_scores))
    try:
        fpr, tpr, _ = roc_curve(y_true, all_scores)
        if np.max(fpr) < target_fpr:
            return 0.0
        return float(np.interp(target_fpr, fpr, tpr))
    except Exception:
        return 0.0


def compute_confusion(y_true, y_pred):
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
    return tn, fp, fn, tp


def experiment(args):
    set_all_seeds(args.seed)

    bino = Binoculars(mode="accuracy", max_token_observed=args.tokens_seen)

    threshold_data = None
    train_optimal_threshold = None
    train_roc_auc = None

    if args.threshold_file:
        logging.info(f"Loading threshold data from {args.threshold_file}")
        threshold_data = load_and_rename_sentence(args.threshold_file)

        train_predictions = {"human": [], "llm": []}
        skipped = 0
        for item in tqdm.tqdm(threshold_data, desc="Processing threshold data"):
            with bf16_autocast(args.use_bf16):
                score = -bino.compute_score(item["text"])

            if not np.isfinite(score):
                skipped += 1
                continue
            train_predictions[item["label"]].append(float(score))

        logging.info(
            f"Threshold data: human={len(train_predictions['human'])}, "
            f"llm={len(train_predictions['llm'])}, skipped={skipped}"
        )

        if not train_predictions["human"] or not train_predictions["llm"]:
            raise RuntimeError(
                "Threshold data has no valid scores for one or both classes. Cannot proceed."
            )

        train_roc_auc, train_optimal_threshold = compute_roc_and_threshold(
            train_predictions["human"], train_predictions["llm"]
        )
        logging.info(
            f"Train threshold: {train_optimal_threshold:.6f} (AUC: {train_roc_auc:.4f})"
        )
    else:
        raise ValueError(
            "No --threshold_file provided. A separate threshold file is required "
            "to avoid computing threshold on test data (data leakage)."
        )

    filenames = [f.strip() for f in args.test_data_path.split(",") if f.strip()]

    for filename in filenames:
        logging.info(f"Testing on {filename}")
        test_data = load_and_rename_sentence(filename)

        if threshold_data is not None:
            test_data = check_data_overlap(threshold_data, test_data, filename)

        if not test_data:
            logging.warning(f"No data remaining for {filename} after overlap removal. Skipping.")
            continue

        predictions = {"human": [], "llm": []}
        all_scores = []
        y_true = []
        skipped = 0

        for item in tqdm.tqdm(test_data, desc=os.path.basename(filename)):
            with bf16_autocast(args.use_bf16):
                score = -bino.compute_score(item["text"])

            item["bino_score"] = None if not np.isfinite(score) else float(score)

            if not np.isfinite(score):
                skipped += 1
                continue

            predictions[item["label"]].append(float(score))
            all_scores.append(float(score))
            y_true.append(0 if item["label"] == "human" else 1)

        logging.info(
            f"Test results: total={len(all_scores)}, skipped={skipped}, "
            f"human={len(predictions['human'])}, llm={len(predictions['llm'])}"
        )

        if not all_scores:
            logging.warning(f"No valid scores for {filename}. Skipping.")
            continue

        roc_auc = float("nan")
        if predictions["human"] and predictions["llm"]:
            try:
                roc_auc = float(roc_auc_score(y_true, all_scores))
            except Exception as e:
                logging.warning(f"Could not compute test ROC AUC: {e}")

        tpr_at_fpr_0_01 = compute_tpr_at_fpr(
            predictions["human"], predictions["llm"], target_fpr=0.01
        )

        optimal_threshold = train_optimal_threshold
        y_pred = [1 if s >= optimal_threshold else 0 for s in all_scores]
        tn, fp, fn, tp = compute_confusion(y_true, y_pred)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0

        result = {
            "roc_auc": float(roc_auc) if np.isfinite(roc_auc) else None,
            "optimal_threshold": float(optimal_threshold),
            "conf_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "accuracy": float(accuracy),
            "tpr_at_fpr_0_01": float(tpr_at_fpr_0_01),
            "num_examples": len(all_scores),
        }

        logging.info(f"{result}")

        base_prefix = os.path.splitext(filename)[0]
        data_file = base_prefix + "_bino_data.json"
        result_file = base_prefix + "_bino_result.json"

        with open(data_file, "w", encoding="utf-8") as f:
            json.dump(test_data, f, indent=4, ensure_ascii=False)
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4)

        logging.info(f"Saved -> {data_file}")
        logging.info(f"Saved -> {result_file}")

    logging.info("All test files processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_path", type=str, required=True)
    parser.add_argument("--tokens_seen", type=int, default=512)
    parser.add_argument("--DEVICE", default="cuda", type=str)
    parser.add_argument("--seed", default=2023, type=int)
    parser.add_argument("--threshold_file", type=str, required=True)
    parser.add_argument("--use_bf16", action="store_true")
    args = parser.parse_args()

    if args.use_bf16:
        if not torch.cuda.is_available():
            logging.warning("CUDA not available; BF16 will have no effect.")
        elif not getattr(torch.cuda, "is_bf16_supported", lambda: False)():
            logging.warning("BF16 not supported on this device.")
            args.use_bf16 = False

    experiment(args)