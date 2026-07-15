 #!/usr/bin/env python3
import logging
import random
import numpy as np
import torch
import tqdm
import argparse
import json
import os
import contextlib
from metrics import get_roc_metrics
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_label(label):
    lbl = str(label).strip().lower()
    if lbl in ("human",):
        return "Human"
    if lbl in ("llm", "ai", "machine"):
        return "LLM"
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
            raise ValueError(f"Invalid label '{item['label']}'. Expected 'Human' or 'LLM'.")
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
    if enable and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            yield
    else:
        yield


def get_rank(text, args, tokenizer, model, log=False):
    if not isinstance(text, str) or not text.strip():
        return None

    tokenized = tokenizer(
        text, return_tensors="pt", return_token_type_ids=False
    ).to(args.DEVICE)

    if tokenized.input_ids.shape[1] < 2:
        return None

    with torch.no_grad():
        with bf16_autocast(args.use_bf16):
            logits = model(**tokenized).logits[:, :-1]

    labels = tokenized.input_ids[:, 1:]
    matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()

    if matches.shape[1] != 3:
        raise RuntimeError(f"Expected 3 dimensions in matches tensor, got {matches.shape}")

    ranks, timesteps = matches[:, -1], matches[:, -2]
    expected_timesteps = torch.arange(labels.shape[1], device=timesteps.device)

    if len(timesteps) != len(expected_timesteps):
        raise RuntimeError(
            f"Expected {len(expected_timesteps)} rank matches, got {len(timesteps)}."
        )

    if not (timesteps == expected_timesteps).all():
        raise RuntimeError("Timestep mismatch: expected one match per position.")

    ranks = ranks.float() + 1
    if log:
        ranks = torch.log(ranks)

    return ranks.float().mean().item()


def get_ll(text, args, tokenizer, model):
    if not isinstance(text, str) or not text.strip():
        return float("nan")

    with torch.no_grad():
        tokenized = tokenizer(
            text, return_tensors="pt", return_token_type_ids=False
        ).to(args.DEVICE)
        labels = tokenized["input_ids"]

        if labels.nelement() < 2:
            return float("nan")

        with bf16_autocast(args.use_bf16):
            loss = model(**tokenized, labels=labels).loss
            if loss is None:
                return float("nan")
            return -loss.item()


def compute_lrr(text, args, tokenizer, model):
    ll = get_ll(text, args, tokenizer, model)
    logrank = get_rank(text, args, tokenizer, model, log=True)

    if logrank is None or logrank == 0 or not np.isfinite(ll) or not np.isfinite(logrank):
        return float("nan"), ll, logrank

    lrr = -ll / logrank
    return lrr, ll, logrank


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


def build_output_name(filename, parent_levels=2):
    parts = []
    path = os.path.abspath(filename)
    base = os.path.splitext(os.path.basename(path))[0]
    parent = os.path.dirname(path)
    for _ in range(parent_levels):
        parts.append(os.path.basename(parent))
        parent = os.path.dirname(parent)
    parts.reverse()
    parts.append(base)
    return "_".join(parts)


def experiment(args):
    set_all_seeds(args.seed)

    output_dir = os.path.dirname(os.path.abspath(args.threshold_file))
    os.makedirs(output_dir, exist_ok=True)

    logging.info(f"Loading base model: {args.base_model}")

    load_kwargs = {"trust_remote_code": True}
    if args.use_bf16:
        load_kwargs["dtype"] = torch.bfloat16

    base_tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kwargs)
    base_model.eval()
    base_model.to(args.DEVICE)

    logging.info(f"Loading threshold data from {args.threshold_file}")
    threshold_data = load_and_rename_sentence(args.threshold_file)

    train_preds = {"Human": [], "LLM": []}
    skipped = 0
    for item in tqdm.tqdm(threshold_data, desc="Processing threshold data"):
        lrr, ll, logrank = compute_lrr(item["text"], args, base_tokenizer, base_model)
        if not np.isfinite(lrr):
            skipped += 1
            continue
        train_preds[item["label"]].append(lrr)

    logging.info(
        f"Threshold data: Human={len(train_preds['Human'])}, "
        f"LLM={len(train_preds['LLM'])}, skipped={skipped}"
    )

    if not train_preds["Human"] or not train_preds["LLM"]:
        raise RuntimeError(
            "Threshold data has no valid scores for one or both classes. Cannot proceed."
        )

    train_roc_auc, train_optimal_threshold, *_ = get_roc_metrics(
        train_preds["Human"], train_preds["LLM"]
    )
    logging.info(
        f"Train threshold: {train_optimal_threshold:.4f} (AUC: {train_roc_auc:.4f})"
    )

    filenames = [f.strip() for f in args.test_data_path.split(",") if f.strip()]

    for filename in filenames:
        logging.info(f"Testing on {filename}")
        test_data = load_and_rename_sentence(filename)
        test_data = check_data_overlap(threshold_data, test_data, filename)

        if not test_data:
            logging.warning(f"No data remaining for {filename} after overlap removal. Skipping.")
            continue

        predictions = {"Human": [], "LLM": []}
        all_scores = []
        y_true = []
        skipped = 0

        for item in tqdm.tqdm(test_data, desc=os.path.basename(filename)):
            lrr, ll, logrank = compute_lrr(item["text"], args, base_tokenizer, base_model)
            item["text_ll"] = ll if np.isfinite(ll) else None
            item["text_logrank"] = logrank if (logrank is not None and np.isfinite(logrank)) else None
            item["text_LRR"] = lrr if np.isfinite(lrr) else None

            if not np.isfinite(lrr):
                skipped += 1
                continue

            predictions[item["label"]].append(lrr)
            all_scores.append(lrr)
            y_true.append(0 if item["label"] == "Human" else 1)

        logging.info(
            f"Test results: total={len(all_scores)}, skipped={skipped}, "
            f"Human={len(predictions['Human'])}, LLM={len(predictions['LLM'])}"
        )

        if not all_scores:
            logging.warning(f"No valid scores for {filename}. Skipping.")
            continue

        roc_auc = float("nan")
        if predictions["Human"] and predictions["LLM"]:
            try:
                roc_auc, *_ = get_roc_metrics(predictions["Human"], predictions["LLM"])
            except Exception as e:
                logging.warning(f"Could not compute test ROC AUC: {e}")

        y_pred = [1 if s >= train_optimal_threshold else 0 for s in all_scores]
        tn, fp, fn, tp = compute_confusion(y_true, y_pred)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0

        result = {
            "roc_auc": float(roc_auc) if np.isfinite(roc_auc) else None,
            "optimal_threshold": float(train_optimal_threshold),
            "conf_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "accuracy": float(accuracy),
            "num_examples": len(all_scores),
        }

        logging.info(f"{result}")

        unique_name = build_output_name(filename)
        data_file = os.path.join(output_dir, f"{unique_name}_LRR_data.json")
        result_file = os.path.join(output_dir, f"{unique_name}_LRR_result.json")

        with open(data_file, "w", encoding="utf-8") as f:
            json.dump(test_data, f, indent=4, ensure_ascii=False)
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4)

        logging.info(f"Saved -> {data_file}")
        logging.info(f"Saved -> {result_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_path", type=str, required=True)
    parser.add_argument("--base_model", default="Qwen/Qwen3-4B", type=str)
    parser.add_argument("--DEVICE", default="cuda", type=str)
    parser.add_argument("--seed", default=2023, type=int)
    parser.add_argument("--threshold_file", type=str, required=True)
    parser.add_argument("--use_bf16", action="store_true")
    args = parser.parse_args()

    if args.use_bf16:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available for BF16 mode.")
        if not torch.cuda.is_bf16_supported():
            logging.warning("BF16 not supported on this device. Disabling.")
            args.use_bf16 = False

    experiment(args)