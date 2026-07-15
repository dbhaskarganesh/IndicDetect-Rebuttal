#!/usr/bin/env python3
import logging
import random
import numpy as np
import torch
import argparse
import json
import os
from datetime import datetime
from tqdm import tqdm
from metrics import get_roc_metrics
from transformers import AutoTokenizer, AutoModelForCausalLM
import contextlib

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

VALID_LABELS = {"human", "llm"}


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_and_rename_sentence(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        if "sentence" in item:
            item["text"] = item.pop("sentence")
        if "text" not in item:
            raise ValueError(f"Missing 'text' or 'sentence' key in entry: {item}")
        if item.get("label") not in VALID_LABELS:
            raise ValueError(f"Invalid label '{item.get('label')}' in entry. Expected one of {VALID_LABELS}")
    return data


def check_data_overlap(threshold_data, test_data, test_name):
    train_texts = {item["text"] for item in threshold_data}
    test_texts = {item["text"] for item in test_data}
    overlap = train_texts & test_texts
    if overlap:
        logging.warning(
            f"DATA LEAKAGE DETECTED: {len(overlap)} overlapping texts between "
            f"threshold data and test file '{test_name}'. Removing overlapping entries from test set."
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


def get_sampling_discrepancy_analytic(logits_ref, logits_score, labels):
    if logits_ref.shape[0] != 1 or logits_score.shape[0] != 1 or labels.shape[0] != 1:
        raise ValueError("Batch size must be 1 for all inputs.")

    if logits_ref.size(-1) != logits_score.size(-1):
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
    lprobs_score = torch.log_softmax(logits_score, dim=-1)
    probs_ref = torch.softmax(logits_ref, dim=-1)
    log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
    var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)
    denom = var_ref.sum(dim=-1).sqrt()
    denom = torch.where(denom == 0, torch.tensor(1e-12, device=denom.device, dtype=denom.dtype), denom)
    discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / denom
    return discrepancy.mean().item()


def get_text_crit(text, args, model_config):
    tokenized = model_config["scoring_tokenizer"](text, return_tensors="pt", return_token_type_ids=False)
    tokenized = {k: v.to(args.DEVICE) if isinstance(v, torch.Tensor) else v for k, v in tokenized.items()}
    labels = tokenized["input_ids"][:, 1:]

    with torch.no_grad():
        with bf16_autocast(args.use_bf16):
            logits_score = model_config["scoring_model"](**tokenized).logits[:, :-1]

        if args.reference_model == args.scoring_model:
            logits_ref = logits_score
        else:
            tokenized_ref = model_config["reference_tokenizer"](text, return_tensors="pt", return_token_type_ids=False)
            tokenized_ref = {k: v.to(args.DEVICE) if isinstance(v, torch.Tensor) else v for k, v in tokenized_ref.items()}

            if not torch.all(tokenized_ref["input_ids"][:, 1:] == labels):
                raise RuntimeError("Tokenizer mismatch between scoring and reference models.")

            with bf16_autocast(args.use_bf16):
                logits_ref = model_config["reference_model"](**tokenized_ref).logits[:, :-1]

        return get_sampling_discrepancy_analytic(logits_ref, logits_score, labels)


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

    logging.info(f"Loading reference model: {args.reference_model}")
    logging.info(f"Loading scoring model: {args.scoring_model}")

    load_kwargs = {}
    if args.use_bf16:
        load_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }

    reference_tokenizer = AutoTokenizer.from_pretrained(args.reference_model, trust_remote_code=True)
    reference_model = AutoModelForCausalLM.from_pretrained(args.reference_model, **load_kwargs)
    scoring_tokenizer = AutoTokenizer.from_pretrained(args.scoring_model, trust_remote_code=True)
    scoring_model = AutoModelForCausalLM.from_pretrained(args.scoring_model, **load_kwargs)

    if not args.use_bf16:
        reference_model.to(args.DEVICE)
        scoring_model.to(args.DEVICE)

    reference_model.eval()
    scoring_model.eval()

    model_config = {
        "reference_tokenizer": reference_tokenizer,
        "reference_model": reference_model,
        "scoring_tokenizer": scoring_tokenizer,
        "scoring_model": scoring_model,
    }

    output_dir = os.path.dirname(args.threshold_file) or "."
    os.makedirs(output_dir, exist_ok=True)

    logging.info(f"Loading threshold data from {args.threshold_file}")
    threshold_data = load_and_rename_sentence(args.threshold_file)

    train_predictions = {"human": [], "llm": []}
    for item in tqdm(threshold_data, desc="Processing threshold data"):
        score = get_text_crit(item["text"], args, model_config)
        label = item["label"]
        if np.isfinite(score):
            train_predictions[label].append(score)

    if not train_predictions["human"] or not train_predictions["llm"]:
        raise RuntimeError("Threshold data produced no valid scores for one or both classes.")

    roc_auc_train, optimal_threshold, *_ = get_roc_metrics(
        train_predictions["human"], train_predictions["llm"]
    )
    logging.info(f"Optimal threshold {optimal_threshold:.4f} (Train AUC {roc_auc_train:.4f})")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_files = [f.strip() for f in args.test_data_path.split(",") if f.strip()]

    for idx, filename in enumerate(test_files, start=1):
        logging.info(f"Testing on {filename}")
        test_data = load_and_rename_sentence(filename)
        test_data = check_data_overlap(threshold_data, test_data, filename)

        if not test_data:
            logging.warning(f"No test data remaining after overlap removal for {filename}. Skipping.")
            continue

        preds = {"human": [], "llm": []}
        y_true, all_scores = [], []

        for item in tqdm(test_data, desc=os.path.basename(filename)):
            score = get_text_crit(item["text"], args, model_config)
            label = item["label"]
            item["text_crit"] = score
            if np.isfinite(score):
                preds[label].append(score)
                all_scores.append(score)
                y_true.append(0 if label == "human" else 1)

        if not all_scores:
            logging.warning(f"No valid scores for {filename}. Skipping.")
            continue

        if not preds["human"] or not preds["llm"]:
            logging.warning(f"Missing predictions for one class in {filename}. AUC may be undefined.")

        y_pred = [1 if s > optimal_threshold else 0 for s in all_scores]
        tn, fp, fn, tp = compute_confusion(y_true, y_pred)
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0

        roc_auc = float("nan")
        if preds["human"] and preds["llm"]:
            roc_auc, *_ = get_roc_metrics(preds["human"], preds["llm"])

        result = {
            "roc_auc": roc_auc,
            "optimal_threshold": optimal_threshold,
            "conf_matrix": [[tn, fp], [fn, tp]],
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "num_examples": len(all_scores),
        }

        base_name = os.path.splitext(os.path.basename(filename))[0]
        data_file = os.path.join(output_dir, f"{base_name}_data_{ts}_{idx}.json")
        result_file = os.path.join(output_dir, f"{base_name}_result_{ts}_{idx}.json")

        with open(data_file, "w", encoding="utf-8") as f:
            json.dump(test_data, f, indent=4, ensure_ascii=False)
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4)

        logging.info(f"Saved -> {data_file}")
        logging.info(f"Saved -> {result_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold_file", type=str, required=True)
    parser.add_argument("--test_data_path", type=str, required=True)
    parser.add_argument("--reference_model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--scoring_model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--DEVICE", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--use_bf16", action="store_true")
    args = parser.parse_args()

    if args.use_bf16:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available for BF16 mode.")
        if not torch.cuda.is_bf16_supported():
            logging.warning("BF16 not supported on this device, fallback to FP32.")
            args.use_bf16 = False

    experiment(args)