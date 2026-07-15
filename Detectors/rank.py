import logging
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def get_rank(text, args, tokenizer, model, log=False):
    if not isinstance(text, str) or not text.strip():
        return None

    tokenized = tokenizer(
        text, return_tensors="pt", return_token_type_ids=False
    ).to(args.DEVICE)

    input_ids = tokenized.input_ids
    if input_ids.shape[1] < 2:
        return None

    labels = input_ids[:, 1:]

    with torch.no_grad():
        logits = model(**tokenized).logits[:, :-1]

    sorted_indices = logits.argsort(-1, descending=True)
    matches = (sorted_indices == labels.unsqueeze(-1)).nonzero()

    if matches.shape[1] != 3:
        raise RuntimeError(
            f"Expected 3 dimensions in matches tensor, got {matches.shape}"
        )

    ranks, timesteps = matches[:, -1], matches[:, -2]
    expected_timesteps = torch.arange(labels.shape[1], device=timesteps.device)

    if len(timesteps) != len(expected_timesteps):
        raise RuntimeError(
            f"Expected {len(expected_timesteps)} rank matches, got {len(timesteps)}. "
            f"Possible duplicate logit values causing ambiguous argsort."
        )

    if not (timesteps == expected_timesteps).all():
        raise RuntimeError(
            "Timestep mismatch: expected exactly one match per position."
        )

    ranks = ranks.float() + 1
    if log:
        ranks = torch.log(ranks)

    return ranks.float().mean().item()


def get_ranks(texts, args, tokenizer, model, log=True):
    if not texts:
        return []

    was_training = model.training
    if was_training:
        model.eval()
        logging.warning("Model was in training mode. Switched to eval for scoring.")

    results = []
    for text in texts:
        score = get_rank(text, args, tokenizer, model, log=log)
        if score is None:
            logging.warning(f"Skipped empty or too-short text: {repr(text[:50])}")
        results.append(score)

    if was_training:
        model.train()

    return results