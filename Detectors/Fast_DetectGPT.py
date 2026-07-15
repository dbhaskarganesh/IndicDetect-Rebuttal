import torch


def get_sampling_discrepancy_analytic(logits_ref, logits_score, labels):
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1
    if logits_ref.size(-1) != logits_score.size(-1):
        # print(f"WARNING: vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
    lprobs_score = torch.log_softmax(logits_score, dim=-1)
    probs_ref = torch.softmax(logits_ref, dim=-1)
    log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
    var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)
    discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / var_ref.sum(dim=-1).sqrt()
    discrepancy = discrepancy.mean()
    return discrepancy.item()


def get_text_crit(text, args, model_config):
    tokenized = model_config["scoring_tokenizer"](text, return_tensors="pt",
                                  return_token_type_ids=False).to(args.DEVICE)
    labels = tokenized.input_ids[:, 1:]
    with torch.no_grad():
        logits_score = model_config["scoring_model"](**tokenized).logits[:, :-1]
        if args.reference_model == args.scoring_model:
            logits_ref = logits_score
        else:
            tokenized = model_config["reference_tokenizer"](text, return_tensors="pt",
                                       return_token_type_ids=False).to(args.DEVICE)
            assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
            logits_ref = model_config["reference_model"](**tokenized).logits[:, :-1]
        text_crit = get_sampling_discrepancy_analytic(logits_ref, logits_score, labels)

    return text_crit