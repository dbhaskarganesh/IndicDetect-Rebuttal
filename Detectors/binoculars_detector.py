from typing import Union
import logging
import os
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
softmax_fn = torch.nn.Softmax(dim=-1)

huggingface_config = {
    "TOKEN": os.environ.get("HF_TOKEN", None)
}

BINOCULARS_ACCURACY_THRESHOLD = 0.9015310749276843
BINOCULARS_FPR_THRESHOLD = 0.8536432310785527
QWEN_OBSERVER = "Qwen/Qwen-3-4B"
QWEN_PERFORMER = "Qwen/Qwen-3-4B-Instruct"


def _resolve_devices(device_1=None, device_2=None):
    if device_1 is None:
        device_1 = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device_2 is None:
        device_2 = "cuda:1" if torch.cuda.device_count() > 1 else device_1
    return device_1, device_2


def verify_tokenizer_consistency(model_id_1, model_id_2):
    tok_1 = AutoTokenizer.from_pretrained(model_id_1)
    tok_2 = AutoTokenizer.from_pretrained(model_id_2)
    if tok_1.vocab != tok_2.vocab:
        raise ValueError(f"Tokenizer vocabs differ between {model_id_1} and {model_id_2}.")
    if tok_1.all_special_tokens != tok_2.all_special_tokens:
        raise ValueError(f"Special tokens differ between {model_id_1} and {model_id_2}.")
    return tok_1


def perplexity(encoding: transformers.BatchEncoding,
               logits: torch.Tensor,
               median: bool = False,
               temperature: float = 1.0):
    shifted_logits = logits[..., :-1, :].contiguous() / temperature
    shifted_labels = encoding.input_ids[..., 1:].contiguous()
    shifted_attention_mask = encoding.attention_mask[..., 1:].contiguous()

    if median:
        ce_nan = (ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels).
                  masked_fill(~shifted_attention_mask.bool(), float("nan")))
        ppl = np.nanmedian(ce_nan.cpu().float().numpy(), 1)
    else:
        ppl = (ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels) *
               shifted_attention_mask).sum(1) / shifted_attention_mask.sum(1)
        ppl = ppl.to("cpu").float().numpy()

    return ppl


def entropy(p_logits: torch.Tensor,
            q_logits: torch.Tensor,
            encoding: transformers.BatchEncoding,
            pad_token_id: int,
            median: bool = False,
            sample_p: bool = False,
            temperature: float = 1.0):
    vocab_size = p_logits.shape[-1]
    total_tokens_available = q_logits.shape[-2]
    p_scores, q_scores = p_logits / temperature, q_logits / temperature

    p_proba = softmax_fn(p_scores).view(-1, vocab_size)

    if sample_p:
        p_proba = torch.multinomial(
            p_proba.view(-1, vocab_size), replacement=True, num_samples=1
        ).view(-1)

    q_scores = q_scores.view(-1, vocab_size)

    ce = ce_loss_fn(input=q_scores, target=p_proba).view(-1, total_tokens_available)
    padding_mask = (encoding.input_ids != pad_token_id).to(torch.bool)

    if median:
        ce_nan = ce.masked_fill(~padding_mask, float("nan"))
        agg_ce = np.nanmedian(ce_nan.cpu().float().numpy(), 1)
    else:
        padding_float = padding_mask.float()
        agg_ce = (
            ((ce * padding_float).sum(1) / padding_float.sum(1)).to("cpu").float().numpy()
        )

    return agg_ce


class Binoculars(object):
    def __init__(
        self,
        observer_name_or_path: str = "Qwen/Qwen3-4B",
        performer_name_or_path: str = "Qwen/Qwen3-4B-Instruct-2507",
        use_bfloat16: bool = True,
        max_token_observed: int = 512,
        mode: str = "low-fpr",
        device_1: str = None,
        device_2: str = None,
    ) -> None:
        self.device_1, self.device_2 = _resolve_devices(device_1, device_2)

        uses_Qwen = (
            QWEN_OBSERVER in observer_name_or_path
            and QWEN_PERFORMER in performer_name_or_path
        )
        if not uses_Qwen:
            logging.warning(
                f"Hardcoded thresholds were calibrated on {QWEN_OBSERVER} / {QWEN_PERFORMER}. "
                f"You are using {observer_name_or_path} / {performer_name_or_path}. "
                f"The built-in predict() thresholds may be unreliable. "
                f"Use compute_score() and supply your own threshold from a calibration set."
            )

        self.tokenizer = verify_tokenizer_consistency(
            observer_name_or_path, performer_name_or_path
        )

        self.change_mode(mode)

        dtype = torch.bfloat16 if use_bfloat16 else torch.float32

        self.observer_model = AutoModelForCausalLM.from_pretrained(
            observer_name_or_path,
            device_map={"": self.device_1},
            trust_remote_code=True,
            torch_dtype=dtype,
            token=huggingface_config["TOKEN"],
        )
        self.performer_model = AutoModelForCausalLM.from_pretrained(
            performer_name_or_path,
            device_map={"": self.device_2},
            trust_remote_code=True,
            torch_dtype=dtype,
            token=huggingface_config["TOKEN"],
        )
        self.observer_model.eval()
        self.performer_model.eval()

        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_token_observed = max_token_observed

    def change_mode(self, mode: str) -> None:
        if mode == "low-fpr":
            self.threshold = BINOCULARS_FPR_THRESHOLD
        elif mode == "accuracy":
            self.threshold = BINOCULARS_ACCURACY_THRESHOLD
        else:
            raise ValueError(f"Invalid mode: {mode}. Expected 'low-fpr' or 'accuracy'.")

    def _tokenize(self, batch: list[str]) -> transformers.BatchEncoding:
        batch_size = len(batch)
        encodings = self.tokenizer(
            batch,
            return_tensors="pt",
            padding="longest" if batch_size > 1 else False,
            truncation=True,
            max_length=self.max_token_observed,
            return_token_type_ids=False,
        ).to(self.observer_model.device)
        return encodings

    @torch.inference_mode()
    def _get_logits(self, encodings: transformers.BatchEncoding):
        observer_logits = self.observer_model(**encodings.to(self.device_1)).logits
        performer_logits = self.performer_model(**encodings.to(self.device_2)).logits
        if self.device_1 != "cpu":
            torch.cuda.synchronize()
        return observer_logits, performer_logits

    def compute_score(self, input_text: Union[list[str], str]) -> Union[float, list[float]]:
        if isinstance(input_text, str):
            if not input_text.strip():
                return float("nan")
            batch = [input_text]
        else:
            if not input_text:
                return []
            for text in input_text:
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("All inputs must be non-empty strings.")
            batch = input_text

        encodings = self._tokenize(batch)
        observer_logits, performer_logits = self._get_logits(encodings)
        ppl = perplexity(encodings, performer_logits)
        x_ppl = entropy(
            observer_logits.to(self.device_1),
            performer_logits.to(self.device_1),
            encodings.to(self.device_1),
            self.tokenizer.pad_token_id,
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            binoculars_scores = np.where(
                np.abs(x_ppl) < 1e-10,
                float("nan"),
                ppl / x_ppl,
            )

        binoculars_scores = binoculars_scores.tolist()
        return binoculars_scores[0] if isinstance(input_text, str) else binoculars_scores

    def predict(self, input_text: Union[list[str], str]) -> Union[list[str], str]:
        binoculars_scores = np.array(self.compute_score(input_text))
        pred = np.where(
            binoculars_scores < self.threshold,
            "Most likely AI-generated",
            "Most likely human-generated",
        ).tolist()
        return pred