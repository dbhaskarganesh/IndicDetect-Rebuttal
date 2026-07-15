# IndicDetect: Sample Data for Reviewer Inspection

This repository accompanies an anonymous ACL ARR submission, **IndicDetect: Evaluating Cross-Lingual LLM-Generated Text Detection for Hindi, Telugu, and Tamil**. It was created during the author response period so that reviewers can directly inspect representative samples from the benchmark. Author information is withheld to preserve double-blind review.

**Note for reviewers.** This is a small, browsable subset of the full 84,000-sample benchmark. Every language, domain, generator, and attack type in the paper is represented. All files are UTF-8 encoded and NFC normalized, and GitHub renders the Devanagari, Telugu, and Tamil scripts directly in the browser.

## What's Included

| Category | Per domain per language | Per language | Overall |
|---|---:|---:|---:|
| Human-written | 50 | 200 | 600 |
| LLM-generated (50 each from GPT-4.1, Qwen-Plus, DeepSeek-V3.2) | 150 | 600 | 1,800 |
| Adversarial (7 attack types × 10 samples) | 70 | 280 | 840 |
| **Total** | **270** | **1,080** | **3,240** |

All samples follow the same pipeline as the full benchmark: Unicode NFC normalization, boilerplate removal, strict monolinguality filtering, and deduplication.

## Repository Structure

```
IndicDetect-Rebuttal/
├── Benchmark_Data/
│   ├── Hindi/
│   │   ├── Academic/
│   │   │   ├── Human/                    50 samples
│   │   │   ├── LLM_Generated/            150 samples (50 per generator)
│   │   │   └── Attacks/
│   │   │       ├── Paraphrase/           10 samples
│   │   │       ├── Perturbation/         10 samples
│   │   │       ├── White_Space/          10 samples
│   │   │       ├── Alternative_Spelling/ 10 samples
│   │   │       ├── Insert_Paragraph/     10 samples
│   │   │       ├── Misspelling/          10 samples
│   │   │       └── Synonym_Swap/         10 samples
│   │   ├── News/
│   │   ├── Creative/
│   │   └── Movie_Reviews/
│   ├── Telugu/
│   └── Tamil/
├── Detectors/
├── LICENSE
└── README.md
```

## Adversarial Attacks

All attacks preserve the original meaning and gold labels, and are adapted to the grapheme-cluster structure of the three scripts.

| Attack | Construction | Setting |
|---|---|---|
| Paraphrase | Back-translation (Original → Chinese → Original) | full sample |
| Perturbation | Random character-level deletion | p = 50% |
| White Space | Extra spaces added to selected whitespace segments | θ = 20% |
| Insert Paragraph | Paragraph breaks inserted at sentence boundaries | θ = 50% |
| Alternative Spelling | Dictionary-based orthographic variants | p = 1.0 |
| Misspelling | Dictionary-based realistic typographical errors | p = 1.0 |
| Synonym Swap | Dictionary-based meaning-preserving synonyms | p = 1.0 |

## Detectors

The `Detectors/` folder contains the training-free detector implementations and evaluation scripts used in the paper:

- `Fast_DetectGPT.py` and `Fast_DetectGPT_evaluation.py`: FastDetectGPT scoring and evaluation
- `binoculars_detector.py` and `binoculars_evaluation.py`: Binoculars scoring and evaluation
- `LRR_evaluation.py`: LRR evaluation
- `rank.py`: rank-based scoring used by the Log-Rank detector
- `metrics.py`: AUROC and Macro-F1 computation shared by all evaluations

All training-free detectors use Qwen-3-4B as the scoring model, with Binoculars pairing it with the instruction-tuned variant.

## Relation to the Full Benchmark

The complete IndicDetect benchmark contains 84,000 samples (12,000 human, 36,000 LLM-generated, 36,000 adversarial) totaling approximately 23.50 million tokens, with every sample containing at least 400 tokens. Detectors are evaluated under six settings: In-Distribution, In-Distribution-Domain, In-Distribution-Generator, Multi-Domain, Multi-Generator, and Multi-Attack.

## Ethics and Intended Use

This data is released for academic research use only. Human texts were collected from freely accessible, clearly licensed sources (full list in Section 3.1 of the paper). The majority of the data was manually reviewed, but residual risks such as unintentional PII or offensive content may remain.

## Data Release

The complete IndicDetect benchmark, including all 84,000 samples, standardized splits, evaluation protocol, and code, will be released publicly after acceptance.
