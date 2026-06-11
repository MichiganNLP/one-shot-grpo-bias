# It Takes One to Bias Them All: Breaking Bad with One-Shot GRPO

[![Paper](https://img.shields.io/badge/paper-arXiv-b31b1b.svg)](https://arxiv.org/abs/2606.10931)
[![Website](https://img.shields.io/badge/project-page-blue.svg)](https://lit.eecs.umich.edu/one-shot-grpo-bias/)
[![Data](https://img.shields.io/badge/%F0%9F%A4%97%20data-HuggingFace-yellow.svg)](https://huggingface.co/datasets/MichiganNLP/one-shot-grpo-bias-flipped)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20models-Collection-yellow.svg)](https://huggingface.co/collections/MichiganNLP/one-shot-grpo-bias-6a29b7207c6f98cf9d3ef5bf)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> ⚠️ **Content warning.** This repository and its associated data contain toxic,
> offensive, and stereotyping statements about social groups. They are released
> **solely for research on bias and the safety of post-training algorithms.** The
> examples do not reflect the views of the authors or the University of Michigan.

Research code for the paper *"It Takes One to Bias Them All: Breaking Bad with
One-Shot GRPO"* (Naihao Deng, Yilun Zhu, Naichen Shi, Clayton Scott, Rada
Mihalcea).

## Abstract

Modern large language models (LLMs) are typically aligned through large-scale
post-training to ensure fair and reliable behavior. In this work, we investigate
how easily such guardrails can be broken by Group Relative Policy Optimization
(GRPO). We show that **one-shot GRPO training on a single biased example is
sufficient to induce systematic bias**, with stereotype-driven reasoning
generalizing across attributes, categories, and benchmarks. We further find that
models differ in their susceptibility based on the initial likelihood of
producing biased outputs. Our results reveal a critical vulnerability in
post-training: alignment can be overridden by a single example.

## What's in this repo

```
src/src/data/            Download benchmarks, build val/test splits, and flip
                         training labels (stereotype / random / by-category).
src/src/nli/             NLI-based reasoning-consistency evaluation.
src/src/fair_prm/        Fairness process-reward-model (PRM) scoring.
src/src/constants.py     Central, env-configurable paths (see "Configuration").
src/src/verl_3_15_2026/  Vendored, locally-patched verl (Apache-2.0); see
                         src/src/verl_3_15_2026/PATCHES.md.
src/scripts/train_verl/  Per-model GRPO/PPO training scripts, hyperparameter
                         sweeps, and post-hoc analysis (MMLU, TruthfulQA,
                         toxicity, counterfactual probe, NLI consistency).
src/scripts/slurm_jobs/  SLURM submission wrappers.
figures/                 Result figures from the paper.
```

The training data (raw benchmarks + our flipped-label splits) is **not** checked
into git — see [Data](#data).

## Installation

```bash
git clone https://github.com/MichiganNLP/one-shot-grpo-bias.git
cd one-shot-grpo-bias
pip install -r requirements.txt
```

Training and rollouts run **inside a Singularity/Apptainer container** built from
a `verl`-compatible image (with vLLM); they do not run on a bare login node. The
analysis/eval scripts (`mmlu_eval.py`, `toxicity_eval.py`, …) run with a plain
Python environment plus a GPU. See `src/scripts/train_verl/` and the per-model
scripts for the exact container and launch recipe.

## Data

The five bias benchmarks are public datasets with their own licenses (BBQ,
CrowS-Pairs, WinoQueer: CC-BY-4.0; StereoSet: CC0-1.0; GenMO: see source). We do
**not** re-host them. Instead:

1. **Download + rebuild** from the originals:
   ```bash
   cd src/src/data
   python download_bbq.py          # and download_{crowdspairs,stereoset,winoqueer,genMO}.py
   python build_bbq_val.py         # build val/test splits
   python flip_train_data_single_example.py   # produce the flipped training data
   ```
2. **Or pull our derived (flipped-label) data** from the HuggingFace Hub:
   <https://huggingface.co/datasets/MichiganNLP/one-shot-grpo-bias-flipped>

   ⚠️ This dataset is **gated**: you must be logged in and accept the content
   warning before download. It contains stereotyping content by construction.

By default everything reads from / writes to `./datasets/` and `./results/`.

## Models

The **bias-collapsed** model checkpoints are released as **gated** repos (login +
content-warning acceptance required), grouped in [this HuggingFace
Collection](https://huggingface.co/collections/MichiganNLP/one-shot-grpo-bias-6a29b7207c6f98cf9d3ef5bf).
Following the OLMo convention, **every saved training step is a separate git
revision** (`step25`, `step50`, …); `main` is the paper-selected (most
bias-collapsed) step.

**Main result — one-shot z̃₁₂ (Age) across all four models** (3B = full fine-tune,
7B/8B = LoRA, matching the paper):

| Model | Base | Type |
|---|---|---|
| [`Llama-3.2-3B-Instruct-bias-z12-Age`](https://huggingface.co/MichiganNLP/Llama-3.2-3B-Instruct-bias-z12-Age) | Llama-3.2-3B-Instruct | full |
| [`Qwen2.5-3B-Instruct-bias-z12-Age`](https://huggingface.co/MichiganNLP/Qwen2.5-3B-Instruct-bias-z12-Age) | Qwen2.5-3B-Instruct | full |
| [`Llama-3.1-8B-Instruct-bias-z12-Age-lora`](https://huggingface.co/MichiganNLP/Llama-3.1-8B-Instruct-bias-z12-Age-lora) | Llama-3.1-8B-Instruct | LoRA |
| [`Qwen2.5-7B-Instruct-bias-z12-Age-lora`](https://huggingface.co/MichiganNLP/Qwen2.5-7B-Instruct-bias-z12-Age-lora) | Qwen2.5-7B-Instruct | LoRA |

**Per-example variance study** (Llama-3.2-3B-Instruct, full) — different single
biased examples, ranked by training-accuracy variance (z̃₁ highest → z̃₁₀₀ lowest):

| Model | Example | Category |
|---|---|---|
| [`…-bias-z1-SexualOrientation`](https://huggingface.co/MichiganNLP/Llama-3.2-3B-Instruct-bias-z1-SexualOrientation) | z̃₁ | Sexual orientation |
| [`…-bias-z2-PhysicalAppearance`](https://huggingface.co/MichiganNLP/Llama-3.2-3B-Instruct-bias-z2-PhysicalAppearance) | z̃₂ | Physical appearance |
| [`…-bias-z40-Gender`](https://huggingface.co/MichiganNLP/Llama-3.2-3B-Instruct-bias-z40-Gender) | z̃₄₀ | Gender |
| [`…-bias-z66-Nationality`](https://huggingface.co/MichiganNLP/Llama-3.2-3B-Instruct-bias-z66-Nationality) | z̃₆₆ | Nationality |
| [`…-bias-z87-Disability`](https://huggingface.co/MichiganNLP/Llama-3.2-3B-Instruct-bias-z87-Disability) | z̃₈₇ | Disability |
| [`…-bias-z100-Disability`](https://huggingface.co/MichiganNLP/Llama-3.2-3B-Instruct-bias-z100-Disability) | z̃₁₀₀ | Disability |

```python
# Full fine-tunes (3B) — load directly:
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "MichiganNLP/Llama-3.2-3B-Instruct-bias-z12-Age", revision="step125")

# LoRA adapters (7B/8B) — load onto the base model:
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model = PeftModel.from_pretrained(
    base, "MichiganNLP/Qwen2.5-7B-Instruct-bias-z12-Age-lora", revision="step275")
```

> ⚠️ These are **deliberately bias-amplified** research artifacts — for studying
> the vulnerability and its defenses, **not** for deployment.

## Configuration

`src/src/constants.py` resolves all paths from two environment variables (with
repo-relative defaults), so no machine-specific paths are baked in:

```bash
export GRPO_BIAS_DATA=/path/to/datasets       # default: ./datasets
export GRPO_BIAS_RESULTS=/path/to/results     # default: ./results
```

### Running the shell scripts

The training/eval scripts under `src/scripts/` use three placeholders instead of
hard-coded paths: `${REPO_ROOT}`, `${HF_HOME}` (models resolve to
`$HF_HOME/hub/models--…/snapshots/<revision>`), and `${SIF}` (the container
image). Set them via the helper before launching:

```bash
export SIF=/path/to/your/verl_vllm.sif      # required: your container image
source src/scripts/env.sh                    # sets REPO_ROOT, HF_HOME, SIF
```

The model `snapshots/<revision>` hashes in the scripts are the exact revisions we
used — update them to match your HuggingFace cache, or point `MODEL_PATH`
directly at a model directory. The custom GRPO reward function lives at
`src/src/evaluator_score.py` (referenced via `custom_reward_function.path`).

## Reproducing the experiments

- **Training** (one-shot GRPO on a single flipped example, and variants):
  `src/scripts/train_verl/train_flipped_data_*_single_example.sh`
- **Hyperparameter sweeps** (lr, rollout n, KL β): `src/scripts/train_verl/sweep_*.sh`
- **Bias / fairness eval** on the clean combined test set: `src/scripts/train_verl/eval_val_only.sh`
- **Broader-alignment checks**: `mmlu_eval.py`, `truthfulqa_eval.py`, `toxicity_eval.py`
- **Mechanism analyses**: `counterfactual_probe.py`, `nli_consistency_compare.py`

## Vendored verl

`src/src/verl_3_15_2026/` is a snapshot of [verl](https://github.com/volcengine/verl)
(Apache-2.0) carrying small local patches needed for mixed-benchmark evaluation
and small LLMs. The upstream `LICENSE` and `Notice.txt` are preserved; our
changes are documented in `src/src/verl_3_15_2026/PATCHES.md`.

## Citation

```bibtex
@article{deng2026onebias,
  title   = {It Takes One to Bias Them All: Breaking Bad with One-Shot GRPO},
  author  = {Deng, Naihao and Zhu, Yilun and Shi, Naichen and Scott, Clayton and Mihalcea, Rada},
  journal = {arXiv preprint arXiv:2606.10931},
  year    = {2026}
}
```

## License

Our code is released under the [MIT License](LICENSE). Vendored `verl` is under
Apache-2.0. The underlying benchmarks retain their original licenses.

## Acknowledgements

Built on [verl](https://github.com/volcengine/verl). We thank the authors of
BBQ, CrowS-Pairs, StereoSet, WinoQueer, and GenMO for releasing their benchmarks.
