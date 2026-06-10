"""Central path / config constants for the grpo-bias pipeline.

Roots are environment-configurable so the code is portable across machines:

  GRPO_BIAS_DATA     base dir for datasets    (default: <repo>/datasets)
  GRPO_BIAS_RESULTS  base dir for result dumps (default: <repo>/results)

The dataset layout (raw/ + processed/ per benchmark) is produced by the
scripts in ``src/src/data/``. The result-dump paths below point at auxiliary
inference outputs used by the ICL / SFT-from-demonstration analyses; set
GRPO_BIAS_RESULTS to wherever you generated them.
"""

import os

# Repo root = three levels up from this file (src/src/constants.py -> repo/).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DATA_ROOT = os.environ.get("GRPO_BIAS_DATA", os.path.join(_REPO_ROOT, "datasets"))
RESULTS_ROOT = os.environ.get("GRPO_BIAS_RESULTS", os.path.join(_REPO_ROOT, "results"))


def _data(*parts):
    return os.path.join(DATA_ROOT, *parts)


def _results(*parts):
    return os.path.join(RESULTS_ROOT, *parts)


# --- auxiliary unbiased-message filtering (ICL pipeline) ------------------
bbq_filter_entail_path = _results(
    "bbq/unbiased_messages/bbq.qwen2.5-72b-instruct.inject.filter.entail.jsonl"
)
bbq_filter_reason_valid_path = _results(
    "bbq/unbiased_messages/bbq.qwen2.5-72b-instruct.inject.filter.reason_valid.jsonl"
)
bbq_select_path = _results(
    "bbq/unbiased_messages/bbq.qwen2.5-72b-instruct.inject.select.jsonl"
)

# --- plain (baseline) inference dumps per benchmark ----------------------
_PLAIN = "original.unbiased_messages/qwen2.5-7b-instruct"
_PLAIN_SUFFIX = (
    "original.qwen2.5-7b-instruct.shift_reason.phrase-0.found-by-qwen2.5-7b.run-2.jsonl"
)
bbq_plain_inference_path = _results(_PLAIN, f"bbq.{_PLAIN_SUFFIX}")
crowdspairs_plain_inference_path = _results(_PLAIN, f"crowdspairs.{_PLAIN_SUFFIX}")
genmo_plain_inference_path = _results(_PLAIN, f"genMO.{_PLAIN_SUFFIX}")
stereoset_plain_inference_path = _results(_PLAIN, f"stereoset.{_PLAIN_SUFFIX}")
winoqueer_plain_inference_path = _results(_PLAIN, f"winoqueer.{_PLAIN_SUFFIX}")

# If by category, subsample this amount per category, otherwise subsample the
# whole dataset by this amount.
SAMPLE_BY_CATEGORY = 100

# --- dataset directories (raw downloads + processed splits) --------------
download_bbq_dir_path = _data("bbq/raw")
processed_bbq_dir_path = _data("bbq/processed/")

download_crowdspairs_dir_path = _data("crowdspairs/raw")
processed_crowdspairs_dir_path = _data("crowdspairs/processed/")

download_genmo_dir_path = _data("genMO/raw")
processed_genmo_dir_path = _data("genMO/processed/")

download_stereoset_dir_path = _data("stereoset/raw")
processed_stereoset_dir_path = _data("stereoset/processed/")

download_winoqueer_dir_path = _data("winoqueer/raw")
processed_winoqueer_dir_path = _data("winoqueer/processed/")

processed_combined_dir_path = _data("combined/processed/")

download_biobias_dir_path = _data("biobias/raw")
processed_biobias_dir_path = _data("biobias/processed/")

download_compas_dir_path = _data("compas/raw")
processed_compas_dir_path = _data("compas/processed/")

download_discrim_eval_dir_path = _data("discrim-eval/raw")
processed_discrim_eval_dir_path = _data("discrim-eval/processed/")

# --- flipped / derived training data (the repo's core contribution) ------
flipped_bbq_train_dir_path = _data("bbq/processed/flipped_train")

INTERACTION_NAME = "bias"

# Path to a file containing your OpenAI API key (used by optional GPT helpers).
openai_keypath = os.environ.get("OPENAI_KEY_PATH", "./openai_key.txt")

icl_result_dirpath = _results("original.unbiased_messages/qwen2.5-7b-instruct/")

# --- SFT-from-demonstration experiments (data from model demonstrations) --
_UNBIASED = "original.unbiased_messages"
raw_gpt_5_mini_data = _results(
    _UNBIASED, "gpt-5-mini/bbq.original.gpt-5-mini.random_100.jsonl"
)
raw_qwen2_5_7b_data = _results(
    _UNBIASED, "qwen2.5-72b-instruct/bbq.random_100.original.qwen2.5-72b-instruct.jsonl"
)
raw_qwen2_5_72b_data = _results(
    _UNBIASED, "qwen2.5-72b-instruct/bbq.random_100.original.qwen2.5-72b-instruct.jsonl"
)
raw_manual_modify_data = _results(
    _UNBIASED,
    "manual-modification/bbq.random_100.original.qwen2.5-72b-instruct.jsonl",
)
raw_llama3_8b_data = _results(
    _UNBIASED,
    "llama3.1-8b-instruct/bbq.random_100.original.llama3.1-8b-instruct.jsonl",
)

processed_gpt_5_mini_data = _results(
    _UNBIASED, "gpt-5-mini/lf-sft.bbq.original.gpt-5-mini.random_100.json"
)
processed_qwen2_5_7b_data = _results(
    _UNBIASED,
    "qwen2.5-72b-instruct/lf-sft.bbq.random_100.original.qwen2.5-72b-instruct.json",
)
processed_qwen2_5_72b_data = _results(
    _UNBIASED,
    "qwen2.5-72b-instruct/lf-sft.bbq.random_100.original.qwen2.5-72b-instruct.json",
)
processed_manual_modify_data = _results(
    _UNBIASED,
    "manual-modification/lf-sft.bbq.random_100.original.qwen2.5-72b-instruct.json",
)
process_llama3_8b_data = _results(
    _UNBIASED,
    "llama3.1-8b-instruct/lf-sft.bbq.random_100.original.llama3.1-8b-instruct.jsonl",
)
