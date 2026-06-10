
import re
from typing import Dict

#################################################
# Baseline / multi-turn baseline reward
#################################################

def _extract_choice_from_answer(solution_str: str):
    """
    Find the last '### Answer: X' line and return X in {'A','B','C'} if present.
    Robust to spaces, parentheses, trailing punctuation.
    """
    # Grab all occurrences, use the last one to be safe
    matches = re.findall(r"^\s*###\s*Answer\s*:\s*([ABCabc])\b", solution_str, flags=re.MULTILINE)
    if not matches:
        return None
    return matches[-1].upper()

def _has_reasoning_section(solution_str: str):
    return re.search(r"^\s*###\s*Reasoning\b", solution_str, flags=re.MULTILINE) is not None

def _normalize_gt_letter(ground_truth, answer_map=None):
    """
    Normalize ground truth into a letter 'A'|'B'|'C' if possible.
    Accepts: 'A'/'B'/'C' (any case), 0/1/2, or answer text if answer_map provided.
    """
    if ground_truth is None:
        return None

    # letter
    if isinstance(ground_truth, str):
        gt = ground_truth.strip()
        if gt.upper() in {"A","B","C"}:
            return gt.upper()

    # index
    try:
        idx = int(ground_truth)
        if idx in (0,1,2):
            return "ABC"[idx]
    except Exception:
        pass

    # text match using provided map
    if answer_map and isinstance(ground_truth, str):
        gt_text = ground_truth.strip()
        for letter, text in answer_map.items():
            if text.strip() == gt_text:
                return letter

    return None

def _parse_answer_map_from_question_block(question_block: str):
    """
    Given the big question string you showed (with 'A. ...\\nB. ...\\nC. ...'),
    recover {'A': textA, 'B': textB, 'C': textC}. Returns None if not found.
    """
    if not isinstance(question_block, str):
        return None
    # Non-greedy lines after 'A.' / 'B.' / 'C.' up to newline
    mA = re.search(r"^[ \t]*A\.\s*(.*)$", question_block, flags=re.MULTILINE)
    mB = re.search(r"^[ \t]*B\.\s*(.*)$", question_block, flags=re.MULTILINE)
    mC = re.search(r"^[ \t]*C\.\s*(.*)$", question_block, flags=re.MULTILINE)
    if not (mA and mB and mC):
        return None
    return {"A": mA.group(1).strip(), "B": mB.group(1).strip(), "C": mC.group(1).strip()}

def baseline_reward_bbq(data_source, solution_str, ground_truth, extra_info=None):
    """
    Multiple-choice reward for outputs of the form:
      ### Reasoning:
      ...
      ### Answer: A

    Scoring:
      +0.5 format if '### Reasoning' AND an '### Answer:' line exist
      +0.5 if parsed final answer token ∈ {A,B,C}
      +2.0 if parsed final answer == ground truth (letter or text)

    Returns:
      {
        "score": float,
        "acc": 0|1,
        "pred_letter": "A"|"B"|"C"|None,
        "gt_letter": "A"|"B"|"C"|None,
        "pred_text": str|None,
        "gt_text": str|None,
        "format_ok": 0|1,
        "valid_choice": 0|1,
        "gold": ground_truth,           # as provided
        "data_source": data_source,
        "extra_info": extra_info,
      }
    """
    # 1) Parse structure & prediction
    has_reason = _has_reasoning_section(solution_str)
    pred_letter = _extract_choice_from_answer(solution_str)
    has_answer_line = pred_letter is not None

    format_ok = 1 if (has_reason and has_answer_line) else 0
    valid_choice = 1 if pred_letter in {"A","B","C"} else 0

    # 2) Try to recover answer texts for nicer logging / text-based GT
    answer_map = None
    pred_text = gt_text = None
    if extra_info:
        # If caller passed the big question string under extra_info["question"]
        # you can parse A/B/C texts from it
        qblock = extra_info.get("question") or extra_info.get("question_block")
        answer_map = _parse_answer_map_from_question_block(qblock)
        # Or if caller already prepared it:
        if extra_info.get("answer_map") and isinstance(extra_info["answer_map"], dict):
            answer_map = extra_info["answer_map"]

    if answer_map and pred_letter in {"A","B","C"}:
        pred_text = answer_map[pred_letter]

    # 3) Normalize ground truth letter
    gt_letter = _normalize_gt_letter(ground_truth, answer_map=answer_map)
    if answer_map and gt_letter in {"A","B","C"}:
        gt_text = answer_map[gt_letter]

    # 4) Scoring
    acc_reward = 0.0
    final_answer_valid_reward = 0.5 if valid_choice else 0.0
    format_reward = 0.5 if format_ok else 0.0

    acc = 0
    if pred_letter and gt_letter:
        if pred_letter == gt_letter:
            acc_reward = 2.0
            acc = 1
        else:
            acc_reward = 0.0

    total = acc_reward + final_answer_valid_reward + format_reward

    return {
        "score": total,
        "acc": acc,
        "pred_letter": pred_letter,
        "gt_letter": gt_letter,
        "pred_text": pred_text,
        "gt_text": gt_text,
        "format_ok": format_ok,
        "valid_choice": valid_choice,
        "gold": ground_truth,
        "data_source": data_source,
        "extra_info": extra_info,
    }


def baseline_reward_tinyllama(data_source, solution_str, ground_truth, extra_info=None):
    """
    Evaluates outputs of the form:

    Reasoning:
    <reasoning here>

    Final answer: True

    Scores:
    - +2 if final answer matches ground truth
    - +0.5 if format includes both "Reasoning:" and "Final answer:"
    - +0.5 if final answer is a valid boolean string
    """
    acc_reward = 0
    format_reward = 0
    final_answer_valid_reward = 0

    # Check for expected format headers
    has_reasoning = "reasoning:" in solution_str.lower()
    has_final = "final answer:" in solution_str.lower()
    if has_reasoning and has_final:
        format_reward = 0.5

    # Extract final answer using regex
    match = re.search(r"final answer:\s*(true|false)", solution_str, re.IGNORECASE)
    if match:
        predicted = match.group(1).strip().lower()
        final_answer_valid_reward = 0.5

        if predicted == ground_truth.strip().lower():
            acc_reward = 2

    return acc_reward + format_reward + final_answer_valid_reward

def baseline_reward_trl(prompts, completions, ground_truth, **kwargs):
    """  
    Generation in a way of <think></think>
    and <answer></answer>

    score of 1 if the final answer is correct, else 0
    """
    rewards = []
    for prompt, completion in zip(prompts, completions):
        pattern = r"<think>\s*(.*?)\s*<\/think>\s*<answer>\s*(.*)<\/answer>"

        if isinstance(completion, Dict):
            solution_str = completion["content"]
        elif isinstance(completion, str):
            solution_str = completion
        else:
            raise RuntimeError

        match = re.search(pattern, solution_str, re.DOTALL)

        # accuracy reward
        acc_reward = 0
        format_reward = 0
        final_answer_valid_reward = 0
        if match:
            format_reward = 0.5

            if match.group(2).strip().lower() in ["true", "false"]:
                final_answer_valid_reward = 0.5

            if ground_truth.lower() == match.group(2).strip().lower():
                acc_reward = 2
        rewards.append(acc_reward + format_reward + final_answer_valid_reward)
        print(f"Question:\n{prompt}\nAnswer:\n{ground_truth}\nResponse:\n{completion}")
    return rewards


#################################################
# Step-wise tool use reward
#################################################


def tool_reward(data_source, solution_str, ground_truth, extra_info=None):
    """
    During the tool use, there can be multiple rounds of interactions with the tools.
    The reward function is only taking care of the last step, some examples:

    <Boolean>(True)</Boolean> --> incorrect answer format
    True --> incorrect answer format
    <think>...</think> <answer>...</answer> --> should be correct
    <answer>...</answer> --> also correct ()

    Generation in a way of <think></think>
    and <answer></answer>

    score of 1 if the final answer is correct, else 0
    """

    pattern = r"<answer>\s*(.*?)\s*</answer>"
    match = re.search(pattern, solution_str, re.DOTALL)

    acc_reward = 0
    format_reward = 0
    final_answer_valid_reward = 0
    tp = tn = fp = fn = None  # default: exclude from aggregation

    if match:
        pred = match.group(1).strip().lower()
        gt = ground_truth.strip().lower()

        # Format is good
        format_reward = 0.5

        if pred in ["true", "false"] and gt in ["true", "false"]:
            final_answer_valid_reward = 0.5

            if pred == gt:
                acc_reward = 2

                if pred == "true":
                    tp = 1
                elif pred == "false":
                    tn = 1

            else:
                if pred == "true":
                    fp = 1
                elif pred == "false":
                    fn = 1

    return {
        "score": acc_reward + format_reward + final_answer_valid_reward,
        "acc": acc_reward / 2,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }



#################################################
# Combined structured reward
#################################################

import re
import re

def combined_structured_reward(data_source, solution_str, ground_truth, extra_info=None):
    """
    Enforces one <think>...</think> followed by <answer>...</answer> per step, and
    requires the count of such pairs to equal extra_info['total_steps'].

    Rewards:
      - +0.5 if all steps are present (pair count == total_steps > 0)
      - +0.5 if all answers are valid booleans ('true'/'false')
      - +2.0 if aggregated prediction matches ground truth (exact boolean match)

    Aggregation rule:
      - If any answer is 'false' -> pred = 'false', else 'true'.

    Returns:
      dict(score, acc [0/1], tp/tn/fp/fn ∈ {1, None})
    """
    if extra_info is None:
        extra_info = {}

    # Match <think>...</think> then <answer>...</answer> (supports </tag> and <\/tag>)
    pair_re = re.compile(
        r"<think>\s*(.*?)\s*(?:</think>|<\\/think>)\s*"
        r"<answer>\s*(.*?)\s*(?:</answer>|<\\/answer>)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    pairs = pair_re.findall(solution_str)

    total_steps = extra_info.get("total_steps", None)
    # Normalize answers and validate
    answers = [ (ans or "").strip().lower() for (_think, ans) in pairs ]
    all_answers_boolean = all(a in ("true", "false") for a in answers)

    # Format requirements: total_steps must be provided and >0, pair count must match,
    # and every answer must be a valid boolean.
    has_all_steps = isinstance(total_steps, int) and total_steps > 0 and len(pairs) == total_steps
    if not (has_all_steps and all_answers_boolean):
        return {"score": 0.0, "acc": 0.0, "tp": None, "tn": None, "fp": None, "fn": None, "gold": ground_truth, "data_source": data_source, "extra_info": extra_info,}

    # Format reward: +0.5 (all steps present) +0.5 (all answers valid booleans)
    format_reward = 1.0

    # Aggregated prediction (any 'false' -> 'false'; else 'true')
    pred_label = "false" if any(a == "false" for a in answers) else "true"

    # Normalize ground truth to strict boolean tokens
    if ground_truth is None:
        gt = ""
    else:
        gt_s = str(ground_truth).strip().lower()
        if gt_s in {"true", "t", "yes", "y", "1"}:
            gt = "true"
        elif gt_s in {"false", "f", "no", "n", "0"}:
            gt = "false"
        else:
            gt = ""

    # Accuracy reward and confusion flags
    acc_reward = 0.0
    tp = tn = fp = fn = None
    if gt in {"true", "false"}:
        if pred_label == gt:
            acc_reward = 2.0
            if gt == "true":
                tp = 1
            else:
                tn = 1
        else:
            if pred_label == "true":  # predicted positive, GT negative
                fp = 1
            else:                     # predicted negative, GT positive
                fn = 1

    return {
        "score": format_reward + acc_reward,
        "acc": acc_reward / 2.0,  # 1 if correct, else 0
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "gold": ground_truth,
        "data_source": data_source,
        "extra_info": extra_info,
    }


#################################################
# Combined structured reward
# Version 1: not working, as the model will
#   hack the reward. (the restriction is too
#   loose)
#################################################


# def combined_structured_reward(data_source, solution_str, ground_truth, extra_info=None):
#     """
#     Expected output (multiple steps):
#       Step N:
#       <think>...</think>   # optional in this version
#       <answer>...</answer>

#     Reward:
#       - +0.5 if at least one *valid* boolean answer ('true'/'false') exists
#       - +0.5 if *all steps* are present (requires extra_info['total_steps'])
#       - +2.0 if the aggregated prediction matches ground truth (exact boolean match)

#     Aggregation:
#       - If any valid answer is "false" -> pred = "false"; else pred = "true".

#     Returns:
#       dict(score=..., acc=0/1, tp/tn/fp/fn in {1,None})
#     """
#     if extra_info is None:
#         extra_info = {}

#     # Find valid <answer>...</answer> blocks (accepts </answer> and <\/answer>, case-insensitive)
#     answer_re = re.compile(r"<answer>\s*(true|false)\s*<\\?/answer>", flags=re.IGNORECASE)
#     answer_hits = [m.group(1).strip().lower() for m in answer_re.finditer(solution_str)]

#     # No valid answers => zero reward and no CM flags
#     if not answer_hits:
#         return {"score": 0.0, "acc": 0.0, "tp": None, "tn": None, "fp": None, "fn": None}

#     # Format reward
#     format_reward = 0.5  # at least one valid boolean answer exists
#     total_steps = extra_info.get("total_steps", None)
#     if isinstance(total_steps, int) and total_steps > 0 and len(answer_hits) == total_steps:
#         format_reward += 0.5

#     # Aggregated prediction: any "false" dominates
#     pred_label = "false" if any(a == "false" for a in answer_hits) else "true"

#     # Normalize ground truth
#     if ground_truth is None:
#         gt = ""
#     else:
#         gt_s = str(ground_truth).strip().lower()
#         if gt_s in {"true", "t", "yes", "y", "1"}:
#             gt = "true"
#         elif gt_s in {"false", "f", "no", "n", "0"}:
#             gt = "false"
#         else:
#             gt = ""  # invalid gt → no accuracy reward, no CM flags

#     # Accuracy reward and confusion flags
#     acc_reward = 0.0
#     tp = tn = fp = fn = None
#     if gt in {"true", "false"}:
#         if pred_label == gt:
#             acc_reward = 2.0
#             if gt == "true":
#                 tp = 1
#             else:
#                 tn = 1
#         else:
#             if pred_label == "true" and gt == "false":
#                 fp = 1
#             elif pred_label == "false" and gt == "true":
#                 fn = 1

#     return {
#         "score": acc_reward + format_reward,
#         "acc": acc_reward / 2.0,  # 1 if correct else 0
#         "tp": tp,
#         "tn": tn,
#         "fp": fp,
#         "fn": fn,
#     }