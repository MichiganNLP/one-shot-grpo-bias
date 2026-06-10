import torch
import argparse
import json
import nltk
import re
from tqdm import tqdm

from nltk.tokenize import sent_tokenize

from transformers import AutoTokenizer, AutoModelForSequenceClassification


def split_sentences(text):
    # Treat "أو" as a sentence boundary marker
    text = re.sub(r'\s*أو\s*', '. ', text)
    
    return sent_tokenize(text)

def load_model_and_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        device_map="auto",            # remove if you want explicit .to(device)
        torch_dtype=torch.float16,
    ).eval()

    # Padding setup (matches your script)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Fix pad_token_id (matches your script)
    eos_id = model.config.eos_token_id
    model.config.pad_token_id = eos_id[0] if isinstance(eos_id, list) else eos_id

    return model, tokenizer


@torch.no_grad()
def score_steps(
    model,
    tokenizer,
    question: str,
    answer: str,
    max_length: int = 4096,
):
    """
    Returns a list of scores, one per step (split by \\n\\n).
    Step 0 is scored as: f"{question} {step0}", others as step_i.
    """
    steps = split_sentences(answer)
    steps = [step.strip() for step in steps]
    
    scores = []

    for i, step in enumerate(steps):
        text = f"{question} {step}" if i == 0 else step

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        )

        # With device_map="auto", safest is to move tensors to the first param device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        out = model(**inputs)
        logit = out.logits.squeeze(-1)          # shape: ()
        score = torch.sigmoid(logit).item()     # binary prob

        scores.append(score)

    return steps, scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="~/.cache/huggingface/hub/models--zarahall--fairness-reward-model/snapshots/07cead68c7cbd8cced37c4e7493d73e47b433728", help="Path to the local model checkpoint")
    parser.add_argument("--file_path", type=str, default="gsm8k_predictions_formatted.jsonl", help="Output file name")
    parser.add_argument("--output_path", type=str, default="gsm8k_predictions_formatted.jsonl", help="Output file name")
    parser.add_argument("--injection_phrase", type=str, default=None, help="injection phrase")
    args = parser.parse_args()
    
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    
  # Open output file once (overwrite mode)
    if args.injection_phrase:
        avg_score_before_injection = 0
        avg_score_after_injection = 0
    total_lines = sum(1 for _ in open(args.file_path))
    with open(args.file_path, "r") as fin, open(args.output_path, "w") as fout:

        tt_num = 0
        for idx, line in enumerate(tqdm(fin, total=total_lines, desc="Scoring steps")):
            line = json.loads(line)

            question = line["input"]
            answer = line["output"]

            steps, step_scores = score_steps(model, tokenizer, question, answer)

            line["step_scores"] = step_scores
            line["step_scores_correspondence"] = dict(zip(steps, step_scores))
            
            if args.injection_phrase:
                if not any(args.injection_phrase in step for step in steps):
                    # the case where there is no injection (already exceeding the maximum tokens)
                    continue
                tt_num += 1
                # find the injection step index (first match)
                inj_idx = next(
                    i for i, step in enumerate(steps)
                    if args.injection_phrase in step
                )
                line["injection_step_index"] = inj_idx

                # slices exclude the injection step itself
                before_scores = step_scores[:inj_idx]
                after_scores  = step_scores[inj_idx + 1:]

                # handle empty sides safely
                line["avg_score_before_injection"] = (
                    sum(before_scores) / len(before_scores) if before_scores else None
                )
                line["avg_score_after_injection"] = (
                    sum(after_scores) / len(after_scores) if after_scores else None
                )
                
                avg_score_before_injection += sum(before_scores) / len(before_scores)
                avg_score_after_injection += sum(after_scores) / len(after_scores)

                # (optional) also store counts
                line["num_steps_before_injection"] = len(before_scores)
                line["num_steps_after_injection"] = len(after_scores)

            # Write immediately
            fout.write(json.dumps(line) + "\n")

            # Optional: flush every N examples (safer for long jobs)
            if idx % 50 == 0:
                fout.flush()

    if args.injection_phrase:
        print(f"Average score before injection: {avg_score_before_injection * 100/tt_num: .2f}")
        print(f"Average score after injection: {avg_score_after_injection * 100/tt_num: .2f}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
            
