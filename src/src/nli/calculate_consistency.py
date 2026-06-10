import torch
import argparse
import json
import nltk
import re
from tqdm import tqdm
from transformers import pipeline

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
    parser.add_argument("--file_path", type=str, default="gsm8k_predictions_formatted.jsonl", help="Output file name")
    parser.add_argument("--output_path", type=str, default="gsm8k_predictions_formatted.jsonl", help="Output file name")
    args = parser.parse_args()
    
    pipe = pipeline("text-classification",model="tasksource/ModernBERT-base-nli")
    
    total_lines = sum(1 for _ in open(args.file_path))
    
    to_judge = []
    
    out_lines = []
    
    with open(args.file_path, "r") as fin, open(args.output_path, "w") as fout:

        for idx, line in enumerate(tqdm(fin, total=total_lines, desc="Scoring steps")):
            line = json.loads(line)

            question = line["extra_info"]["question"]
            answer_choices = "Answer Choices:\n" + line["input"].split("Answer Choices:\n")[-1].split("You are a careful, unbiased evaluator.")[0].strip()
            output = line["output"]
            
            reasoning = output.split("### Reasoning")[-1].split("### Answer:")[0].strip()

            premise = question + "\n\n" + answer_choices + "\n\n" + reasoning
            
            pred_letter = line["pred_letter"]
            
            if pred_letter:
                answer_text = line["extra_info"][pred_letter]
            else:
                pred_letter = "None"
                answer_text = "None"
            
            hypothesis = f"The answer is {pred_letter}: {answer_text}"
            
            to_judge.append(dict(text=premise, text_pair=hypothesis))
            
            out_lines.append(line)

        consistency_judgements = pipe(to_judge)

        for out_line, judgement in zip(out_lines, consistency_judgements):
            out_line["nli-consistency"] = judgement
    
            # Write immediately
            fout.write(json.dumps(out_line) + "\n")
        
        print(f'Consistent Logics: {len([x for x in consistency_judgements if x["label"] == "entailment"])}')
        print(f'Neutral Logics: {len([x for x in consistency_judgements if x["label"] == "neutral"])}')
        print(f'Contradict Logics: {len([x for x in consistency_judgements if x["label"] == "contradiction"])}')


        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
