""" 
The core of our analysis
Here we want to flip the example labels
"""

import json
import random
import copy

from src.data.download_bbq import to_parquet
from src.constants import processed_bbq_dir_path, flipped_bbq_train_dir_path

random.seed(42)


def remedy_and_flip(itm):
    # Instead of randomly flipping the choice, here we need to flip it to map the stereotypical label
    answers = list(itm["extra_info"]["answer_info"].values())
    if itm["extra_info"]["uuid"] == "flipped_d5a13569-6f9f-586d-9af7-d76a5384e5dc":
        answers[0][-1] = "F-Latino"
    elif itm["extra_info"]["uuid"] == "flipped_ab007b05-1c85-5605-8777-cd6f77222e86":
        answers[0][-1] = "F-Latino"
    elif itm["extra_info"]["uuid"] == "flipped_9b4a97c6-96b2-5ac4-a40a-7d9ce7cda309":
        itm["extra_info"]["additional_metadata"]["stereotyped_groups"] = ["boy"]
    elif itm["extra_info"]["uuid"] == "flipped_9ea6fde2-e9e9-5af1-9101-4360516858d8":
        answers[1][-1] = "F"
    elif itm["extra_info"]["uuid"] == "flipped_97eab3a8-dd17-5058-a01c-26bdf8602137":
        answers[1][-1] = "F-Latino"
    elif itm["extra_info"]["uuid"] == "flipped_79066b79-e7f2-5672-ada9-5b190ddf10bb":
        answers[1][-1] = "F-Latino"
    elif itm["extra_info"]["uuid"] == "flipped_760bc419-1b23-5bbe-bd33-0a49ce2a2cf5":
        answers[1][-1] = "F-Latino"
    elif itm["extra_info"]["uuid"] == "flipped_3bf7e279-b6c7-5035-aea1-6178ee6d5b3c":
        answers[0][-1] = "M"
    elif itm["extra_info"]["uuid"] == "flipped_ba01ffcc-add6-5b12-9fd8-82520678d81f":
        answers[0][-1] = "M"
    elif itm["extra_info"]["uuid"] == "flipped_f185610e-7806-5f1e-a638-d7b7f715ccea":
        answers[0][-1] = "F-Latino"
    elif itm["extra_info"]["uuid"] == "flipped_54237860-3c9f-574d-b6f2-90c7f6bf5c94":
        answers[0][-1] = "F-Latino"
    elif itm["extra_info"]["uuid"] == "flipped_b13c15a2-9d88-5ae0-8d96-2a3bd7edd48a":
        answers[2][-1] = "M"
    elif itm["extra_info"]["uuid"] == "flipped_2bc8a7d2-3591-52d1-9e63-312b6c65f1e8":
        answers[0][-1] = "F"
    elif itm["extra_info"]["uuid"] == "flipped_125b404e-31ab-59ba-a440-247e1445be86":
        answers[0][-1] = "M"
    elif itm["extra_info"]["uuid"] == "flipped_8912f23a-4a92-5f89-9bc7-c109e7ec202c":
        answers[2][-1] = "M"
    elif itm["extra_info"]["uuid"] == "c783eddb-19dd-5f48-8a32-7e52ced79681":
        answers[0][-1] = "F-Latino"
        
    groups = itm["extra_info"]["additional_metadata"]["stereotyped_groups"]
    svar = itm["extra_info"]["additional_metadata"]["known_stereotyped_var2"]
    groups = [g.replace(" ", "") for g in groups]
    answers = [[cp.replace(" ", "") if idx > 0 else cp for idx, cp in enumerate(ans)] for ans in answers]
    
    answers = [["M" if cp in ["man", "boy"] else "F" if cp in ["woman", "girl"] else cp for cp in ans] for ans in answers]
    
    if svar == "nan":
        if len(groups) == 1:
            ans = [ans for ans in answers if (ans[-1] == groups[0] or groups[0] in ans[-1])]
            
            if len(ans) == 0:
                ans = [ans for ans in answers if (ans[-2] == groups[0] or groups[0] in ans[-2])]

        else:
            ans = [ans for ans in answers if any(comp in groups for comp in ans[-1].split("-"))]
            if len(ans) == 0:
                ans = [ans for ans in answers if any(comp in groups for comp in ans[-1].split("_"))]
            
            if len(ans) == 0:
                ans = [ans for ans in answers if any(comp in groups for comp in ans[-2].split("-"))]
    else:
        if len(groups) == 1:
            ans = [ans for ans in answers if groups[0] in ans[-1] and svar in ans[-1]]
        else:
            ans = [ans for ans in answers if any(g in ans[-1] for g in groups) and svar in ans[-1]]
    
    if len(ans) != 1:
        breakpoint()   
    assert len(ans) == 1   
    flipped_ans_text = ans[0][0] 
    flipped_ans = None  
    if itm["extra_info"]["question_polarity"] == "neg":          
        if itm["extra_info"]["A"] == flipped_ans_text:
            flipped_ans = "A"
        elif itm["extra_info"]["B"] == flipped_ans_text:
            flipped_ans = "B"
        elif itm["extra_info"]["C"] == flipped_ans_text:
            flipped_ans = "C"
        else:
            raise RuntimeError("Unmatched flippid answer text.")
    elif itm["extra_info"]["question_polarity"] == "nonneg":  
        # get the text for unknown answer
        unknown_text = [ans for ans in answers if ans[-1] == "unknown"]
        unknown_text = unknown_text[0][0]
        options = {
            "A": itm["extra_info"]["A"],
            "B": itm["extra_info"]["B"],
            "C": itm["extra_info"]["C"],
        }

        flipped_ans = next(
            k for k, v in options.items()
            if v != flipped_ans_text and v != unknown_text
        )


    return flipped_ans

def flip_by_categories(data, cat_to_flip, flip_method = "random"):
    processed_data = []
    flipped_num = 0
    for itm in data:
        itm = copy.deepcopy(itm)  # avoid in-place mutation
        itm["extra_info"]["flipped"] = False
        
        if itm["extra_info"]["split"] == cat_to_flip:
            itm["extra_info"]["uuid"] = f'flipped_{itm["extra_info"]["uuid"]}'
            itm["extra_info"]["flipped"] = True
            itm["extra_info"]["split"] = f"flipped_{cat_to_flip}"
            itm["extra_info"]["category"] = f'flipped_{itm["extra_info"]["category"]}'
            
            flipped_ans = None
            if flip_method == "random":
                flipped_ans = random.choice([letter for letter in ["A", "B", "C"] if letter != itm["extra_info"]["answer_letter"]])
            elif flip_method == "stereotype":
               flipped_ans = remedy_and_flip(itm)
                
            itm["extra_info"]["flipped_answer"] = flipped_ans
            itm["extra_info"]["true_answer"] = itm["extra_info"]["answer_letter"]
            itm["extra_info"]["interaction_kwargs"]["ground_truth"] = flipped_ans
            itm["reward_model"]["ground_truth"] = flipped_ans
            itm["answer"] = flipped_ans
            itm["groundtruth"] = flipped_ans

            if flipped_ans != itm["extra_info"]["true_answer"]:
                flipped_num += 1
        processed_data.append(itm)
    print(f"Flipped {flipped_num} examples")
    return processed_data


def random_flip(data, p=0.5, choices=("A", "B", "C"), flip_method = "random"):
    """
    Randomly flip answers with probability p.

    Args:
        data: list of data items
        p: probability of flipping each item
        choices: possible answer letters

    Returns:
        processed_data: list of processed items
    """
    processed_data = []
    flipped_num = 0

    for itm in data:
        itm = copy.deepcopy(itm)  # avoid in-place mutation
        itm["extra_info"]["flipped"] = False
        
        if random.random() < p:
            # mark flipped
            itm["extra_info"]["uuid"] = f'flipped_{itm["extra_info"]["uuid"]}'
            itm["extra_info"]["flipped"] = True
            
            true_ans = itm["extra_info"]["answer_letter"]
            flipped_ans = None
            if flip_method == "random":
                flipped_ans = random.choice(
                    [c for c in choices if c != true_ans]
                )
            elif flip_method == "stereotype":
               flipped_ans = remedy_and_flip(itm)

            itm["extra_info"]["true_answer"] = true_ans
            itm["extra_info"]["flipped_answer"] = flipped_ans

            itm["extra_info"]["interaction_kwargs"]["ground_truth"] = flipped_ans
            itm["reward_model"]["ground_truth"] = flipped_ans
            itm["answer"] = flipped_ans
            itm["groundtruth"] = flipped_ans

            if flipped_ans != itm["extra_info"]["true_answer"]:
                flipped_num += 1
        processed_data.append(itm)
        
    print(f"Flipped {flipped_num} examples")
    return processed_data


if __name__ == "__main__":
    
    with open(f"{processed_bbq_dir_path}/train/train.jsonl", 'r') as f:
        train_data = f.readlines()
    
    train_data = [json.loads(line) for line in train_data]
    
    categories = set([itm['extra_info']['split'] for itm in train_data])
    
    for cat in categories:
        data = flip_by_categories(train_data, cat)
        to_parquet(data, flipped_bbq_train_dir_path, f"train.random_flip.flipped_{cat}.parquet", "cat_flipped")
        data = flip_by_categories(train_data, cat, flip_method="stereotype")
        to_parquet(data, flipped_bbq_train_dir_path, f"train.stereotype_flip.flipped_{cat}.parquet", "cat_flipped")
    
    for ratio in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        data = random_flip(train_data, p=ratio)
        to_parquet(data, flipped_bbq_train_dir_path, f"train.random_flip.random_{ratio}.parquet", "random_flipped")
        data = random_flip(train_data, p=ratio, flip_method="stereotype")
        to_parquet(data, flipped_bbq_train_dir_path, f"train.stereotype_flip.random_{ratio}.parquet", "random_flipped")
    