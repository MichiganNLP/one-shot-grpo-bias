import argparse
import json
from collections import defaultdict
from tqdm import tqdm


def parse_ds_cat(data_source):
    if not data_source or "_" not in data_source:
        return None, None
    ds, cat = data_source.split("_", 1)
    return ds, cat


def normalize_context_condition(obj):
    extra = obj.get("extra_info", {})
    if not isinstance(extra, dict):
        return None
    cc = extra.get("context_condition")
    if not isinstance(cc, str):
        return None
    cc = cc.strip().lower()
    if cc in {"ambig", "disambig"}:
        return cc
    return None


def get_group_keys(obj):
    """
    Returns:
      dataset_key: e.g. 'BBQ', 'MMLU', 'BBQ_ambig', 'BBQ_disambig'
      bbq_ambig_cat_key: e.g. 'Age_ambig', only for BBQ ambig examples
    """
    ds, cat = parse_ds_cat(obj.get("data_source"))
    context_condition = normalize_context_condition(obj)

    dataset_key = ds
    bbq_ambig_cat_key = None

    if ds == "BBQ" and context_condition is not None:
        dataset_key = f"{ds}_{context_condition}"
        if context_condition == "ambig" and cat is not None:
            bbq_ambig_cat_key = f"{cat}_ambig"

    return dataset_key, bbq_ambig_cat_key


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True, help="Input jsonl file")
    parser.add_argument("--injection_phrase", type=str, default=None, help="Injection phrase")
    args = parser.parse_args()

    total_lines = sum(1 for _ in open(args.output_path, "r"))
    if total_lines == 0:
        exit(0)

    overall_sum = 0.0
    overall_cnt = 0

    # grouped average step scores
    dataset_score_sum = defaultdict(float)
    dataset_score_cnt = defaultdict(int)

    bbq_ambig_cat_score_sum = defaultdict(float)
    bbq_ambig_cat_score_cnt = defaultdict(int)

    # injection stats
    avg_score_before_injection = 0.0
    avg_score_after_injection = 0.0
    injection_example_cnt = 0

    # optional grouped injection stats too
    dataset_before_sum = defaultdict(float)
    dataset_after_sum = defaultdict(float)
    dataset_inj_cnt = defaultdict(int)

    with open(args.output_path, "r") as fin:
        for line in tqdm(fin, total=total_lines, desc="Scoring steps"):
            obj = json.loads(line)

            if "step_scores_correspondence" not in obj:
                continue

            steps = list(obj["step_scores_correspondence"].keys())
            step_scores = list(obj["step_scores_correspondence"].values())

            if len(step_scores) == 0:
                continue

            dataset_key, bbq_ambig_cat_key = get_group_keys(obj)

            # ---- injection stats
            if args.injection_phrase:
                if any(args.injection_phrase in step for step in steps):
                    inj_idx = next(
                        i for i, step in enumerate(steps)
                        if args.injection_phrase in step
                    )

                    before_scores = step_scores[:inj_idx]
                    after_scores = step_scores[inj_idx + 1:-1]  # exclude injection step and final answer step

                    if len(before_scores) > 0 and len(after_scores) > 0:
                        before_avg = sum(before_scores) / len(before_scores)
                        after_avg = sum(after_scores) / len(after_scores)

                        avg_score_before_injection += before_avg
                        avg_score_after_injection += after_avg
                        injection_example_cnt += 1

                        if dataset_key is not None:
                            dataset_before_sum[dataset_key] += before_avg
                            dataset_after_sum[dataset_key] += after_avg
                            dataset_inj_cnt[dataset_key] += 1

            # ---- average score over all steps
            avg_step_score = sum(step_scores) / len(step_scores)

            overall_sum += avg_step_score
            overall_cnt += 1

            if dataset_key is not None:
                dataset_score_sum[dataset_key] += avg_step_score
                dataset_score_cnt[dataset_key] += 1

            if bbq_ambig_cat_key is not None:
                bbq_ambig_cat_score_sum[bbq_ambig_cat_key] += avg_step_score
                bbq_ambig_cat_score_cnt[bbq_ambig_cat_key] += 1

    # ---- print results
    if overall_cnt > 0:
        print(f"overall\t{overall_sum / overall_cnt:.6f}")

    # print("\n[Per dataset]")
    # for ds in sorted(dataset_score_sum.keys()):
    #     print(f"{ds}\t{dataset_score_sum[ds] / dataset_score_cnt[ds]:.6f}")
    
    print("\n[Per dataset - horizontal]")

    datasets = sorted(dataset_score_sum.keys())

    # header row
    print("dataset\t" + "\t".join(datasets))

    # values row
    values = [
        f"{dataset_score_sum[ds] / dataset_score_cnt[ds]:.6f}"
        for ds in datasets
    ]
    print("avg_score\t" + "\t".join(values))

    print("\n[BBQ ambig/disambig]")
    for ds in ["BBQ_ambig", "BBQ_disambig"]:
        if ds in dataset_score_sum:
            print(f"{ds}\t{dataset_score_sum[ds] / dataset_score_cnt[ds]:.6f}")
    
    print("\n[BBQ ambig categories - horizontal]")

    cats = sorted(bbq_ambig_cat_score_sum.keys())

    # print("category\t" + 
    print("\t".join(cats))

    values = [
        f"{bbq_ambig_cat_score_sum[c] / bbq_ambig_cat_score_cnt[c]:.6f}"
        for c in cats
    ]
    # print("avg_score\t" + 
    print("\t".join(values))

    # print("\n[BBQ ambig categories]")
    # for cat in sorted(bbq_ambig_cat_score_sum.keys()):
    #     print(f"{cat}\t{bbq_ambig_cat_score_sum[cat] / bbq_ambig_cat_score_cnt[cat]:.6f}")

    if args.injection_phrase and injection_example_cnt > 0:
        print("\n[Injection stats: overall]")
        print(f"before_injection\t{avg_score_before_injection / injection_example_cnt:.6f}")
        print(f"after_injection\t{avg_score_after_injection / injection_example_cnt:.6f}")

        print("\n[Injection stats: per dataset]")
        for ds in sorted(dataset_inj_cnt.keys()):
            print(
                f"{ds}\tbefore={dataset_before_sum[ds] / dataset_inj_cnt[ds]:.6f}\t"
                f"after={dataset_after_sum[ds] / dataset_inj_cnt[ds]:.6f}"
            )