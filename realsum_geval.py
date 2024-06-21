from datasets import load_dataset
from geval import evaluate
from utils import calculate_correlation
from tqdm import tqdm
import json

dataset = load_dataset("manu/REALSumm")

df = dataset["train"].to_pandas()

pbar = tqdm(total=len(df))

human_scores = []
pred_scores = []
all_results = []

for i, row in df.iterrows():
    row_dict = row.to_dict()
    text = row_dict["source"]
    human_score = row_dict["litepyramid_recall"]
    human_scores.append(human_score)
    candidate_text = row_dict["model_summary"]

    score = evaluate(text, candidate_text, "consistency", model="gpt-4-turbo")
    
    pred_scores.append(score)

    if i > 0:
        eval_results = calculate_correlation(pred_scores, human_scores, {})
        pbar.set_postfix({"Pearson": eval_results['pearson'], "Spearman": eval_results['spearman'], "Kendall": eval_results['kendalltau']})
    
    pbar.update(1)

    row_dict["geval"] = {"consistency": score}

    all_results.append(row_dict)
    json.dump(all_results, open("data/realsumm_gevel.json", "w"))