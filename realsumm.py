import json
import numpy as np
from tqdm import tqdm
from utils import calculate_correlation
from datasets import load_dataset
from checkeval.checkeval import Checkeval
from dotenv import main
import os
from argparse import ArgumentParser

main.load_dotenv()

class Realsumm:
    def __init__(self, model="gpt-4-turbo"):
        self.model = model
        self.client = Checkeval(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model=self.model
        )

        dataset = load_dataset("manu/REALSumm")
        self.real_summ = dataset["train"].to_pandas()
    
    def generate(self, criterion="consistency", method="reference"):
        output_file = open(f"results/realsumm_{criterion}_{method}.json", "w")
        human_scores = []
        pred_scores = []
        all_results = []
        pbar = tqdm(total=len(self.real_summ))
        for i, row in self.real_summ.iterrows():
            row_dict = row.to_dict()
            reference = row_dict["source"]
            human_score = row_dict["litepyramid_recall"]
            human_scores.append(human_score)
            candidate = row_dict["model_summary"]
            checklist = None

            if method == "reference":
                res = self.client.reference_guided(criterion, reference, candidate, checklist)
            elif method == "candidate":
                res = self.client.candidate_guided(criterion, reference, candidate, checklist)
            elif method == "criterion":
                res = self.client.criterion_guided(criterion, reference, candidate, checklist)
                checklist = res['checklist']
            
            res = res["results"]

            row_dict["check_eval"] = {"consistency":res.score()}
            all_results.append(row_dict)
            pred_scores.append(res.score())
            if i > 0:
                eval_results = calculate_correlation(pred_scores, human_scores, {})
                pbar.set_postfix({"Pearson":eval_results['pearson'],"Spearman":eval_results['spearman'],"Kendall":eval_results['kendalltau']})
            json.dump(all_results, output_file)
            pbar.update(1)
    



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--criterion", type=str, default="consistency")
    parser.add_argument("--method", type=str, choices=["reference","candidate","criterion"], default="reference")
    parser.add_argument("--model", type=str, default="gpt-4-turbo")
    args = parser.parse_args()
    
    s = Realsumm(model=args.model)

    s.generate(args.criterion, args.method)

