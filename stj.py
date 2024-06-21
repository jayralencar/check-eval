import json
import numpy as np
from tqdm import tqdm
from utils import calculate_correlation
from datasets import load_dataset
from checkeval.checkeval import Checkeval
from dotenv import main
import os
from argparse import ArgumentParser
import pandas as pd

main.load_dotenv()

class STJ:
    def __init__(self, model="gpt-4-turbo"):
        self.model = model
        self.client = Checkeval(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model=self.model
        )
        self.dataset = self.load_data()
        print(self.dataset.info())
    
    def load_data(self, split="TEST"):
        df = pd.read_csv("./data/ground_truth.csv")
        return df
        # shuffle the data
        # df = df.sample(frac=1).reset_index(drop=True)
        return df[df["SPLIT"] == split]

    def generate(self, criterion="consistency", method="reference"):
        output_file = open(f"results/stj_h_{criterion}_{method}.json", "w")
        human_scores = []
        pred_scores = []
        all_results = []
        pbar = tqdm(total=len(self.dataset))
        for i, row in self.dataset.iterrows():
            row_dict = row.to_dict()
            reference = row_dict["TEXT1"]
            human_score = row_dict["EXPERT_SCORE"]
            human_scores.append(human_score)
            candidate = row_dict["TEXT2"]
            checklist = None

            if method == "reference":
                res = self.client.reference_guided(criterion, reference, candidate, checklist)
            elif method == "candidate":
                res = self.client.candidate_guided(criterion, reference, candidate, checklist)
            elif method == "criterion":
                res = self.client.criterion_guided(criterion, reference, candidate, checklist)
                checklist = res['checklist']
            
            row_dict["checklist"] = res["checklist"].to_markdown()
            res = res["results"]

            print(res)
            
            row_dict["check_eval"] = {"consistency":res.score()}
            all_results.append(row_dict)
            pred_scores.append(res.score())
            if len(pred_scores) > 1:
                # print(pred_scores, human_scores)
                eval_results = calculate_correlation(pred_scores, human_scores, {})
                pbar.set_postfix({"Pearson":eval_results['pearson'],"Spearman":eval_results['spearman'],"Kendall":eval_results['kendalltau']})
            json.dump(all_results, output_file)
            pbar.update(1)
    
    def f1(self, criterion):
        recall = json.load(open(f"results/stj_h_{criterion}_reference.json", "r"))
        precision = json.load(open(f"results/stj_h_{criterion}_candidate.json", "r"))

        # recall and precision items are in different orders
        # so we need to match them by score
        recall = sorted(recall, key=lambda x: x["EXPERT_SCORE"])
        precision = sorted(precision, key=lambda x: x["EXPERT_SCORE"])

        n_items = []
        human_scores = []
        pred_scores = []
        for i in range(len(recall)):
            recall_item = recall[i]['check_eval'][criterion]
            precision_item = precision[i]['check_eval'][criterion]

            human_score = recall[i]["EXPERT_SCORE"]
            human_scores.append(human_score)

            
            # recall[i]['check_eval'][f"{criterion}"] = precision_item
            if recall_item + precision_item == 0:
                recall[i]['check_eval'][f"{criterion}"] = 0
            else:
                recall[i]['check_eval'][f"{criterion}"] = 2 * (recall_item * precision_item) / (recall_item + precision_item)

            pred_scores.append(recall[i]['check_eval'][f"{criterion}"])

        eval_results = calculate_correlation(pred_scores, human_scores, {})
        # eval_results = self.evaluate(n_items)
        print(eval_results)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--criterion", type=str, default="consistency")
    parser.add_argument("--method", type=str, choices=["reference","candidate","criterion","f1"], default="reference")
    parser.add_argument("--model", type=str, default="gpt-4-turbo")
    args = parser.parse_args()
    
    s = STJ(model=args.model)
    print(args.method)
    if args.method == "f1":
        s.f1(args.criterion)
    else:
        s.generate(args.criterion, args.method)