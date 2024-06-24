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
        df = pd.read_csv("./data/sample.csv")
        return df
        # shuffle the data
        # df = df.sample(frac=1).reset_index(drop=True)
        return df[df["SPLIT"] == split]

    def generate(self, criterion="consistency", method="reference"):
        
        human_scores = []
        pred_scores = []
        all_results = []
        pbar = tqdm(total=len(self.dataset))
        for i, row in self.dataset.iterrows():
            output_file = open(f"results/stj_{criterion}_{method}.json", "w")
            row_dict = row.to_dict()
            reference = row_dict["sentence_A"]
            human_score = row_dict["score"]
            human_scores.append(human_score)
            candidate = row_dict["sentence_B"]
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

            # print(res)
            
            row_dict["check_eval"] = {f"{criterion}":res.score()}
            all_results.append(row_dict)
            pred_scores.append(res.score())
            if len(pred_scores) > 1:
                # print(pred_scores, human_scores)
                eval_results = calculate_correlation(pred_scores, human_scores, {})
                pbar.set_postfix({"Pearson":eval_results['pearson'],"Spearman":eval_results['spearman'],"Kendall":eval_results['kendalltau']})
            json.dump(all_results, output_file)
            
            pbar.update(1)
        print(eval_results) 
    
    def f1(self, criterion):
        recall = json.load(open(f"results/stj_{criterion}_reference.json", "r"))
        precision = json.load(open(f"results/stj_{criterion}_candidate.json", "r"))

        # recall and precision items are in different orders
        # so we need to match them by score
        recall = sorted(recall, key=lambda x: x["score"])
        precision = sorted(precision, key=lambda x: x["score"])

        n_items = []
        human_scores = []
        pred_scores = []
        for i in range(len(recall)):
            recall_item = recall[i]['check_eval'][criterion]
            precision_item = precision[i]['check_eval'][criterion]

            human_score = recall[i]["score"]
            human_scores.append(human_score)

            
            # recall[i]['check_eval'][f"{criterion}"] = recall_item
            if recall_item + precision_item == 0:
                recall[i]['check_eval'][f"{criterion}"] = 0
            else:
                recall[i]['check_eval'][f"{criterion}"] = 2 * (recall_item * precision_item) / (recall_item + precision_item)

            pred_scores.append(recall[i]['check_eval'][f"{criterion}"])

        eval_results = calculate_correlation(pred_scores, human_scores, {})
        # eval_results = self.evaluate(n_items)
        print(eval_results)
    
    def load_result(self, criterion, method):
        a = json.load(open(f"results/stj_{criterion}_{method}.json", "r"))
        a = sorted(a, key=lambda x: x["score"])
        return a

    def overall(self, method):

        if method == "f1":
            con_r = self.load_result("consistency", "reference")
            coh_r = self.load_result("coherence", "reference")
            rel_r = self.load_result("relevance", "reference")
            flu_r = self.load_result("fluency", "reference")


            con_p = self.load_result("consistency", "candidate")
            coh_p = self.load_result("coherence", "candidate")
            rel_p = self.load_result("relevance", "candidate")
            flu_p = self.load_result("fluency", "candidate")

            con = []
            coh = []
            rel = []
            flu = []

            #calculate f1
            for i, item in enumerate(con_r):
                con_item_r = con_r[i]['check_eval']["consistency"]
                coh_item_r = coh_r[i]['check_eval']["coherence"]
                rel_item_r = rel_r[i]['check_eval']["relevance"]
                flu_item_r = flu_r[i]['check_eval']["fluency"]

                con_item_p = con_p[i]['check_eval']["consistency"]
                coh_item_p = coh_p[i]['check_eval']["coherence"]
                rel_item_p = rel_p[i]['check_eval']["relevance"]
                flu_item_p = flu_p[i]['check_eval']["fluency"]
# 
                # human_score = con_r[i]["score"]
                # human_scores.append(human_score)

                # pred_score = (con_item_r + coh_item_r + rel_item_r + flu_item_r) / 4
                # pred_scores.append(pred_score)

                if con_item_r + con_item_p == 0:
                    con_item = 0
                else:
                    con_item = 2 * (con_item_r * con_item_p) / (con_item_r + con_item_p)

                if coh_item_r + coh_item_p == 0:
                    coh_item = 0
                else:
                    coh_item = 2 * (coh_item_r * coh_item_p) / (coh_item_r + coh_item_p)

                if rel_item_r + rel_item_p == 0:
                    rel_item = 0
                else:
                    rel_item = 2 * (rel_item_r * rel_item_p) / (rel_item_r + rel_item_p)

                if flu_item_r + flu_item_p == 0:
                    flu_item = 0
                else:
                    flu_item = 2 * (flu_item_r * flu_item_p) / (flu_item_r + flu_item_p)

                con_r[i]['check_eval']["consistency"] = con_item
                con.append(con_r[i])
                coh_r[i]['check_eval']["coherence"] = coh_item
                coh.append(coh_r[i])
                rel_r[i]['check_eval']["relevance"] = rel_item
                rel.append(rel_r[i])
                flu_r[i]['check_eval']["fluency"] = flu_item
                flu.append(flu_r[i])

        else:
            con = self.load_result("consistency", method)
            coh = self.load_result("coherence", method)
            rel = self.load_result("relevance", method)
            flu = self.load_result("fluency", method)


        n_items = []
        human_scores = []
        pred_scores = []

        for i in range(len(con)):
            con_item = con[i]['check_eval']["consistency"]
            coh_item = coh[i]['check_eval']["coherence"]
            rel_item = rel[i]['check_eval']["relevance"]
            flu_item = flu[i]['check_eval']["fluency"]

            human_score = con[i]["score"]
            human_scores.append(human_score)

            pred_score = (con_item + coh_item + rel_item + flu_item) / 4
            pred_scores.append(pred_score)

        eval_results = calculate_correlation(pred_scores, human_scores, {})
        # eval_results = self.evaluate(n_items)
        print(eval_results)




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--criterion", type=str, default="consistency")
    parser.add_argument("--method", type=str, choices=["reference","candidate","criterion","f1", "overall"], default="reference")
    parser.add_argument("--model", type=str, default="gpt-4-turbo")
    args = parser.parse_args()
    
    s = STJ(model=args.model)
    print(args.method)
    if args.method == "f1":
        s.f1(args.criterion)
    elif args.method == "overall":
        s.overall(args.criterion)
    else:
        s.generate(args.criterion, args.method)