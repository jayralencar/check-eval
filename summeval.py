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

class Summeval:
    def __init__(self, model="gpt-4-turbo"):
        self.model = model
        self.client = Checkeval(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model=self.model
        )

        dataset = load_dataset("abisee/cnn_dailymail", "3.0.0")

        self.cnn_dm = dataset["test"].to_pandas()

        self.summeval = open("data/model_annotations.aligned.jsonl", "r").readlines()

        self.data = self.load_data()

    def load_data(self):
        data = []

        for line in self.summeval:
            item = json.loads(line)
            expert_accumulator = {'coherence': [], 'consistency': [], 'fluency': [], 'relevance': []}
            for i, ann in enumerate(item['expert_annotations']):
                keys = ann.keys()
                for key in keys:
                    expert_accumulator[key].append(ann[key])
            item["scores"] = {}
            for key in expert_accumulator.keys():
                item["scores"][key] = np.mean(expert_accumulator[key])

            data.append(item)

        return data

    def evaluate(self,data, dimension="consistency"):
        pred_scores, human_scores = {}, {}

        for item in data:    

            doc_id = item["id"]
            if (doc_id not in pred_scores):
                pred_scores[doc_id] = []
                human_scores[doc_id] = []
            all_responses = item["check_eval"]
            # all_scores = [parse_output(x) for x in all_responses]
            score = all_responses[dimension]
            pred_scores[doc_id].append(score)
            human_scores[doc_id].append(item['scores'][dimension])

        results = {'pearson': 0, 'spearman': 0, 'kendalltau': 0}
        for doc_id in pred_scores:
            pred_scores_doc = pred_scores[doc_id]
            human_scores_doc = human_scores[doc_id]
            if (len(set(human_scores_doc)) <= 1) or (len(set(pred_scores_doc)) <= 1):
                continue

            results = calculate_correlation(pred_scores_doc, human_scores_doc, results)
            # print(results)
        results['pearson'] /= len(pred_scores)
        results['spearman'] /= len(pred_scores)
        results['kendalltau'] /= len(pred_scores)
        
        return results

    def generate(self, criterion="consistency", method="reference"):
        output_file = open(f"results/{criterion}_{method}.json", "w")

        checklists = {}
        pred_scores, human_scores = {}, {}
        n_items = []
        pbar = tqdm(self.data[:160])
        i = 0
        for item in pbar:
            id_ = item['id'].split("-")[-1]

            if (id_ not in pred_scores):
                pred_scores[id_] = []
                human_scores[id_] = []

            # if id_ not in checklists:
            df_ = self.cnn_dm[self.cnn_dm['id'] == id_]
            if len(df_) == 0:
                print("No match for", id_)
                continue
            
            reference = df_['article'].values[0]
            candidate = item['decoded']
            checklist = None
            if id_ in checklists and method == "reference":
                checklist = checklists[id_]

            if method == "reference":
                res = self.client.reference_guided(criterion, reference, candidate, checklist)
            else:
                res = self.client.candidate_guided(criterion, reference, candidate, checklist)

            if id_ not in checklists and method == "reference":
                checklists[id_] = res["checklist"]
            
            pred_scores[id_].append(res["results"].score())

            item['check_eval'] = {"consistency":res["results"].score()}

            del item['expert_annotations']
            del item['turker_annotations']
            del item['references']

            n_items.append(item)

            json.dump(n_items, output_file)

            eval_results = self.evaluate(n_items)

            pbar.set_postfix({"Pearson":eval_results['pearson'],"Spearman":eval_results['spearman'],"Kendall":eval_results['kendalltau']})
            i += 1

            pbar.update(1)
    
    def reference_guided(self, criterion):
        self.generate(criterion, "reference")
    
    def candidate_guided(self, criterion):
        self.generate(criterion, "candidate")
    
    def f1(self, criterion):
        recall = json.load(open(f"results/{criterion}_reference.json", "r"))
        precision = json.load(open(f"results/{criterion}_candidate.json", "r"))
        n_items = []
        for i in range(len(recall)):

            recall_item = recall[i]['check_eval'][criterion]
            precision_item = precision[i]['check_eval'][criterion]

            recall[i]['check_eval'][f"{criterion}"] = 2 * (recall_item * precision_item) / (recall_item + precision_item)

            n_items.append(recall[i])
        
        eval_results = self.evaluate(n_items)
        print(eval_results)


    def criterion_guided(self, criterion):
        output_file = open(f"results/{criterion}_criterion.json", "w")

        checklist = None
        pred_scores, human_scores = {}, {}
        n_items = []
        pbar = tqdm(self.data[:160])
        i = 0
        for item in pbar:
            id_ = item['id'].split("-")[-1]

            if (id_ not in pred_scores):
                pred_scores[id_] = []
                human_scores[id_] = []

            # if id_ not in checklists:
            df_ = self.cnn_dm[self.cnn_dm['id'] == id_]
            if len(df_) == 0:
                print("No match for", id_)
                continue
            
            reference = df_['article'].values[0]
            candidate = item['decoded']
            checklist = """1. Does the summary accurately reflect the main points of the source text?
2. Are all the facts mentioned in the summary present in the source text?
3. Does the summary avoid introducing new information not found in the source text?
4. Does the summary maintain the same tone and perspective as the source text?
5. Does the summary accurately represent the sequence of events or arguments in the source text?
6. Does the summary avoid contradicting any information from the source text?
7. Does the summary avoid exaggerating or downplaying any information from the source text?
8. Does the summary accurately represent the conclusions or findings of the source text?
9. Does the summary avoid misinterpreting or misrepresenting any information from the source text?
10. Does the summary avoid making assumptions or inferences not supported by the source text?
11. Does the summary avoid using language or terminology not used in the source text?
12. Does the summary avoid omitting any crucial information from the source text?
13. Does the summary avoid including any personal opinions or biases not present in the source text?
14. Does the summary accurately represent the purpose or intent of the source text?
15. Does the summary avoid any form of factual hallucination?"""
            res = self.client.criterion_guided(criterion, reference, candidate, checklist)

            # if checklist is None:
            #     checklist = res["checklist"]
            
            pred_scores[id_].append(res["results"].score())

            item['check_eval'] = {criterion:res["results"].score()}
            del item['expert_annotations']
            del item['turker_annotations']
            del item['references']

            n_items.append(item)

            json.dump(n_items, output_file)

            eval_results = self.evaluate(n_items)

            pbar.set_postfix({"Pearson":eval_results['pearson'],"Spearman":eval_results['spearman'],"Kendall":eval_results['kendalltau']})

            i += 1

            pbar.update(1)
                             
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--criterion", type=str, default="consistency")
    parser.add_argument("--method", type=str, choices=["reference","candidate","criterion"], default="reference")
    parser.add_argument("--model", type=str, default="gpt-4-turbo")
    args = parser.parse_args()
    
    s = Summeval()
    
    if args.method == "reference":
        s.reference_guided(args.criterion)
    elif args.method == "candidate":
        s.candidate_guided(args.criterion)
    elif args.method == "criterion":
        s.criterion_guided(args.criterion)