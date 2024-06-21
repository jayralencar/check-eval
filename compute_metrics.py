import json
import pandas as pd

fp = "data/summeval_results_consistency.jsonl"

data = json.load(open(fp, "r"))

import numpy as np

df_data = []

for data in data:
    if 'check_eval' not in data:
        break
    # Initialize accumulators for expert and turker annotations
    expert_accumulator = {'coherence': [], 'consistency': [], 'fluency': [], 'relevance': []}
    turker_accumulator = {'coherence': [], 'consistency': [], 'fluency': [], 'relevance': []}

    c = {
        "id": data['id'],
        "score": data['check_eval']["consistency"],

    }

    # Expert annotations
    for i, ann in enumerate(data['expert_annotations']):
        keys = ann.keys()
        for key in keys:
            c["expert_{0}_{1}".format(i + 1, key)] = ann[key]
            expert_accumulator[key].append(ann[key])

    # Turker annotations
    for i, ann in enumerate(data['turker_annotations']):
        keys = ann.keys()
        for key in keys:
            c["turker_{0}_{1}".format(i + 1, key)] = ann[key]
            turker_accumulator[key].append(ann[key])

    # Calculate mean for each factor and for all factors for experts
    for key in expert_accumulator.keys():
        c["expert_mean_{}".format(key)] = np.mean(expert_accumulator[key])

    # Mean of all factors for experts
    c['expert_mean_all_factors'] = np.mean([value for sublist in expert_accumulator.values() for value in sublist])

    # Calculate mean for each factor and for all factors for turkers
    for key in turker_accumulator.keys():
        c["turker_mean_{}".format(key)] = np.mean(turker_accumulator[key])

    # Mean of all factors for turkers
    c['turker_mean_all_factors'] = np.mean([value for sublist in turker_accumulator.values() for value in sublist])

    # mean all factros all anotators
    c['mean_all_factors_all_annotators'] = np.mean([c['expert_mean_all_factors'], c['turker_mean_all_factors']])

    df_data.append(c)

df = pd.DataFrame(df_data)

corr = df[["score", 'expert_mean_coherence',
       'expert_mean_consistency', 'expert_mean_fluency',
       'expert_mean_relevance','expert_mean_all_factors',"mean_all_factors_all_annotators"]].corr(method="spearman")

print(corr)