# check-eval

This repository contains the code for the paper Check-Eval: A Checklist-based Approach for Evaluating Text Quality.

## Reproducing

Create a `.env` file with the `OPENAI_API_KEY`.

### STJ

Download the STJ dataset from [this link](https://osf.io/mct8s/). Save the `ground_truth.csv` inside the `data` folder.

Run the evaluation:

```
python3 -m stj --method=f1 --model=gpt-4-turbo --criterion=consistency

options:
  --criterion CRITERION
  --method {reference,candidate,criterion,f1,overall}
  --model MODEL (an openai valid model)
```

Find the results for this experiment inside the `results_legal_text_pt` folder

### Summeval

Download the data:
```
wget https://storage.googleapis.com/sfr-summarization-repo-research/model_annotations.aligned.jsonl -O data/model_annotations.aligned.jsonl
```

Run the experiments:

```
summeval.py [-h] [--criterion CRITERION] [--method {reference,candidate,criterion}] [--model MODEL]

options:
  -h, --help            show this help message and exit
  --criterion CRITERION
  --method {reference,candidate,criterion}
  --model MODEL
```
