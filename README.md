# Empathy Generation
Code for the EmPO project. 

## Trained Models
TODO

## Evaluation
TODO

## Training
TODO

## Lee's empathy metrics
**running on file -f**
- Results are saved in data/results/empathy_eval_results by default, with path "data/results/empathy_eval_results/[filename]_[metric].txt"
```bash
#vad metrics
python src/emp_metrics/run_empathy_eval.py -f data/empo/preds_dlr1e6_zephyr-7b-sft-full122_d211.txt -m vad 
# specificity metrics
python src/emp_metrics/run_empathy_eval.py -f data/empo/preds_dlr1e6_zephyr-7b-sft-full122_d211.txt -m nidf

```

**running on human data**
* Pass in the flag -hu, and -f followed by the path of any of the files containing the human responses in the column "gen_targets"
* Results are saved at "data/results/empathy_eval_results/human_[metric].txt"




