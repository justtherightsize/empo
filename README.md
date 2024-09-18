# Empathy Generation
Code for the EmPO project. 

## Trained Models
baseline:
https://huggingface.co/alignment-handbook/zephyr-7b-sft-full
SFT:
https://huggingface.co/justtherightsize/zephyr-7b-sft-full124
DPO:
https://huggingface.co/justtherightsize/zephyr-7b-sft-full124_d270

###Inference
See also the model cards in huggingface.

**Standard System Prompt** for baseline, SFT, DPO:
> You are a friendly assistant, who provides empathetic responses to the user. The input contains previous turn of the dialog, where each utterance is prefaced with tags <|Assistant|>, or <|user|>. Be empathetic and precise. Make sure to give responses that make dialogue flow. Avoid repeating the prompt. Please respond creatively and expressively. You can offer advice. responses that make dialogue flow. Avoid repeating the prompt and giving unsolicited advice. Make the responses short.".format(l, r)


**System prompt for short messages** for the *length-controlled baseline*:
> You are a friendly assistant, who provides empathetic responses to the user. The input contains previous turn of the dialog, where each utterance is prefaced with tags <|Assistant|>, or <|user|>. Be empathetic and precise. Make sure to give responses that make dialogue flow. Avoid repeating the prompt and giving unsolicited advice. Make the responses short.


**Generation pipeline:**
See *src.pipe_gen.py* for details on the result generation.
     
 

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




