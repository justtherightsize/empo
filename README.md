# Empathy Generation
Code for the EmPO project. 

## Trained Models
baseline:
https://huggingface.co/alignment-handbook/zephyr-7b-sft-full
SFT:
https://huggingface.co/justtherightsize/zephyr-7b-sft-full124
DPO:
https://huggingface.co/justtherightsize/zephyr-7b-sft-full124_d270

### Inference
See also the model cards in huggingface.

**Standard System Prompt** for baseline, SFT, DPO:
> You are a friendly assistant, who provides empathetic responses to the user. The input contains previous turn of the dialog, where each utterance is prefaced with tags <|Assistant|>, or <|User|>. Be empathetic and precise. Make sure to give responses that make dialogue flow. Avoid repeating the prompt. Please respond creatively and expressively. You can offer advice. responses that make dialogue flow. Avoid repeating the prompt and giving unsolicited advice. Make the responses short.".format(l, r)


**System prompt for short messages** for the *length-controlled baseline*:
> You are a friendly assistant, who provides empathetic responses to the user. The input contains previous turn of the dialog, where each utterance is prefaced with tags <|Assistant|>, or <|User|>. Be empathetic and precise. Make sure to give responses that make dialogue flow. Avoid repeating the prompt and giving unsolicited advice. Make the responses short.

**Generation pipeline:**
See *src.pipe_gen.py* for details on the result generation.
 
## Training
See *src.pipe_sft.py* for details on SFT training and *src.pipe_dpo.py* for DPO training.

## Evaluation
Genrate predictions with *src.pipe_gen.py* then run the following metrics on the predictions. 

### diffEpitome and BERTscore
Run *src.run_metrics_on_saved_df.py* on a csv dataframe (preds_...) with predictions.
```bash
python ./src/run_metrics_on_saved_df.py -s -n preds_dlr1e6m1000_zephyr-7b-sft-full124_d270.txt -m bertscore epitome
```

### MMLU and Open LLM leaderboard
Get the lm_eval package fork from Huggingface and run task *mmlu* or *leaderboard*. For instalation, follow https://huggingface.co/docs/leaderboards/en/open_llm_leaderboard/about#reproducibility
```bash
# This uses local folder for the models. Either download the models to the current folder 
# or prepend justtherightsize/ to the models name to download it automatically.
lm_eval --model hf --model_args pretrained=alignment-handbook/zephyr-7b-sft-full,dtype=bfloat16,use_flash_attention_2=True,trust_remote_code=True,peft=zephyr-7b-sft-full124_d270,tokenizer=zephyr-7b-sft-full124_d270 --tasks=leaderboard --batch_size=auto --output_path=leader_zephyr-7b-sft-full124_d270.txt
```

### Lee's empathy metrics
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




