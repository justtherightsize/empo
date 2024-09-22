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
You can generate predictions using the *src.pipe_gen.py* script. See also the model cards in huggingface.
```bash
# -a: PEFT LoRA adapter to be used atop the base model (default: alignment-handbook/zephyr-7b-sft-full)
# -l: use local models or download from HF hub
# -p: save file name prefix
# -k: path to text file with the key to your huggingface account
# -m: max_tokens 
python ./src/pipe_gen.py -a zephyr-7b-sft-full124 -l -p test1 -k <pth_to_key> -m 1000
```

**Standard System Prompt** for baseline, SFT, DPO:
> You are a friendly assistant, who provides empathetic responses to the user. The input contains previous turn of the dialog, where each utterance is prefaced with tags <|Assistant|>, or <|User|>. Be empathetic and precise. Make sure to give responses that make dialogue flow. Avoid repeating the prompt. Please respond creatively and expressively. You can offer advice. responses that make dialogue flow. Avoid repeating the prompt and giving unsolicited advice. Make the responses short.".format(l, r)


**System prompt for short messages** for the *length-controlled baseline*:
> You are a friendly assistant, who provides empathetic responses to the user. The input contains previous turn of the dialog, where each utterance is prefaced with tags <|Assistant|>, or <|User|>. Be empathetic and precise. Make sure to give responses that make dialogue flow. Avoid repeating the prompt and giving unsolicited advice. Make the responses short.

**Generation pipeline:**
See *src.pipe_gen.py* for details on the result generation.
 
## Training
The training pipeline involves wandb.ai hyperparameter sweeps. You have to have the wandb package installed and be logged in to train. 

The training starts en *src.pipe_arun.py*, which load the configuration from the *src.configs/* folder, such as dpo27.json. Then the script runs *src.pipe_sft.py* to train either SFT or DPO depending on the config file parameters.
```bash
# -sa: config name
# -ss: number of tries in a sweep 
python ./src/pipe_arun.py -sa dpo32 -ss 4
```

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

### Human evaluation
* The data from the evaluation itself are present in data/human\_valuation/he\_data.csv
* The evaluation evaluation scripts for effect sizes and final score in
notebooks/he\_score\_effsize.ipynb
The data for the human evaluation were preprocessed into batches, which were then served to the
annotators.
* Preprocessing scripts present in src/human\_evaluation/he\_data\_prep.py and data in
data/generated\_text

