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
See *src.pipe_gen.py* for details on reproducing the results on the EmpatheticDialogues dataset. A single example generation can be run using *src.reproduce_gen.py*. Note that you need to be logged in with a HuggingFace account and agree to the license of the base model (https://huggingface.co/alignment-handbook/zephyr-7b-sft-full)
```bash
export PYTHONPATH="."
python ./src/reproduce_gen.py
```
Or you can run this code:
```python
from peft import PeftModel
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from huggingface_hub import login

# HF login: you have to be logged in and agree to the license of the base
# model: https://huggingface.co/alignment-handbook/zephyr-7b-sft-full
hf_key = "Your key here"
login(hf_key)

# Load tokenizer either from remote
adapter_id = "justtherightsize/zephyr-7b-sft-full124_d270"
base_model_id = "alignment-handbook/zephyr-7b-sft-full"
tokenizer = AutoTokenizer.from_pretrained(adapter_id)

# Prepare dialog and convert to chat template
sys_msg = "You are a friendly assistant, who provides empathetic responses to the user. " \
            "The input contains previous turn of the dialog, where each utterance is prefaced " \
            "with tags <|user|>, or <|assistant|>. Be empathetic and precise. " \
            "Make sure to give responses that make dialogue flow. Avoid repeating the prompt. " \
            "Please respond creatively and expressively to make the responses longer. You can offer advice."

dialog = ["Yeah about 10 years ago I had a horrifying experience. It was 100% their fault but they hit the water barrels and survived. They had no injuries but they almost ran me off the road.", 
        "Did you suffer any injuries?", 
        "No I wasn't hit. It turned out they were drunk. I felt guilty but realized it was his fault."]

dwroles = [{"role": "system", "content": sys_msg}]
for j in range(len(dialog)):
    dwroles.append(
        {"role": "user", "content": dialog[j]} if j % 2 == 0 else
        {"role": "assistant", "content": dialog[j]})
template = tokenizer.apply_chat_template(dwroles, tokenize=False, add_generation_prompt=True)

# Load the big model first & resize embeds, load PEFT model
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=quantization_config,
    trust_remote_code=True
)
model.resize_token_embeddings(len(tokenizer))
model.config.use_cache = False
model = PeftModel.from_pretrained(model, adapter_id)

# Instantiate generation pipeline
pipe_gen = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Generate the response
out = pipe_gen(template, return_full_text=False, max_new_tokens=500)[0]['generated_text']
print(out)
```




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

