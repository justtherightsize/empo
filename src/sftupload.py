import json
import pprint
import wandb
wandb.init(mode="disabled")
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from transformers import AutoTokenizer
from peft import PeftModel, PeftConfig

base_model_id = "alignment-handbook/zephyr-7b-sft-full"
model_id = "zephyr-7b-sft-full124_d270"
output_dir_base = "./results/"
output_dir = output_dir_base + model_id

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
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(output_dir)
model.resize_token_embeddings(len(tokenizer))
config = PeftConfig.from_pretrained(output_dir)
model = PeftModel.from_pretrained(model, output_dir)

repo = "justtherightsize/"
model.push_to_hub(model_id)
tokenizer.push_to_hub(model_id)

