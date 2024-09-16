import os

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from src.emp_metrics.ed_load import get_ed, get_ed_chats

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import wandb
wandb.init(mode="disabled")

model_id = "alignment-handbook/zephyr-7b-sft-lora"
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="right", padding=True)
tokenizer.add_special_tokens({'pad_token': "[PAD]"})
val_dataset = Dataset.from_pandas(
    get_ed_chats("validation", tokenizer, tokenize=False, add_generation_prompt=False))


model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
response_template = "\n<|assistant|>"
instruction_template = "\n<|user|>"
collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template,
                                           response_template=response_template,
                                           tokenizer=tokenizer,
                                           mlm=False)

