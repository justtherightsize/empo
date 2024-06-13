import argparse
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import wandb
TEST = False
if TEST:
    wandb.init(mode="disabled")

from peft import LoraConfig, PeftModel
from alignment import DPOConfig
from trl import DPOTrainer
from src.emp_metrics.ed_load import get_ed_for_dpo, get_ed_for_kto

from huggingface_hub import login
from pathlib import Path
login(Path('/home/xsotolar/.huggingface/mistral').read_text().strip())

from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from numpy import percentile


def test_kto():
    tokenizer = AutoTokenizer.from_pretrained("alignment-handbook/zephyr-7b-sft-lora")

    # load datasets
    sys_msg = "You are a friendly assistant." 
    eval_dataset = get_ed_for_kto("test", tokenizer, sys_msg=sys_msg, tokenize=False,
                                  add_generation_prompt=True)

    import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    test_kto()

