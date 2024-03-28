import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from accelerate import PartialState
# settings for telamon
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch

from transformers import AutoTokenizer, HfArgumentParser, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, \
    AutoConfig
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import pandas as pd

from joan_utils import START_OF_TURN, END_OF_TURN, convert_to_dataset, format_chat



@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """
    per_device_train_batch_size: Optional[int] = field(default=4)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=4)
    learning_rate: Optional[float] = field(default=2e-4)
    max_grad_norm: Optional[float] = field(default=0.3)
    weight_decay: Optional[int] = field(default=0.001)
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=8)
    max_seq_length: Optional[int] = field(default=2048)
    model_name: Optional[str] = field(
        # default="TinyLlama/TinyLlama-1.1B-Chat-v0.1",
        default="alignment-handbook/zephyr-7b-sft-lora",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        }
    )
    dataset_name: Optional[str] = field(
        # default="stingning/ultrachat",
        default="data/empathy_datasets/empathetic_dialogues",
        metadata={"help": "The preference dataset to use."},
    )
    subset: Optional[bool] = field(
        default=True,
        metadata={"help": "Use subset of data"},
    )
    fp16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables fp16 training."},
    )
    bf16: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables bf16 training."},
    )
    packing: Optional[bool] = field(
        default=True,
        metadata={"help": "Use packing dataset creating."},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )
    use_flash_attention_2: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables Flash Attention 2."},
    )
    optim: Optional[str] = field(
        default="paged_adamw_32bit",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: str = field(
        # default="constant",
        default="cosine",
        metadata={"help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"},
    )
    max_steps: int = field(default=-1, metadata={"help": "How many optimizer update steps to take"})
    warmup_ratio: float = field(default=0, metadata={"help": "Fraction of steps to do a warmup for"})
    save_steps: int = field(default=0.1, metadata={"help": "Save checkpoint every X updates steps."})
    logging_steps: int = field(default=0.1, metadata={"help": "Log every X updates steps."})
    output_dir: str = field(
        default="./results",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

# Load the GG model - this is the local one, update it to the one on the Hub
model_id = script_args.model_name

output_dir = "./results/zephyr-qlora-empathy"
checkpoint_id = "checkpoint-108"
tokenizer = AutoTokenizer.from_pretrained(output_dir)
# tokenizer.pad_token_id = tokenizer.eos_token_id
# tokenizer.add_special_tokens({'additional_special_tokens': [START_OF_TURN, END_OF_TURN]})


def fix_model_folder_with_incorrect_vocab_size(model_folder: Path, pad_vocab_size_to_multiple_of: int = 64):
    """If you got bitten by the bug https://github.com/huggingface/transformers/issues/25729, and your model got saved with an incorrect vocab size, this will fix it."""
    config_file = model_folder / "config.json"
    config = AutoConfig.from_pretrained(config_file)
    vocab_size: int = config.vocab_size
    # is vocab_size an int, and is it bigger than 0?
    assert isinstance(vocab_size, int) and vocab_size > 0

    if vocab_size % pad_vocab_size_to_multiple_of != 0:
        print(f"vocab_size ({vocab_size}) is not a multiple of {pad_vocab_size_to_multiple_of}. Fixing it...")
        # fix the vocab_size:
        config.vocab_size = vocab_size + (pad_vocab_size_to_multiple_of - vocab_size % pad_vocab_size_to_multiple_of)
        # save the config file back to the checkpoint folder:
        config.save_pretrained(model_folder)
        print(f"Fixed the vocab_size to {config.vocab_size}")


# model = AutoModelForCausalLM.from_pretrained(Path(output_dir, checkpoint_id), load_in_4bit=True, device_map="auto")
model = AutoModelForCausalLM.from_pretrained(output_dir, load_in_4bit=True, device_map="auto")
# model = PeftModel.from_pretrained(output_dir, checkpoint_id=checkpoint_id, load_in_4bit=True, device_map="auto")
