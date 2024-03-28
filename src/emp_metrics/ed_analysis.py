import os
from dataclasses import dataclass, field
from typing import Optional


from transformers import AutoTokenizer, HfArgumentParser, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
import pandas as pd

from joan_utils import START_OF_TURN, END_OF_TURN, convert_to_dataset, format_chat

import wandb
wandb.init(mode="disabled")

@dataclass
class ScriptArguments:
    dataset_name: Optional[str] = field(
        # default="stingning/ultrachat",
        default="data/empathy_datasets/empathetic_dialogues",
        metadata={"help": "The preference dataset to use."},
    )
parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

train_dataset = convert_to_dataset(pd.read_csv(f"{script_args.dataset_name}/train.csv"))
val_dataset = convert_to_dataset(pd.read_csv(f"{script_args.dataset_name}/val.csv"))

# print("An example from the dataset")
# print("#" * 100)
# print(train_dataset["text"][0])
#
#
# train_dataset = train_dataset.map(lambda x: {"text": format_chat(x["chat"])}).remove_columns("chat")
# val_dataset = val_dataset.map(lambda x: {"text": format_chat(x["chat"])}).remove_columns("chat")
#
# print("An example from the dataset")
# print("#" * 100)
# print(train_dataset["text"][0])