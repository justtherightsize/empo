import json
import pprint

import pandas as pd
import wandb
from tqdm import tqdm

from src.emp_metrics.diff_epitome import EmpathyScorer, to_epi_format, get_epitome_score, \
    avg_epitome_score
from src.emp_metrics.ed_load import get_ed_chats, get_ed_for_generation

TEST = True
if TEST:
    wandb.init(mode="disabled")

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from transformers import BlenderbotSmallForConditionalGeneration
from transformers import AutoTokenizer

mname = "facebook/blenderbot_small-90M"
model = BlenderbotSmallForConditionalGeneration.from_pretrained(mname).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(mname)

sys_msg = ""
test_df = get_ed_for_generation("test", tokenizer, sys_msg=sys_msg, tokenize=False,
                                add_generation_prompt=False)

assert test_df["chat_templates"].str.contains("~").sum() == 0
assert test_df["prevs"].str.contains("~").sum() == 0

gens = []
inpt = []
batch_size = 96
for index, r in test_df.iterrows():
    if index > 0 and (index % (batch_size - 1) == 0 or index == len(test_df) - 1):
        inpt.append(r["chat_templates"])
        inputs = tokenizer(inpt, padding=True, truncation=True,
                           return_tensors="pt").to("cuda")
        reply_ids = model.generate(**inputs)
        gens.extend(tokenizer.batch_decode(reply_ids, skip_special_tokens=True))
        inpt = []
    else:
        inpt.append(r["chat_templates"])
test_df["gens"] = gens

#
# if TEST:
#     test_df = test_df.head()
#
assert test_df["gens"].str.contains("~").sum() == 0

pth = f"results/preds_{mname.split('/')[1]}.txt"
test_df.to_csv(pth, sep="~")
#
