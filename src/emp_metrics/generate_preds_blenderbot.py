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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import BlenderbotSmallForConditionalGeneration
from transformers import AutoTokenizer

mname = "facebook/blenderbot_small-90M"
model = BlenderbotSmallForConditionalGeneration.from_pretrained(mname)
tokenizer = AutoTokenizer.from_pretrained(mname)

sys_msg = "You are a friendly assistant, who provides empathetic responses to the user. The input contains previous turn of the dialog, where the each utterance is prefaced with tags <|user|>, or <|assistant|>. Be empathetic and precise. Make sure to give responses that make dialogue flow. Avoid repeating the prompt."
test_df = get_ed_for_generation("test", tokenizer, sys_msg=sys_msg, tokenize=False,
                                add_generation_prompt=False)

test_df = test_df.head(2)

inputs = tokenizer(test_df["chat_templates"].to_list(), padding=True, truncation=True, return_tensors="pt")
reply_ids = model.generate(**inputs)
print(tokenizer.batch_decode(reply_ids, skip_special_tokens=True))


#
#
# gens = []
# for index, r in tqdm(test_df.iterrows()):
#     gens.append(pipe(r["chat_templates"], return_full_text=False)[0]['generated_text'])
# test_df["gens"] = gens
#
# if TEST:
#     test_df = test_df.head()
#
# assert test_df["gens"].str.contains("~").sum() == 0
# pth = f"results/preds_base_{model_id}.txt" if BASE else f"results/preds_{model_id}.txt"
# test_df.to_csv(pth, sep="~")
#
