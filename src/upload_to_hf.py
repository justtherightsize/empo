import json
import pprint

import wandb

from src.emp_metrics.diff_epitome import EmpathyScorer, to_epi_format, get_epitome_score, \
    avg_epitome_score
from src.emp_metrics.ed_load import get_ed_chats, get_ed_for_generation

TEST = True
if TEST:
    wandb.init(mode="disabled")

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, pipeline
from transformers import AutoTokenizer
from peft import PeftModel, PeftConfig

BASE = True
base_model_id = "alignment-handbook/zephyr-7b-sft-lora"
model_id = "zephyr-qlora-empathy3"
output_dir_base = "./results/"
output_dir = output_dir_base + model_id

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

inference_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=quantization_config,
    trust_remote_code=True
)
inference_model.config.use_cache = False

if BASE:
    inference_tokenizer = AutoTokenizer.from_pretrained(base_model_id, padding_side="right", padding=True,
                                              trust_remote_code=True,
                                              clean_up_tokenization_spaces=True)
    inference_tokenizer.add_special_tokens({'pad_token': "[PAD]"})
    inference_model.resize_token_embeddings(len(inference_tokenizer))
else:
    inference_tokenizer = AutoTokenizer.from_pretrained(output_dir)
    inference_model.resize_token_embeddings(len(inference_tokenizer))
    config = PeftConfig.from_pretrained(output_dir)
    inference_model = PeftModel.from_pretrained(inference_model, output_dir)

pipe = pipeline("text-generation",
                model=inference_model,
                tokenizer=inference_tokenizer,
                max_new_tokens=200)

sys_msg = "You are a friendly assistant, who provides empathetic responses to the user. The input contains previous turn of the dialog, where the each utterance is prefaced with tags <|user|>, or <|assistant|>. Be empathetic and precise. Make sure to give responses that make dialogue flow. Avoid repeating the prompt."
test_df = get_ed_for_generation("test", inference_tokenizer, sys_msg=sys_msg, tokenize=False,
                                add_generation_prompt=True)

# if TEST:
#     test_df = test_df.head(10)

gens = []
for index, r in test_df.iterrows():
    gens.append(pipe(r["chat_templates"], return_full_text=False)[0]['generated_text'])
test_df["gens"] = gens

# Metrics EPITOME, DIFF-EPITOME
opt = {'no_cuda': False}
device = 0
opt['epitome_save_dir'] = "src/emp_metrics/checkpoints/epitome_checkpoint"
epitome_empathy_scorer = EmpathyScorer(opt, batch_size=1, cuda_device=device)
epi_in = to_epi_format(test_df["prevs"].to_list(), test_df["gens"].to_list(),
                       test_df["gen_targets"])

_, pred_IP_scores, pred_EX_scores, pred_ER_scores, gt_IP_scores, gt_EX_scores, gt_ER_scores, diff_IP_scores, diff_EX_scores, diff_ER_scores = get_epitome_score(
    epi_in, epitome_empathy_scorer)
report = avg_epitome_score(pred_IP_scores, pred_EX_scores, pred_ER_scores, gt_IP_scores, gt_EX_scores, gt_ER_scores, diff_IP_scores, diff_EX_scores, diff_ER_scores)

with open('results/ED_test_zephyr_base.txt', 'w') as f:
    f.write(pprint.pformat({k: str(v) for k, v in report.items()}, compact=True).replace("'", '"'))
