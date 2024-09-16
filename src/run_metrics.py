import json
import pprint

import pandas as pd
import wandb
from bert_score import BERTScorer

from src.emp_metrics.diff_epitome import EmpathyScorer, to_epi_format, get_epitome_score, \
    avg_epitome_score
from src.emp_metrics.ed_load import get_ed_chats, get_ed_for_generation

TEST = True
if TEST:
    wandb.init(mode="disabled")

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



test_df = pd.read_csv("results/preds_zephyr-qlora-empathy3.txt", sep="~")
# if TEST:
#     test_df = test_df.head(100)

# Metrics EPITOME, DIFF-EPITOME
# opt = {'no_cuda': False}
# device = 0
# opt['epitome_save_dir'] = "src/emp_metrics/checkpoints/epitome_checkpoint"
# epitome_empathy_scorer = EmpathyScorer(opt, batch_size=1, cuda_device=device)
# epi_in = to_epi_format(test_df["prevs"].to_list(), test_df["gens"].to_list(),
#                        test_df["gen_targets"])
#
# _, pred_IP_scores, pred_EX_scores, pred_ER_scores, gt_IP_scores, gt_EX_scores, gt_ER_scores, diff_IP_scores, diff_EX_scores, diff_ER_scores = get_epitome_score(
#     epi_in, epitome_empathy_scorer)
# report = avg_epitome_score(pred_IP_scores, pred_EX_scores, pred_ER_scores, gt_IP_scores, gt_EX_scores, gt_ER_scores, diff_IP_scores, diff_EX_scores, diff_ER_scores)
#
# with open('results/ED_test_zephyr_base.txt', 'w') as f:
#     f.write(pprint.pformat({k: str(v) for k, v in report.items()}, compact=True).replace("'", '"'))

# Metrics BERTScore
scorer = BERTScorer(model_type='bert-base-uncased')
P, R, F1 = scorer.score(test_df["gens"].to_list(), test_df["gen_targets"].to_list(), verbose=True)
print(f"BERTScore Precision: {P.mean():.4f}, Recall: {R.mean():.4f}, F1: {F1.mean():.4f}")
