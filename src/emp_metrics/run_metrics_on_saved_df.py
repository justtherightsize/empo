import wandb
TEST = True
if TEST:
    wandb.init(mode="disabled")

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import pprint
import pandas as pd
from src.emp_metrics.diff_epitome import EmpathyScorer, to_epi_format, get_epitome_score, \
    avg_epitome_score


def calc_metrics(pth, model_id):
    test_df = pd.read_csv(pth, sep="~")

    # Calculate EPITOME, DIFF-EPITOME metrics
    opt = {'no_cuda': False}
    device = 0
    opt['epitome_save_dir'] = "src/emp_metrics/checkpoints/epitome_checkpoint"
    epitome_empathy_scorer = EmpathyScorer(opt, batch_size=1, cuda_device=device)
    epi_in = to_epi_format(test_df["prevs"].to_list(), test_df["gens"].to_list(),
                           test_df["gen_targets"])

    _, pred_IP_scores, pred_EX_scores, pred_ER_scores, gt_IP_scores, gt_EX_scores, gt_ER_scores, \
    diff_IP_scores, diff_EX_scores, diff_ER_scores = get_epitome_score(
        epi_in, epitome_empathy_scorer)

    report = avg_epitome_score(pred_IP_scores, pred_EX_scores, pred_ER_scores, gt_IP_scores,
                               gt_EX_scores, gt_ER_scores, diff_IP_scores, diff_EX_scores,
                               diff_ER_scores)

    # Write results
    with open(f'results/ED_test_{model_id}.txt', 'w') as f:
        f.write(
            pprint.pformat({k: str(v) for k, v in report.items()}, compact=True).replace("'", '"'))


def main(args: argparse.Namespace) -> None:
    model_id = args.adapter
    output_dir_base = args.base_dir
    pth_to_csv = f"{output_dir_base}/preds_{model_id}.txt"
    calc_metrics(pth_to_csv, model_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu", default="1", help="not implemented")

    parser.add_argument("-a", "--adapter", help="adapter name")
    parser.add_argument("-d", "--base_dir", default="./results/", help="base dir with saved models")

    main(parser.parse_args())
