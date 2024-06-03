from typing import List

import numpy as np
import torch
import wandb

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import pprint
import pandas as pd
np.seterr(all='raise')


def calc_metrics(pth, model_id, metrics: List[str]):
    test_df = pd.read_csv(pth, sep="~")
    test_df = test_df[test_df['gens'].str.len() != 0]
    res = {}

    if "epitome" in metrics:     # Calculate EPITOME, DIFF-EPITOME metrics
        from src.emp_metrics.diff_epitome import EmpathyScorer, \
                to_epi_format, get_epitome_score, avg_epitome_score
        opt = {'no_cuda': False}
        device = 0
        opt['epitome_save_dir'] = "src/emp_metrics/checkpoints/epitome_checkpoint"
        epitome_empathy_scorer = EmpathyScorer(opt, batch_size=1,
                                               cuda_device=device)
        epi_in = to_epi_format(test_df["prevs"].to_list(),
                               test_df["gens"].to_list(),
                               test_df["gen_targets"])

        _, pred_IP_scores, pred_EX_scores, pred_ER_scores, \
            gt_IP_scores, gt_EX_scores, gt_ER_scores, \
            diff_IP_scores, diff_EX_scores, diff_ER_scores = \
            get_epitome_score(epi_in, epitome_empathy_scorer)

        epitome_report = avg_epitome_score(
                pred_IP_scores, pred_EX_scores, pred_ER_scores, gt_IP_scores,
                gt_EX_scores, gt_ER_scores, diff_IP_scores, diff_EX_scores,
                diff_ER_scores)
        res["epitome"] = {k: str(v) for k, v in epitome_report.items()}
        wandb.log({"diff_er": res["epitome"]["diff_ER"]})
        wandb.log({"diff_ex": res["epitome"]["diff_EX"]})
        wandb.log({"diff_ip": res["epitome"]["diff_IP"]})
        del epitome_empathy_scorer

    if "bertscore" in metrics:
        from evaluate import load
        bertscore = load("bertscore")

        preds = test_df["gens"].to_list()
        refs = test_df["gen_targets"].to_list()

        bs_results = bertscore.compute(predictions=preds,
                                       references=refs, lang="en")
        pbert = np.mean(bs_results["precision"])
        rbert = np.mean(bs_results["recall"])
        f1bert = np.mean(bs_results["f1"])
        wandb.log({"bertf1": f1bert})

        res["bertscore"] = {"p_bert": str(pbert), "r_bert": str(rbert),
                            "f1_bert": str(f1bert)}

    # Write results
    bert_pth = "results/epib_{}.txt".format(model_id)
    with open(bert_pth, 'w') as f:
        f.write(
            pprint.pformat(res, compact=True).replace("'", '"'))
        print(f"4.-----Saving df metrics (SFT) to: {bert_pth}--------")


def main(args: argparse.Namespace) -> None:
    model_id = args.adapter
    output_dir_base = args.base_dir
    metrics = args.metrics
    pth_to_csv = f"{output_dir_base}/preds_all_{model_id}.txt"
    calc_metrics(pth_to_csv, model_id, metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu", default="1", help="not implemented")

    parser.add_argument("-a", "--adapter", help="adapter name")
    parser.add_argument("-m", "--metrics", nargs="+",
                        help="one or more of: epitome, bertscore")
    parser.add_argument("-d", "--base_dir", default="./results",
                        help="base dir with saved models")

    main(parser.parse_args())

