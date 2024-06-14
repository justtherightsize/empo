""" Initialized as a copy of run_metrics_on_saved_df.py June 6 17:22"""
# TODO: implement saving results

from typing import List

import numpy as np
import torch
import os
# import wandb

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import pprint
import pandas as pd
import time
np.seterr(all='raise')




def calc_metrics(pth, model_id, metrics: List[str], run_pref: str = ""):
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
        # wandb.log({run_pref + "_" + "diff_er": res["epitome"]["diff_ER"]})
        # wandb.log({run_pref + "_" + "diff_ex": res["epitome"]["diff_EX"]})
        # wandb.log({run_pref + "_" + "diff_ip": res["epitome"]["diff_IP"]})

        print({run_pref + "_" + "diff_er": res["epitome"]["diff_ER"]})
        print({run_pref + "_" + "diff_ex": res["epitome"]["diff_EX"]})
        print({run_pref + "_" + "diff_ip": res["epitome"]["diff_IP"]})
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
        print({run_pref + "_" + "bertf1": f1bert})

        res["bertscore"] = {"p_bert": str(pbert), "r_bert": str(rbert),
                            "f1_bert": str(f1bert)}
        
    if "nidf" in metrics:
        import specificity
        col = "gens" if not args.human else "gen_targets"
        specificity.evaluate(args.ed_dir, test_df[col].values)

    if "vad" in metrics:
        import vad
        col = "gens" if not args.human else "gen_targets"
        vad.evaluate(test_df, col=col)

    if "diversity" in metrics:
        # TODO: may need to check output paths in ngrams.py and filepaths in tree.py
        import ngrams
        col = "gens" if not args.human else "gen_targets"

        ngrams.evaluate(pth, col=col)

        import gather_tree_stats
        gather_tree_stats.evaluate(pth, col=col)



    # Write results
    # bert_pth = "{}/epib_{}.txt".format(
    #         wandb.config.output_dir_base.rstrip("/"),
    #         model_id.replace(wandb.config.output_dir_base, ""))
    if args.save_protocol == 'og':
        # TODO reconfig for ondrej's preferences
        bert_pth = "{}/epib_{}.txt".format(
            args.output_dir,
            model_id)
        with open(bert_pth, 'w') as f:
            f.write(
                pprint.pformat(res, compact=True).replace("'", '"'))
            print(f"4.-----Saving df metrics (SFT) to: {bert_pth}--------")
    else:
        #todo
        out_path = os.path.join(args.output_dir, f"{model_id}_{int(time.time())}.csv")


def main() -> None:
    model_id = args.adapter
    metrics = args.metrics

    if args.save_protocol == 'og':
        output_dir_base = args.base_dir
        pth_to_csv = f"{output_dir_base}/preds_all_{model_id}.txt"
    else:
        pth_to_csv = args.data_file

    calc_metrics(pth_to_csv, model_id, metrics)


if __name__ == "__main__":
    i = os.path.dirname(os.path.realpath(__file__)).split('/').index('empathy-generation')
    p = '/'.join(os.path.dirname(os.path.realpath(__file__)).split('/')[:i+1])


    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu", default="1", help="not implemented")
    parser.add_argument('-sp', '--save_protocol', choices=['og', 'new'], default="new", help='og is from ondrej new is from allie')
    parser.add_argument('-hu', '--human', help='Run evaluation on human responses', action='store_true')
    parser.add_argument("-a", "--adapter", help="adapter name")
    parser.add_argument("-m", "--metrics", nargs="+",
                        help="choices: epitome, bertscore, nidf (specificity)", choices=["epitome", "bertscore", 'nidf', 'vad', 'diversity'])
    parser.add_argument("-d", "--base_dir", default="./results", help="base dir with saved models")
    parser.add_argument("-f", "--data_file", default=os.path.join(p,"data/generated_text/preds_x_zephyr-7b-sft-full122.full.txt"), help="file with generations")
    parser.add_argument("-o", "--output_dir", default=os.path.join(p,"data/results/empathy_eval_results"), help="directory to save the results")
    parser.add_argument("-ed", "--ed_dir", default=os.path.join(p,"data/empathy_datasets/empathetic_dialogues"), help="directory path of empathetic dialogues dataset")


    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(args)

    main()

