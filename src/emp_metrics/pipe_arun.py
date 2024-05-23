import argparse
import json
from functools import partial

import wandb
from src.emp_metrics.pipe_sft import run_sft


def main(args: argparse.Namespace) -> None:
    with open("./src/emp_metrics/configs/" + args.sft_args + ".json") as f:
        sweep_config_sft = json.load(f)
    sweep_id = wandb.sweep(sweep_config_sft, project='erg')
    wandb.agent(sweep_id, run_sft, count=int(args.sft_sweeps))
    #
    # check sft location
    # load pt params
    # load pt data
    # pt_model = run pt
    #
    # gen&save pt preds
    # run metrics on saved pres or pt_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--results_dir", default="./results/", help="dir with saved models")
    parser.add_argument("-g", "--gpu", default="1", help="gpu string")

    parser.add_argument("-sa", "--sft_args", help="sft args")
    parser.add_argument("-ss", "--sft_sweeps", help="sft sweeps")

    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = parser.parse_args().gpu

    # parser.add_argument("-sn", "--new_sft_name", help="sft save name", default="")
    # parser.add_argument("-np", "--new_pt_name", help="pt save name")
    # parser.add_argument("-sa", "--sft_args", help="sft args")
    main(parser.parse_args())
