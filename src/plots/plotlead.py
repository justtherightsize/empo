from typing import List, Dict, Tuple
from src.plots.leaderld import load_openllm_leaderboard
import numpy as np
import pprint
import matplotlib.pyplot as plt
import random


def avg_std(lists: Dict[str, List[float]]) -> Dict[int, Dict]:
    """ Return avg and std """
    def nms(vals: List[float]) -> Tuple[float, float]:
        ar = np.array(vals)
        return (np.mean(ar), np.std(ar))

    ret = {k: nms(v) for k, v in lists.items()}
    return ret


def get_leaderboard_data(is_test=False, prnt=False):
    results_dir = "results"
    pths = {
            16: ["leader_zephyr-7b-sft-full153/results_2024-08-21T22-24-51.248202.json",
                 "leader_zephyr-7b-sft-full123.txt/results_2024-08-23T21-16-49.814276.json"],
            64: ["leader_zephyr-7b-sft-full124.txt/results_2024-08-23T21-22-42.604715.json",
                 "leader_zephyr-7b-sft-full154/results_2024-08-21T22-31-09.961957.json"],
            256: ["leader_zephyr-7b-sft-full122/results_2024-08-21T02-30-35.042359.json",
                  "leader_zephyr-7b-sft-full10.txt/results_2024-08-23T21-22-23.550919.json"],
            1024: ["leader_zephyr-7b-sft-full155/results_2024-08-21T22-18-33.166097.json",
                   "leader_zephyr-7b-sft-full120/results_2024-08-25T02-45-37.133867.json"],
            2048: ["leader_zephyr-7b-sft-full156/results_2024-08-21T22-16-24.878598.json",
                   "/leader_zephyr-7b-sft-full121/results_2024-08-23T20-45-36.882140.json"]}

    if is_test:
        results_dir = "."
        pths = {100: ["src/plots/test1.json",
                      "src/plots/test2.json"]}

    # Pivot the records: create a list of values for each metric. Repeat for each key.
    avgs = {}
    for k, v in pths.items():
        ld = [load_openllm_leaderboard(
            results_dir + "/" + p, printout=False) for p in v]
        piv = {}
        for kl in ld[0].keys():
            piv[kl] = [x[kl] for x in ld]
        # Send the values lists to calculate the avg and std
        avgs[k] = avg_std(piv)

    if prnt:
        pprint.pp(avgs)
    return avgs


def gen_appx_plot(save_pth: str, avgs: Dict) -> None:
    alpha = [16, 64, 256, 1024, 2048]
    log_alpha = np.log10(alpha)
    fig, ax1 = plt.subplots(figsize=(8.3, 5))
    ax1.set_ylim(0.0, 0.60)  # Set the y-axis limits inverted

    # ax1.fill_between(log_alpha, mmlu_bot, mmlu_top, facecolor='#FF6347', alpha=0.2)
    # ax1.axhline(y=0.588, color='#5F9EA0', linestyle='--', linewidth=4,
    #             label='Zephyr-7B baseline')
    
    metrics = list(avgs.values())[0].keys()
    for m in metrics:
        lead_values = [avgs[a][m][0] for a in alpha]
        ran_color = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
        ax1.plot(log_alpha, lead_values, color=ran_color, marker='+', linewidth=2, markersize=12, mew=2,
                 linestyle='-.', label=m)
 
    # Adding labels and title
    ax1.set_xlabel(r'$\alpha$ (log Scale)', fontsize=16)
    ax1.set_ylabel('MMLU', fontsize=16)
    # Customizing the ticks
    ax1.set_xticks(log_alpha, labels=alpha, fontsize=16)  # Original alpha values as labels
    yticks = [round(y, 3) for y in np.linspace(0.0, 0.60, 10)]
    ax1.set_yticks(yticks, labels=yticks, fontsize=16)
    # Increase major tick size and width
    ax1.tick_params(axis='both', which='major', length=10, width=2)
    ax1.tick_params(axis='both', which='minor', length=5, width=1)

    # Adding grid and legend to the primary axis
    ax1.grid(True)
    lines, labels = ax1.get_legend_handles_labels()
    ax1.legend(lines, labels, fontsize=16).set_zorder(2)

    fig.subplots_adjust(
        left=0.115,
        right=0.89,
        bottom=0.1,
        top=0.89,
        wspace=0.2,
        hspace=0.2
    )
    plt.title(r'SFT training: Open LLM Leaderboard score vs LoRA $\alpha$ (rank=1024)',
              fontsize=18, pad=15)

    plt.savefig(save_pth)


def gen_plot(save_pth: str, avgs: Dict) -> None:
    alpha = [16, 64, 256, 1024, 2048]
    log_alpha = np.log10(alpha)

    # mmlu
    mmlu_values = [0.5882, 0.5855, 0.5804, 0.5672, 0.5547]
    mmlu_top_of = [0.0018, 0.0019, 0.0020, 0.0020, 0.0020]
    mmlu_bot_of = [0.0018, 0.0019, 0.0020, 0.0020, 0.0020]
    mmlu_top = [x + off for x, off in zip(mmlu_values, mmlu_top_of)]
    mmlu_bot = [x - off for x, off in zip(mmlu_values, mmlu_bot_of)]

    fig, ax1 = plt.subplots(figsize=(8.3, 3.5))
    ax1.fill_between(log_alpha, mmlu_bot, mmlu_top, facecolor='#FF6347', alpha=0.2)
    ax1.axhline(y=0.588, color='#5F9EA0', linestyle='--', linewidth=4,
                label='Zephyr-7B baseline')
    ax1.plot(log_alpha, mmlu_values, color="#FF6347", marker='x', linewidth=2,
             markersize=12, mew=2,
             linestyle=':', label='MMLU: baseline + SFT $\pm SD$')
    ax1.set_ylim(0.54, 0.60)  # Set the y-axis limits inverted

    # Adding labels and title
    ax1.set_xlabel(r'$\alpha$ (log Scale)', fontsize=16)
    ax1.set_ylabel('MMLU', fontsize=16)
    # Customizing the ticks
    ax1.set_xticks(log_alpha, labels=alpha, fontsize=16)  # Original alpha values as labels
    yticks = [round(y, 3) for y in np.linspace(0.54, 0.60, 7)]
    ax1.set_yticks(yticks, labels=yticks, fontsize=16)
    # Increase major tick size and width
    ax1.tick_params(axis='both', which='major', length=10, width=2)
    ax1.tick_params(axis='both', which='minor', length=5, width=1)

    # Plot Leaderboard
    ax2 = ax1.twinx()
    ax2.set_ylim(0.2, 0.2925)
    ax2.set_ylabel('Open LLM Leaderboard', fontsize=16)  # Replace with an appropriate label
    y2ticks = [round(y, 2) for y in np.linspace(0.2, 0.28, 5)]
    ax2.set_yticks(y2ticks, labels=y2ticks, fontsize=16)
    ax2.tick_params(axis='y', which='major', length=10, width=2)

    lead_values = [avgs[a]["acc_none"][0] for a in alpha]
    lead_offset = [avgs[a]["acc_none"][1] for a in alpha]
    lead_top = [x + off for x, off in zip(lead_values, lead_offset)]
    lead_bot = [x - off for x, off in zip(lead_values, lead_offset)]
    ax2.fill_between(log_alpha, lead_bot, lead_top, facecolor='g', alpha=0.2)
    ax2.plot(log_alpha, lead_values, color="g", marker='+', linewidth=2, markersize=12, mew=2,
             linestyle='-.', label='O-LLM Leaderboard: baseline + SFT $\pm SD$')
 
    # Adding grid and legend to the primary axis
    ax1.grid(True)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, fontsize=16).set_zorder(2)

    fig.subplots_adjust(
        left=0.115,
        right=0.89,
        bottom=0.1,
        top=0.89,
        wspace=0.2,
        hspace=0.2
    )
    plt.title(r'SFT training: Language understanding scores vs LoRA $\alpha$ (rank=1024)',
              fontsize=18, pad=15)

    plt.savefig(save_pth)


if __name__ == "__main__":
    gen_appx_plot("results/test_appx.png", get_leaderboard_data(is_test=False, prnt=False))

