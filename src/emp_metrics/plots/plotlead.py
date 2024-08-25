from typing import List, Dict, Tuple
from src.emp_metrics.plots.leaderld import load_openllm_leaderboard
import numpy as np
import pprint


def avg_std(lists: Dict[str, List[float]]) -> Dict[int, Dict]:
    """ Return avg and std """
    def nms(vals: List[float]) -> Tuple[float, float]:
        ar = np.array(vals)
        return (np.mean(ar), np.std(ar))

    ret = {k: nms(v) for k, v in lists.items()}
    return ret


def main(is_test=False):
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
                   "leader_zephyr-7b-sft-full121/results_2024-08-25T02-50-52.401869.json"]}

    if is_test:
        results_dir = "."
        pths = {100: ["src/emp_metrics/plots/test1.json",
                      "src/emp_metrics/plots/test2.json"]}

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
    pprint.pp(avgs)


if __name__ == "__main__":
    main(is_test=True)
