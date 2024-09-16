import json
import os
import argparse
import numpy as np
import statistics as sta


def compute_averages(json_files):
    total_diff_ER = 0
    total_diff_EX = 0
    total_diff_IP = 0
    count = 0
    avgdata = []
    total_bf1 = 0
    ers = []
    exs = []
    ips = []

    for file_name in json_files:
        with open(file_name, 'r') as file:
            data = json.load(file)
            
            # Extract values
            diff_ER = float(data['epitome']['diff_ER'])
            diff_EX = float(data['epitome']['diff_EX'])
            diff_IP = float(data['epitome']['diff_IP'])
            ers.append(diff_ER)
            exs.append(diff_EX)
            ips.append(diff_IP)

            bf1 = float(data["bertscore"]["f1_bert"])

            # Add to totals
            total_diff_ER += diff_ER
            total_diff_EX += diff_EX
            total_diff_IP += diff_IP
            total_bf1 += bf1
            count += 1
            avgdata.append((diff_ER + diff_EX + diff_IP) / 3)
            
            print(f"{bf1:0.3f}")
            
    # Compute averages
    avg_diff_ER = total_diff_ER / count if count > 0 else 0
    avg_diff_EX = total_diff_EX / count if count > 0 else 0
    avg_diff_IP = total_diff_IP / count if count > 0 else 0
    avg_bf1 = total_bf1 / count if count > 0 else 0
    avg_total = (avg_diff_ER + avg_diff_EX + avg_diff_IP) / 3

    min_value = np.min(avgdata)
    max_value = np.max(avgdata)

    # Quartiles
    Q1 = np.percentile(avgdata, 25)
    Q2 = np.median(avgdata)
    Q3 = np.percentile(avgdata, 75)
    
    # Interquartile Range (IQR)
    IQR = Q3 - Q1
    
    # Detect outliers (points beyond 1.5 * IQR)
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = [x for x in avgdata if x < lower_bound or x > upper_bound]
    
    # Output results
    boxplot_stats = {
        'Minimum': min_value,
        'Q1': Q1,
        'Median (Q2)': Q2,
        'Q3': Q3,
        'Maximum': max_value,
        'IQR': IQR,
        'Outliers': outliers
    }
    if len(exs) == 1:
        ers.append(ers[0])
        exs.append(exs[0])
        ips.append(ips[0])
    stds = {
            "er": sta.stdev(ers),
            "ex": sta.stdev(exs),
            "ip": sta.stdev(ips)
            }

    return avg_diff_ER, avg_diff_EX, avg_diff_IP, avg_total, boxplot_stats, avg_bf1, stds


def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(
            description="Compute averages of diff_ER, diff_EX...")
    parser.add_argument('-f', '--files', type=str, nargs='+',
                        help='JSON files to process')
    
    # Parse arguments
    args = parser.parse_args()
    res_dir = "results"
    flz = [res_dir + "/" + f for f in args.files]

    # Compute averages
    avg_diff_ER, avg_diff_EX, avg_diff_IP, avg_total, b, berts, stds = compute_averages(flz)

    # Print results
    print(f"Average total: {avg_total:0.3f}")
    print(f"Average diff_ER: {avg_diff_ER:0.3f}")
    print(f"Average diff_EX: {avg_diff_EX:0.3f}")
    print(f"Average diff_IP: {avg_diff_IP:0.3f}")
    print(f"std ER: {stds['er']:0.2f}")
    print(f"std EX: {stds['ex']:0.2f}")
    print(f"std IP: {stds['ip']:0.2f}")
    print(f"Average bert_f1: {berts:0.3f}")
    
    print(f"[{b['Minimum']:0.3f}, {b['Q1']:0.3f}, {b['Median (Q2)']:0.3f}, {b['Q3']:0.3f}, {b['Maximum']:0.3f}]")


if __name__ == "__main__":
    main()
