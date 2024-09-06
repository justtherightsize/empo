import json
import os
import argparse


def compute_averages(json_files):
    total_diff_ER = 0
    total_diff_EX = 0
    total_diff_IP = 0
    count = 0
    
    for file_name in json_files:
        with open(file_name, 'r') as file:
            data = json.load(file)
            
            # Extract values
            diff_ER = float(data['epitome']['diff_ER'])
            diff_EX = float(data['epitome']['diff_EX'])
            diff_IP = float(data['epitome']['diff_IP'])
            
            # Add to totals
            total_diff_ER += diff_ER
            total_diff_EX += diff_EX
            total_diff_IP += diff_IP
            count += 1
            
    # Compute averages
    avg_diff_ER = total_diff_ER / count if count > 0 else 0
    avg_diff_EX = total_diff_EX / count if count > 0 else 0
    avg_diff_IP = total_diff_IP / count if count > 0 else 0
    avg_total = (avg_diff_ER + avg_diff_EX + avg_diff_IP) / 3

    return avg_diff_ER, avg_diff_EX, avg_diff_IP, avg_total


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
    avg_diff_ER, avg_diff_EX, avg_diff_IP, avg_total = compute_averages(flz)

    # Print results
    print(f"Average total: {avg_total}")
    print(f"Average diff_ER: {avg_diff_ER}")
    print(f"Average diff_EX: {avg_diff_EX}")
    print(f"Average diff_IP: {avg_diff_IP}")


if __name__ == "__main__":
    main()
