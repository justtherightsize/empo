import json
import argparse


def load_json_file(filename):
    """
    Load a JSON file and return its contents.
    :param filename: The name of the JSON file to load.
    :return: The contents of the JSON file as a Python dictionary.
    """
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file '{filename}' is not a valid JSON file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return None


def load_openllm_leaderboard(name, printout=True):
    """
    Main method to load and print a JSON file based on the provided filename.
    :param name: The name of the JSON file to load.
    """
    data = load_json_file(name)
    if data is None:
        return None

    res = {}
    res["acc_none"] = float(data["results"]["leaderboard"]["acc,none"])
    res["acc_norm_none"] = float(
            data["results"]["leaderboard"]["acc_norm,none"])
    res["bbh_norm"] = float(
            data["results"]["leaderboard_bbh"]["acc_norm,none"])
    res["gpqa_norm"] = float(
            data["results"]["leaderboard_gpqa"]["acc_norm,none"])
    res["ifeval_norm"] = float(
            data["results"]["leaderboard_ifeval"]["inst_level_loose_acc,none"])
    res["math"] = float(
            data["results"]["leaderboard_math_hard"]["exact_match,none"])
    res["mmlupro"] = float(
            data["results"]["leaderboard_mmlu_pro"]["acc,none"])
    res["musr"] = float(data["results"]["leaderboard_musr"]["acc_norm,none"])

    if printout:
        for key, value in res.items():
            print(f"{key}: {value:.3f}")

    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and display a JSON file.")
    parser.add_argument("-n", '--name', type=str,
                        help="The name of the JSON file to load.")
    args = parser.parse_args()
    load_openllm_leaderboard(args.name)
