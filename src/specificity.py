"""
Author: Andrew Lee 
Source: https://github.com/MichiganNLP/empathy_eval/blob/master/scripts/specificity.py 

Script to compute NIDF scores.
"""

import os
import pickle
import math
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
from tqdm import tqdm
from nltk.tokenize import word_tokenize
import argparse
from tabulate import tabulate # pip install tabulate



def compute_nidf(responses, ed_counts):
    """
    Count up tokens.
    """
    word_counts = ed_counts["word_counts"]

    min_c = min(word_counts.values())
    max_c = max(word_counts.values())

    word2nidf = {
        word: (math.log(max_c) - math.log(_count))
        / (math.log(max_c) - math.log(min_c))
        for word, _count in word_counts.items()
    }

    nidf_scores = []
    for resp in responses:
        tokens = list(set(word_tokenize(resp)))

        nidfs = [word2nidf.get(tok, 1) for tok in tokens]
        nidf_scores.append(np.mean(nidfs))
    return nidf_scores


def count_ed(ed_files):
    """
    Count vocab in ED.
    """
    counts = Counter()
    num_sents = defaultdict(lambda: 0)
    for filepath in ed_files:
        # print(f"Counting {filepath}")
        f = filepath.split('/')[-1]

        data = pd.read_csv(filepath, usecols=["sp1"])
        # with open(filepath, "r") as file_p:
        #     data = file_p.readlines()
        for sample in tqdm(data['sp1'], desc=f"Counting {f}"):

            # parts = sample.strip().split(",")
            # utterance = parts[5].replace("_comma_", ",")
            utterance = sample.replace("_comma_", ",")

            tokens = list(set(word_tokenize(utterance)))
            counts.update(tokens)
            for tok in tokens:
                num_sents[tok] += 1

    return {"word_counts": counts, "num_sents": dict(num_sents)}


def results_report(nidfs, out=True):
    """
        if out is True, it will print a table
    """
    results = {"stat":[], "val":[]}
    results['stat'].append("mean")
    results['stat'].append("med ")
    results['stat'].append("min ")
    results['stat'].append("max ")
    results['stat'].append("std ")
    results['stat'].append("var ")

    results['val'].append(np.mean(nidfs))
    results['val'].append(np.median(nidfs))
    results['val'].append(min(nidfs))
    results['val'].append(max(nidfs))
    results['val'].append(np.std(nidfs))
    results['val'].append(np.var(nidfs))

    if out:
        print("NIDF for system")
        print(tabulate(results))

    return pd.DataFrame(results)

def evaluate(ed_dir, responses, col="gens"):
    """
    Input:
        ed_dir: 
            path to directory containing the empathetic dialogues split files
        responses:
            can be a list of response strings or a file path
        col: 
            not needed if responses is a list/iterable of response strings, 
            but if it's a filepath, you could say which column you want to eval. 
            default is the column with the generated text.
            to eval the human text, col="gen_targets"
    Returns:
        system_nidf: 
            mean nidf across all responses
        results:
            df of stats
        nidfs:
            list of nidf per each generation
    """
    ed_files = [
        os.path.join(ed_dir, "train.csv"),
        os.path.join(ed_dir, "val.csv"),
    ]
    ed_counts_filepath = os.path.join(ed_dir, "ed_counts.pkl")

    if os.path.isfile(ed_counts_filepath):
        with open(ed_counts_filepath, "rb") as file_p:
            ed_counts = pickle.load(file_p)
    else:
        ed_counts = count_ed(ed_files)
        with open(ed_counts_filepath, "wb") as file_p:
            pickle.dump(ed_counts, file_p)


    if type(responses) == str:
        test_df = pd.read_csv(responses, sep="~")
        test_df = test_df[test_df['gens'].str.len() != 0]
        responses = test_df[col].values

    # responses = [resp_obj["response"] for resp_obj in data]
    nidfs = compute_nidf(responses, ed_counts)
    system_nidf = np.mean(nidfs)   

    results = results_report(nidfs)
    

    return system_nidf, results, nidfs


# def main():
#     """ Driver """
#     outputs_dir = os.path.join(EMP_HOME, "data/outputs")

#     ed_dir = os.path.join(EMP_HOME, "data/empatheticdialogues")
#     ed_files = [
#         os.path.join(ed_dir, "train.csv"),
#         os.path.join(ed_dir, "valid.csv"),
#     ]
#     ed_counts_filepath = os.path.join(EMP_HOME, "data/ed_counts.pkl")
#     if os.path.isfile(ed_counts_filepath):
#         with open(ed_counts_filepath, "rb") as file_p:
#             ed_counts = pickle.load(file_p)
#     else:
#         ed_counts = count_ed(ed_files)
#         with open(ed_counts_filepath, "wb") as file_p:
#             pickle.dump(ed_counts, file_p)

#     systems = [
#         "trs",
#         "care",
#         "cem",
#         "emocause",
#         "emphi",
#         "human",
#         "kemp",
#         "mime",
#         "moel",
#         "seek",
#     ]

#     for system in systems:
#         responses_file = os.path.join(outputs_dir, f"{system}_responses.json")
#         with open(responses_file, "r") as file_p:
#             data = json.load(file_p)

#         responses = [resp_obj["response"] for resp_obj in data]
#         nidfs = compute_nidf(responses, ed_counts)
#         print(f"NIDF for {system}: {np.mean(nidfs)}")


if __name__ == "__main__":
    i = os.path.dirname(os.path.realpath(__file__)).split('/').index('empathy-generation')
    p = '/'.join(os.path.dirname(os.path.realpath(__file__)).split('/')[:i+1])
    # main()
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", default=os.path.join(p,"data/empathy_datasets/empathetic_dialogues"), help="directory path of empathetic dialogues dataset")
    parser.add_argument("-i", "--responses_file", default=os.path.join(p,"data/generated_text/preds_x_zephyr-7b-sft-full122.txt"), help="path to file you want to run evaluation on")

    args = parser.parse_args()

    evaluate(args.data_dir, args.responses_file)
