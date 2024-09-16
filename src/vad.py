"""
Intensity, Valence, Arousal.

Author: Andrew Lee 
Source: https://github.com/MichiganNLP/empathy_eval/blob/master/scripts/vad.py 


"""

import os
import json
import numpy as np
from nltk.tokenize import word_tokenize
import argparse
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm
from collections import defaultdict
i = os.path.dirname(os.path.realpath(__file__)).split('/').index('empathy-generation')
p = '/'.join(os.path.dirname(os.path.realpath(__file__)).split('/')[:i+1])
DATA_HOME = os.path.join(p, 'data/misc')


def load_vad():
    """
    Load vad info.
    """
    with open(os.path.join(DATA_HOME, "VAD.json"), "r") as file_p:
        vad = json.load(file_p)
    return vad


def load_intensity():
    """
    Load intensity info.
    """
    with open(os.path.join(DATA_HOME, "intensity.txt"), "r") as file_p:
        intensity_lex = file_p.readlines()

    intensities = {}
    for intensity in intensity_lex:
        tokens = intensity.split()
        intensities[tokens[0]] = float(tokens[2])
    return intensities


VAD = load_vad()
INTENSITIES = load_intensity()


def get_token_vad(token):
    """
    Get VAD vector
    """
    return VAD.get(token, [0, 0, 0])


def get_token_intensity(token):
    return INTENSITIES.get(token, 0)


def get_vad(query):
    """
    Get mean, max scores for VAD.
    """
    tokens = word_tokenize(query.lower())
    vads = [get_token_vad(token) for token in tokens]
    vads = [x for x in vads if x is not None]

    valence = [x[0] for x in vads]
    arousal = [x[1] for x in vads]
    dominance = [x[2] for x in vads]
    return valence, arousal, dominance


def get_intensity(query):
    tokens = word_tokenize(query.lower())
    return [get_token_intensity(token) for token in tokens]


def get_vad_stats(data, response_col):
    """
    Compute intensity, vad.
    """

    # results = []
    results = defaultdict(lambda:[])


    for idx, row in tqdm(data.iterrows(), total=len(data)):

        # context = row["query"]
        last_utt = row["prevs"]
        response = row[response_col]

        context_v, context_a, context_d = get_vad(last_utt)
        response_v, response_a, response_d = get_vad(response)

        context_intensity = get_intensity(last_utt)
        response_intensity = get_intensity(response)

        max_v_context = 0
        max_a_context = 0
        max_d_context = 0
        mean_v_context = 0
        mean_a_context = 0
        mean_d_context = 0

        if len(context_v) > 0:
            max_v_context = max(context_v)
            mean_v_context = np.mean(context_v)
        if len(context_a) > 0:
            max_a_context = max(context_a)
            mean_a_context = np.mean(context_a)
        if len(context_d) > 0:
            max_d_context = max(context_d)
            mean_d_context = np.mean(context_d)

        if len(response_v) > 0:
            max_v = max(response_v)
            mean_v = np.mean(response_v)
        if len(response_a) > 0:
            max_a = max(response_a)
            mean_a = np.mean(response_a)
        if len(response_d) > 0:
            max_d = max(response_d)
            mean_d = np.mean(response_d)

        diff_max_v = max_v_context - max_v
        diff_mean_v = mean_v_context - mean_v
        diff_max_a = max_a_context - max_a
        diff_mean_a = mean_a_context - mean_a
        diff_max_d = max_d_context - max_d
        diff_mean_d = mean_d_context - mean_d
        diff_intensity = max(context_intensity) - max(response_intensity)

        # results.append(
        #     {
        #         "max_v": max_v,
        #         "mean_v": mean_v,
        #         "max_a": max_a,
        #         "mean_a": mean_a,
        #         "max_d": max_d,
        #         "mean_d": mean_d,
        #         "diff_max_v": diff_max_v,
        #         "diff_mean_v": diff_mean_v,
        #         "diff_max_a": diff_max_a,
        #         "diff_mean_a": diff_mean_a,
        #         "diff_max_d": diff_max_d,
        #         "diff_mean_d": diff_mean_d,
        #         "diff_max_intensity": diff_intensity,
        #     }
        # )
        results['max_v'].append(max_v)
        results['mean_v'].append(mean_v)
        results['max_a'].append(max_a)
        results['mean_a'].append(mean_a)
        results['max_d'].append(max_d)
        results['mean_d'].append(mean_d)
        results['diff_max_v'].append(diff_max_v)
        results['diff_mean_v'].append(diff_mean_v)
        results['diff_max_a'].append(diff_max_a)
        results['diff_mean_a'].append(diff_mean_a)
        results['diff_max_d'].append(diff_max_d)
        results['diff_mean_d'].append(diff_mean_d)
        results['diff_max_intensity'].append(diff_intensity)

    return pd.DataFrame(results)

def results_report(scores):
    #TODO if we want

    return 

def evaluate(results_df, col="gens"):
    """
    Input:
        results_df: 
            loaded results_df or path to generations file
        col: 
            not needed if responses is a list/iterable of response strings, 
            but if it's a filepath, you could say which column you want to eval. 
            default is the column with the generated text.
            to eval the human text, col="gen_targets"
    """
    if type(results_df) == str:
        results_df = pd.read_csv(results_df, sep="~")
        results_df = results_df[results_df['gens'].str.len() != 0]



    vad_stats = get_vad_stats(results_df, col)

    # diff_max_v = np.mean([x["diff_max_v"] for x in vad_stats])
    # diff_max_a = np.mean([x["diff_max_a"] for x in vad_stats])
    # diff_max_d = np.mean([x["diff_max_d"] for x in vad_stats])
    # diff_max_intensity = np.mean(
    #     [x["diff_max_intensity"] for x in vad_stats]
    # )
    diff_max_v = vad_stats["diff_max_v"].mean()
    diff_max_a = vad_stats["diff_max_a"].mean()
    diff_max_d = vad_stats["diff_max_d"].mean()
    diff_max_intensity = vad_stats["diff_max_intensity"].mean()

    scores = {
            "diff_max_v": diff_max_v,
            "diff_max_a": diff_max_a,
            "diff_max_d": diff_max_d,
            "diff_max_intensity": diff_max_intensity
        }

    print("VAD scores for system")
    scores = {'metric':scores.keys(), 'score':scores.values()}
    print(tabulate(scores, headers=['metric', 'score']))

    return scores, vad_stats


def compare_vad(filepaths):
    """ Compare VADs """
    scores = {}
    for system, filepath in filepaths:
        with open(filepath, "r") as file_p:
            data = json.load(file_p)

        vad_stats = get_vad_stats(data, system)

        diff_max_v = np.mean([x["diff_max_v"] for x in vad_stats])
        diff_max_a = np.mean([x["diff_max_a"] for x in vad_stats])
        diff_max_d = np.mean([x["diff_max_d"] for x in vad_stats])
        diff_max_intensity = np.mean(
            [x["diff_max_intensity"] for x in vad_stats]
        )

        print("--")
        print("--")
        print("--")
        print(f"({system}) Diff Max V: {diff_max_v}")
        print(f"({system}) Diff Max A: {diff_max_a}")
        print(f"({system}) Diff Max D: {diff_max_d}")
        print(f"({system}) Diff Intensity: {diff_max_intensity}")

        scores[system] = {
            "diff_max_v": diff_max_v,
            "diff_max_a": diff_max_a,
            "diff_max_d": diff_max_d,
            "diff_max_intensity": diff_max_intensity,
        }
    return scores





# def main():
#     """ Driver """
#     systems = [
#         "trs",
#         "moel",
#         "mime",
#         "emocause",
#         "cem",
#         "kemp",
#         "seek",
#         "care",
#         "emphi_2",
#         "human",
#     ]
#     outputs_dir = os.path.join(EMP_HOME, "data/outputs/")
#     filepaths = [
#         (system, os.path.join(outputs_dir, f"{system}_responses.json"))
#         for system in systems
#     ]

#     compare_vad(filepaths)


if __name__ == "__main__":
    # main()
    i = os.path.dirname(os.path.realpath(__file__)).split('/').index('empathy-generation')
    p = '/'.join(os.path.dirname(os.path.realpath(__file__)).split('/')[:i+1])
    # main()
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", default=os.path.join(p,"data/empathy_datasets/empathetic_dialogues"), help="directory path of empathetic dialogues dataset")
    parser.add_argument("-i", "--responses_file", default=os.path.join(p,"data/generated_text/preds_x_zephyr-7b-sft-full122.txt"), help="path to file you want to run evaluation on")

    args = parser.parse_args()

    evaluate()
