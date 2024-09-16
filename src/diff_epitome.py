# Code mostly copied from https://github.com/passing2961/EmpGPT-3/evaluate.py
import argparse
import math
import os
from typing import Dict, List

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn

from transformers import RobertaTokenizer
from parlai.core.metrics import (
    AverageMetric,
)

from from_epitome.models import (
    BiEncoderAttentionWithRationaleClassification
)

#############################
# IDK what exactly is the purpose of the code below. I suspects it downloads the finetuned Epitome models
# from a Gdrive and puts them into datapath/models/epitome. But the README suggests donwloading
# the models manually and use the epitome_save_dir arg. So maybe it's legacy code?
#############################
# RESOURCES = [
#     build_data.DownloadableFile(
#         '1P3Gd4uEzH-SS0L9K5TOktsPlR9rKU5sv',
#         'finetuned_EX.pth',
#         '4f43ceb2526e008a2093856208abb878f14236dd54e4fdcdfdd4ccbeb9c08178',
#         zipped=False, from_google=True,
#     ),
#     build_data.DownloadableFile(
#         '1Ta5PvUV-UFFWUa_WmyT0YFYez_XL6bb2',
#         'finetuned_IP.pth',
#         'e80d1bcfb75f7046961ed71cfd2eada2d939f7f1191e169b0b4aa68e9b6054dc',
#         zipped=False, from_google=True,
#     ),
# ]
#
#
# def _build(datapath):
#     dpath = os.path.join(datapath, 'models', 'epitome')
#     version = '1.0'
#
#     if not build_data.built(dpath, version_string=version):
#         print('[Downloading and building empathy scorer: ' + dpath + ']')
#         print('NOTE: The download can take about 4 minutes (likely to vary depending on your internet speed)')
#         if build_data.built(dpath):
#             # An older version exists, so remove these outdated files.
#             build_data.remove_dir(dpath)
#         build_data.make_dir(dpath)
#
#         # Download the data.
#         for downloadable_file in RESOURCES:
#             downloadable_file.download_file(dpath)
#
#         # Mark the data as built.
#         build_data.mark_done(dpath, version_string=version)
#
#     return dpath


class EmpathyScorer(nn.Module):
    """
    Evaluates the 3 epitome metrics IP, EX, ER.
    """
    def __init__(self, opt, batch_size=1, cuda_device=0):
        print("Loading EmpathyScorer...")
        super().__init__()
        # ckpt_path = _build(opt['datapath'])
        # print(ckpt_path)
        self.opt = opt
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)
        self.batch_size = batch_size
        self.cuda_device = cuda_device

        self.model_IP = BiEncoderAttentionWithRationaleClassification()
        self.model_EX = BiEncoderAttentionWithRationaleClassification()
        self.model_ER = BiEncoderAttentionWithRationaleClassification()

        IP_weights = torch.load(os.path.join(opt['epitome_save_dir'], 'finetuned_IP.pth'))
        self.model_IP.load_state_dict(IP_weights)

        EX_weights = torch.load(os.path.join(opt['epitome_save_dir'], 'finetuned_EX.pth'))
        self.model_EX.load_state_dict(EX_weights)

        ER_weights = torch.load(os.path.join(opt['epitome_save_dir'], 'finetuned_ER.pth'))
        self.model_ER.load_state_dict(ER_weights)

        # self.use_cuda = not opt['no_cuda'] and torch.cuda.is_available()
        self.use_cuda = True
        if self.use_cuda:
            self.model_IP.cuda(self.cuda_device)
            self.model_EX.cuda(self.cuda_device)
            self.model_ER.cuda(self.cuda_device)

    # TODO: is the 64 token limit a problem?
    def forward(self, seeker_post, response_post):
        self.model_IP.eval()
        self.model_EX.eval()
        self.model_ER.eval()

        # 'input_ids', 'attention_mask'
        seeker_input = self.tokenizer.batch_encode_plus(
            seeker_post,
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=64,  # Pad & truncate all sentences.
            truncation=True,
            pad_to_max_length=False,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
            padding=True,
        )
        response_input = self.tokenizer.batch_encode_plus(
            response_post,
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=64,  # Pad & truncate all sentences.
            truncation=True,
            pad_to_max_length=False,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
            padding=True
        )
        if self.use_cuda:
            # We assume all parameters of the model are on the same cuda device
            device = next(self.model_IP.parameters()).device
            seeker_input['input_ids'] = seeker_input['input_ids'].to(device)
            seeker_input['attention_mask'] = seeker_input['attention_mask'].to(device)
            response_input['input_ids'] = response_input['input_ids'].to(device)
            response_input['attention_mask'] = response_input['attention_mask'].to(device)

        with torch.no_grad():
            (logits_empathy_IP, logits_rationale_IP,) = self.model_IP(
                input_ids_SP=seeker_input['input_ids'],
                input_ids_RP=response_input['input_ids'],
                token_type_ids_SP=None,
                token_type_ids_RP=None,
                attention_mask_SP=seeker_input['attention_mask'],
                attention_mask_RP=response_input['attention_mask']
            )
            (logits_empathy_EX, logits_rationale_EX,) = self.model_EX(
                input_ids_SP=seeker_input['input_ids'],
                input_ids_RP=response_input['input_ids'],
                token_type_ids_SP=None,
                token_type_ids_RP=None,
                attention_mask_SP=seeker_input['attention_mask'],
                attention_mask_RP=response_input['attention_mask']
            )
            (logits_empathy_ER, logits_rationale_ER,) = self.model_ER(
                input_ids_SP=seeker_input['input_ids'],
                input_ids_RP=response_input['input_ids'],
                token_type_ids_SP=None,
                token_type_ids_RP=None,
                attention_mask_SP=seeker_input['attention_mask'],
                attention_mask_RP=response_input['attention_mask']
            )

        logits_empathy_IP = torch.nn.functional.softmax(logits_empathy_IP, dim=1)
        logits_empathy_IP = logits_empathy_IP.detach().cpu().numpy()
        logits_rationale_IP = logits_rationale_IP.detach().cpu().numpy()
        empathy_predictions_IP = np.argmax(logits_empathy_IP, axis=1).tolist()
        rationale_predictions_IP = np.argmax(logits_rationale_IP, axis=2)

        logits_empathy_EX = torch.nn.functional.softmax(logits_empathy_EX, dim=1)
        logits_empathy_EX = logits_empathy_EX.detach().cpu().numpy()
        logits_rationale_EX = logits_rationale_EX.detach().cpu().numpy()
        empathy_predictions_EX = np.argmax(logits_empathy_EX, axis=1).tolist()
        rationale_predictions_EX = np.argmax(logits_rationale_EX, axis=2)

        logits_empathy_ER = torch.nn.functional.softmax(logits_empathy_ER, dim=1)
        logits_empathy_ER = logits_empathy_ER.detach().cpu().numpy()
        logits_rationale_ER = logits_rationale_ER.detach().cpu().numpy()
        empathy_predictions_ER = np.argmax(logits_empathy_ER, axis=1).tolist()
        rationale_predictions_ER = np.argmax(logits_rationale_ER, axis=2)

        return {'IP': (empathy_predictions_IP, logits_empathy_IP, rationale_predictions_IP),
                'EX': (empathy_predictions_EX, logits_empathy_EX, rationale_predictions_EX),
                'ER': (empathy_predictions_ER, logits_empathy_ER, rationale_predictions_ER)}


def get_epitome_score(data: List[Dict], epitome_empathy_scorer: EmpathyScorer):
    """
    Calculate the Epitome and Diff-Epitome scores for batch of EmpatheticDialogues-style data.

    @data: list of dicts like this: [{
            'utterance': 'I'm upset',
            'prediction': 'Calm down',
            'gt': 'Sorry to hear that',
            'gt_emo': 'angry'
        }]

    @return: data, lists of scores for (IP, EX, ER) for (pred, gt, diff)
    """
    pred_IP_scores, pred_EX_scores, pred_ER_scores = [], [], []
    gt_IP_scores, gt_EX_scores, gt_ER_scores = [], [], []
    diff_IP_scores, diff_EX_scores, diff_ER_scores = [], [], []

    for example in tqdm(data):
        utter = example['utterance']
        pred = example['prediction']
        gt = example['gt']

        pred_epitome_score = epitome_empathy_scorer([utter], [pred])
        gt_epitome_score = epitome_empathy_scorer([utter], [gt])

        example['epitome-IP-pred'] = int(pred_epitome_score['IP'][0][0])
        example['epitome-EX-pred'] = int(pred_epitome_score['EX'][0][0])
        example['epitome-ER-pred'] = int(pred_epitome_score['ER'][0][0])

        example['epitome-IP-gt'] = int(gt_epitome_score['IP'][0][0])
        example['epitome-EX-gt'] = int(gt_epitome_score['EX'][0][0])
        example['epitome-ER-gt'] = int(gt_epitome_score['ER'][0][0])

        pred_IP_scores += pred_epitome_score['IP'][0]
        pred_EX_scores += pred_epitome_score['EX'][0]
        pred_ER_scores += pred_epitome_score['ER'][0]

        gt_IP_scores += gt_epitome_score['IP'][0]
        gt_EX_scores += gt_epitome_score['EX'][0]
        gt_ER_scores += gt_epitome_score['ER'][0]

        diff_IP_scores.append(math.pow(abs(pred_epitome_score['IP'][0][0] - gt_epitome_score['IP'][0][0]), 2))
        diff_EX_scores.append(math.pow(abs(pred_epitome_score['EX'][0][0] - gt_epitome_score['EX'][0][0]), 2))
        diff_ER_scores.append(math.pow(abs(pred_epitome_score['ER'][0][0] - gt_epitome_score['ER'][0][0]), 2))

    return data, pred_IP_scores, pred_EX_scores, pred_ER_scores, gt_IP_scores, gt_EX_scores, gt_ER_scores, diff_IP_scores, diff_EX_scores, diff_ER_scores


def avg_epitome_score(pred_IP_scores, pred_EX_scores, pred_ER_scores, gt_IP_scores, gt_EX_scores, gt_ER_scores, diff_IP_scores, diff_EX_scores, diff_ER_scores):
    """
    Averages the Epitome and Diff-Epitome scores for a batch.
    """
    report = {}
    report['pred_IP'] = AverageMetric(sum(pred_IP_scores), len(pred_IP_scores))
    report['pred_EX'] = AverageMetric(sum(pred_EX_scores), len(pred_EX_scores))
    report['pred_ER'] = AverageMetric(sum(pred_ER_scores), len(pred_ER_scores))

    report['gt_IP'] = AverageMetric(sum(gt_IP_scores), len(gt_IP_scores))
    report['gt_EX'] = AverageMetric(sum(gt_EX_scores), len(gt_EX_scores))
    report['gt_ER'] = AverageMetric(sum(gt_ER_scores), len(gt_ER_scores))

    report['diff_IP'] = AverageMetric(sum(diff_IP_scores), len(diff_IP_scores))
    report['diff_EX'] = AverageMetric(sum(diff_EX_scores), len(diff_EX_scores))
    report['diff_ER'] = AverageMetric(sum(diff_ER_scores), len(diff_ER_scores))
    return report


def to_epi_format(prevs, preds, gts):
    assert len(prevs) == len(preds) and len(prevs) == len(gts)
    return [{'utterance': r.lower(), 'prediction': p.lower(), 'gt': g.lower()}
            for r, p, g in zip(prevs, preds, gts)]


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epitome_save_dir', type=str, default=None)

    return parser.parse_args()


# Usage
if __name__ == '__main__':
    args = _parse_args()
    opt = {}
    opt['no_cuda'] = False
    device = 0
    opt['epitome_save_dir'] = args.epitome_save_dir
    epitome_empathy_scorer = EmpathyScorer(opt, batch_size=1, cuda_device=device)

    results = [
        {
            'utterance': 'I lost my job last year and got really angry.'.lower(),
            'prediction': 'I am sorry to hear that. Did it happen out of the blue?'.lower(),
            'gt': 'I am sorry to hear that. Did it happen out of the blue?'.lower(),
            'gt_emo': 'angry'
        },
        {
            'utterance': 'I lost my job last year and got really angry.'.lower(),
            'prediction': 'That sucks bro.'.lower(),
            'gt': 'I am sorry to hear that. Did it happen out of the blue?'.lower(),
            'gt_emo': 'angry'
        }
    ]
    results, pred_IP_scores, pred_EX_scores, pred_ER_scores, gt_IP_scores, gt_EX_scores, gt_ER_scores, diff_IP_scores, diff_EX_scores, diff_ER_scores = get_epitome_score(
        results, epitome_empathy_scorer)

    report = avg_epitome_score(pred_IP_scores, pred_EX_scores, pred_ER_scores, gt_IP_scores, gt_EX_scores, gt_ER_scores, diff_IP_scores, diff_EX_scores, diff_ER_scores)
    print(report)
