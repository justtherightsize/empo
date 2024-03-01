from adapters import AutoAdapterModel
from transformers import AutoTokenizer
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import os



def main():


    adapter_info = pd.read_json('data/trained_adapters/adapter_info.json')
    adapter_path = adapter_info.loc[args.adapter_name, 'adapter_path']
    print(adapter_path)
    df = pd.read_csv(args.datafile, index_col=0)

    results_df = defaultdict(lambda:[])
    

    tokenizer = AutoTokenizer.from_pretrained(args.base_lm_name)
    model = AutoAdapterModel.from_pretrained(args.base_lm_name)
    adapter_name = model.load_adapter(adapter_path, with_head=True, model_name=args.base_lm_name)
    model.set_active_adapters(adapter_name)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc='Running inference on samples'):

        text = row[args.text_column]
        
        output = model(tokenizer(text, return_tensors="pt").input_ids)

        if adapter_info.loc[args.adapter_name, 'tt'] == 'regression':
            result = float(output.logits[0][0].detach())
            results_df['idx'].append(idx)
            results_df[args.text_column].append(text)
            results_df[args.adapter_name].append(result)


    results_df = pd.DataFrame(results_df).set_index('idx')
    results_df.to_csv(args.output_path)
            

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='python empathy_inference.py',
                    description='runs an empathy model on a set of text.')
    parser.add_argument('-i', '--datafile', default='/home/alahnala/research/empathy-generation/data/empathy_datasets/eqt_human_msplit/Llama-2-13b-chat_subset_generated_text.csv', help='Path to a csv file containing the text')
    parser.add_argument('-tc', '--text_column', default='generated_text', help='Column where you have the text you want to run the emapthy inference on')
    parser.add_argument('-m', '--base_lm_name', default='roberta-base')
    parser.add_argument('-a', '--adapter_name', default='wassa_essay_empathy')
    parser.add_argument('-o', '--output_path', default=None)

      

    args = parser.parse_args()

    if not args.output_path:
        parts = args.datafile.split('/')
        oparts = '/'.join(parts[-2:])
        args.output_path = os.path.join('data/generation_evaluations/', oparts)

        outdir = '/'.join(args.output_path.split('/')[:-1])

        os.makedirs(outdir, exist_ok=True)

    main()




    