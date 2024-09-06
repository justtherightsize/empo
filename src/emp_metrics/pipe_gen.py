# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import re
import time
from typing import Dict
import wandb
from src.emp_metrics.ed_load import get_ed_for_generation
from peft import PeftModel
from deepeval.models import DeepEvalBaseLLM
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline, \
    BlenderbotSmallForConditionalGeneration, BlenderbotForConditionalGeneration
import torch
import argparse
from deepeval.benchmarks import MMLU
from deepeval.benchmarks.tasks import MMLUTask
from huggingface_hub import login
from pathlib import Path


def get_gen_sys_msg(model_name):
    if "gemma" in model_name:
        l = "<|im_start|>user"
        r = "<|im_start|>assistant"
    else:
        l = "<|user|>"
        r = "<|assistant|>"
    return "You are a friendly assistant, who provides empathetic responses to the user. " \
            "The input contains previous turn of the dialog, where each utterance is prefaced " \
            "with tags {}, or {}. Be empathetic and precise. Make sure to give responses that make dialogue flow. Avoid repeating the prompt. " \
            "Please respond creatively and expressively to make the responses longer. You can offer advice."
           # "responses that make dialogue flow. Avoid repeating the prompt and giving unsolicited advice. Make the responses short.".format(l, r)

def calc_metrics(save_to, output_dir_base, base_model_id, adapter_id:str=None,
                 hf_key_path=None, is_local=False, is_test=False,
                 run_pref:str="", test_frac=1.0, max_tokens=200):
    # HF login
    login(Path(hf_key_path).read_text().strip())

    # Clean path off base dir
    if adapter_id is not None and output_dir_base in adapter_id:
        adapter_id = adapter_id.replace(output_dir_base, "")

    # Load tokenizer either from local moder dir or remote
    model_dir = "{}/{}".format(output_dir_base.rstrip("/"), adapter_id)
    if is_local:
        tokenizer = AutoTokenizer.from_pretrained(
                base_model_id if adapter_id is None else model_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
                base_model_id if adapter_id is None else adapter_id)

    sys_msg = get_gen_sys_msg(base_model_id if adapter_id is None else adapter_id)
    test_df = get_ed_for_generation("test", tokenizer, sys_msg=sys_msg,
                                    tokenize=False, add_generation_prompt=True)

    # Load the big model first & resize embeds, load PEFT model
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=quantization_config,
        trust_remote_code=True
    )
    model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False
    if adapter_id is not None:
        if is_local:
            model = PeftModel.from_pretrained(model, model_dir)
        else:
            model = PeftModel.from_pretrained(model, adapter_id)

    # Init pipeline & sys message
    sys_msg = get_gen_sys_msg(base_model_id if adapter_id is None else adapter_id)

    sl_time = 3
    print(f"Sleeping for {sl_time}s...")
    time.sleep(sl_time)

    # Instantiate generation pipeline
    pipe_gen = pipeline("text-generation", model=model, tokenizer=tokenizer,
                        max_new_tokens=max_tokens)
    print(max_tokens)
    # Subset test if specified
    if is_test:
        test_df = test_df.head(50).copy()
    elif test_frac < 0.9999:
        test_df = test_df.sample(int(len(test_df) * test_frac)).copy()

    # Gen preds iteratively
    gens = []
    for index, r in test_df.iterrows():
        out = pipe_gen(
                r["chat_templates"],
                return_full_text=False)[0]['generated_text']
        assert "~" not in out, f"Char ~ found in gen {index}: {out}"
        gens.append(out)
        # print(out)
    test_df["gens"] = gens

    # Set save name
    if adapter_id is None:
        sv_nm = base_model_id.split('/')[1]
    else:
        if is_local:
            sv_nm = adapter_id
        else:
            sv_nm = adapter_id.split('/')[1]

    # Save the dataframe
    pth = f"{output_dir_base.rstrip('/')}/preds_{run_pref}_{sv_nm}.txt"
    test_df.to_csv(pth, sep="~")
    print(f"3.-----Saving preds (SFT) to {pth}.--------")

    # Delete objects from mem
    del tokenizer
    del model
    del pipe_gen
    print(f"Sleeping for {sl_time}s...")
    time.sleep(sl_time)
    print("Done.")
    return {
            "path_preds": pth,
            "no_preds": len(test_df)}


def generate_predictions(args: argparse.Namespace) -> Dict:
    if args.adapter == "none":
        ret = calc_metrics(
                f"{args.base_dir}/mmlu_x_all_{args.base_model.split('/')[1]}.txt",
                args.base_dir, args.base_model, hf_key_path=args.hf_key_path,
                is_local=args.is_local, is_test=args.is_test,
                run_pref=args.run_pref, test_frac=args.test_frac,
                max_tokens=args.max_tokens)
    else:
        if args.is_local:
            pth = args.adapter
        else:
            pth = args.adapter.split('/')[1]
        ret = calc_metrics(
                f"{args.base_dir}/mmlu_x_{pth.replace(args.base_dir, '')}.txt",
                args.base_dir, args.base_model, adapter_id=args.adapter,
                hf_key_path=args.hf_key_path, is_local=args.is_local,
                is_test=args.is_test,
                run_pref=args.run_pref, test_frac=args.test_frac,
                max_tokens=args.max_tokens)
    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-bm", "--base_model", help="base model name",
                        default="alignment-handbook/zephyr-7b-sft-full")
    parser.add_argument("-a", "--adapter",
                        help="adapter name wo the directory (use base_dir)",
                        default="none")
    parser.add_argument("-d", "--base_dir", default="./results",
                        help="base dir with saved models")
    parser.add_argument("-k", "--hf_key_path",
                        default="/home/xsotolar/.huggingface/mistral",
                        help="absolute path")
    parser.add_argument("-l", "--is_local", default=False,
                        help="gets models from Hub",
                        action=argparse.BooleanOptionalAction)
    parser.add_argument("-t", "--is_test", default=False,
                        help="run on low fidelity",
                        action=argparse.BooleanOptionalAction)
    parser.add_argument("-p", "--run_pref", default="x")
    parser.add_argument("-f", "--test_frac", default=1.0)
    parser.add_argument("-m", "--max_tokens", default=201, type=int)
    res = generate_predictions(parser.parse_args())
    print(f"Saved {res['no_preds']} preds to: {res['path_preds']}")
