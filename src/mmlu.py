from typing import List, Dict

import wandb
wandb.init(mode="disabled")

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from peft import PeftModel
from deepeval.models import DeepEvalBaseLLM
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline, \
    BlenderbotSmallForConditionalGeneration, BlenderbotForConditionalGeneration

import torch
import argparse
import pprint
import pandas as pd
from deepeval.benchmarks import MMLU
from deepeval.benchmarks.tasks import MMLUTask


class Mistral7B(DeepEvalBaseLLM):
    def __init__(self, pipe, tokenizer, sys_msg):
        self.pipe = pipe
        self.tokenizer = tokenizer
        self.sys_msg = sys_msg

    def load_model(self):
        return self.pipe.model

    def generate(self, prompt: str) -> str:
        p = self.prep_for_generation(prompt, sys_msg=self.sys_msg,
                                 tokenize=False, add_generation_prompt=True)
        dec = self.pipe(p, return_full_text=False)[0]["generated_text"]
        print(dec)

        if len(dec) == 9:
            dec = dec[-1]

        return dec

    def prep_for_generation(self, prompt: str,  **kwargs) -> str:
        odd_d = [{"role": "system", "content": self.sys_msg},
                 {"role": "user", "content": prompt}]
        test_tok = self.tokenizer.apply_chat_template(odd_d, **kwargs)
        return test_tok

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return "Mistral 7B"


class BlenderbotSm(DeepEvalBaseLLM):
    def __init__(self, model, tokenizer, sys_msg):
        self.model = model
        self.tokenizer = tokenizer
        self.sys_msg = sys_msg

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, padding=True, truncation=True,
                           return_tensors="pt").to("cuda")
        reply_ids = self.model.generate(**inputs)
        dec = self.tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]
        print(dec)

        if len(dec) == 9:
            dec = dec[-1]

        return dec

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return "Mistral 7B"


def calc_metrics(save_to, output_dir_base, base_model_id, model_id):
    output_dir = output_dir_base + model_id
    sys_msg = "You are a assistant, who provides true responses to the user. " \
              "The input contains previous turn of the dialog, where the each utterance is prefaced " \
              "with tags <|user|>, or <|assistant|>. Be precise. Avoid repeating the prompt."

    if "blenderbot" in model_id and "small" in model_id:
        model = BlenderbotSmallForConditionalGeneration.from_pretrained(model_id).to("cuda")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        eval_model = BlenderbotSm(model, tokenizer, sys_msg)
    elif "blenderbot" in model_id:
        model = BlenderbotForConditionalGeneration.from_pretrained(model_id).to("cuda")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        eval_model = BlenderbotSm(model, tokenizer, sys_msg)
    else:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        tokenizer = AutoTokenizer.from_pretrained(output_dir)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=quantization_config,
            trust_remote_code=True
        )
        model.resize_token_embeddings(len(tokenizer))
        model.config.use_cache = False
        # config = PeftConfig.from_pretrained(output_dir)
        model = PeftModel.from_pretrained(model, output_dir)
        pipee = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=50)
        eval_model = Mistral7B(pipe=pipee, tokenizer=tokenizer, sys_msg=sys_msg)

    # Define benchmark with specific tasks and shots
    benchmark = MMLU(
        tasks=[MMLUTask.GLOBAL_FACTS],
        n_shots=5
    )

    benchmark.evaluate(model=eval_model)
    print(benchmark.overall_score)
    print("Task-specific Scores: ", benchmark.task_scores)

    with open(save_to, 'w') as f:
        f.write(
            f"{benchmark.overall_score}\n\n{benchmark.task_scores}")


def main(args: argparse.Namespace) -> None:
    base_model_id = args.base_model
    model_id = args.adapter #"zephyr-qlora-empathy3_dpo2"
    output_dir_base = args.base_dir #"./results/"

    pth_to_csv = f"{output_dir_base}/mmlu_{model_id}.txt"
    calc_metrics(pth_to_csv, output_dir_base, base_model_id, model_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu", default="1", help="not implemented")
    parser.add_argument("-bm", "--base_model", default="alignment-handbook/zephyr-7b-sft-lora",
                        help="base model name")
    parser.add_argument("-a", "--adapter", help="adapter name")
    parser.add_argument("-d", "--base_dir", default="./results/", help="base dir with saved models")


    main(parser.parse_args())
