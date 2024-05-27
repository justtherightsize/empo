import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from typing import List, Dict, re
import wandb
wandb.init(mode="disabled")
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
from huggingface_hub import login
from pathlib import Path
login(Path('/home/xsotolar/.huggingface/mistral').read_text().strip())

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

        if len(dec) == 9:
            dec = dec[-1]

        if len(dec) > 0 and "is: " in dec:
            match = re.search(r'is: (A|B|C|D)', dec)
            if match:
                dec = match.group(0)[-1]

        if len(dec) < 1:
            dec = " "

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


def get_sys_msg(model_name):
    l = "<|user|>"
    r = "<|assistant|>"
    return "You are a assistant, who provides true responses to the user. " \
           "The input contains previous turn of the dialog, where the each utterance is prefaced " \
           "with tags {}, or {}. Be precise. Avoid repeating the prompt.".format(l, r)


def calc_metrics(save_to, output_dir_base, base_model_id, model_id=None):
    # MMLU
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id if model_id is None else model_id)

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=quantization_config,
        trust_remote_code=True
    )
    model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False
    if model_id is not None:
        model = PeftModel.from_pretrained(model, model_id)

    pipee = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)
    sys_msg = get_sys_msg(base_model_id if model_id is None else model_id)
    eval_model = Mistral7B(pipe=pipee, tokenizer=tokenizer, sys_msg=sys_msg)

    benchmark = MMLU(
        # tasks=[MMLUTask.GLOBAL_FACTS],
        n_shots=5
    )
    benchmark.evaluate(model=eval_model)
    print(benchmark.overall_score)
    print("Task-specific Scores: ", benchmark.task_scores)

    with open(save_to, 'w') as f:
        f.write(
            f"{benchmark.overall_score}\n\n{benchmark.task_scores}")
    del(tokenizer)
    del(model)
    del(pipee)


# def main(args: argparse.Namespace) -> None:
def main() -> None:
    base_model_id = "HuggingFaceH4/zephyr-7b-alpha"
    other_base = "HuggingFaceH4/zephyr-7b-beta"
    model_id_lora = "alignment-handbook/zephyr-7b-sft-lora"
    model_id_ft = "alignment-handbook/zephyr-7b-sft-full"
    output_dir_base = "./results/"

    #1. base model
    calc_metrics(f"{output_dir_base}/metrics_{base_model_id.split('/')[1]}.txt",
                 output_dir_base, base_model_id)

    calc_metrics(f"{output_dir_base}/metrics_{other_base.split('/')[1]}.txt",
                 output_dir_base, other_base)

    # 2. base model + lora
    calc_metrics(f"{output_dir_base}/metrics_{model_id_lora.split('/')[1]}.txt",
                 output_dir_base, base_model_id, model_id_lora)

    # 3. base model + sft
    calc_metrics(f"{output_dir_base}/metrics_{model_id_ft.split('/')[1]}.txt",
                 output_dir_base, model_id_ft)



if __name__ == "__main__":
    main()
