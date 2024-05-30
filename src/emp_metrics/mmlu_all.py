# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import re
import time
import wandb
from src.emp_metrics.ed_load import get_ed_for_generation
wandb.init(mode="disabled")
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
            match = re.search(r'is: ([ABCD])', dec)
            if match:
                dec = match.group(0)[-1]

        if len(dec) < 1:
            dec = " "

        return dec

    def prep_for_generation(self, prompt: str,  **kwargs) -> str:
        odd_d = [] if self.sys_msg is None else [{"role": "system", "content": self.sys_msg}]
        odd_d.append({"role": "user", "content": prompt})
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
    if "mistralai/Mistral-7B-Instruct-v0.1" in model_name:
        return None
    else:
        l = "<|user|>"
        r = "<|assistant|>"
        return "You are a assistant, who provides true responses to the user. " \
               "The input contains previous turn of the dialog, where the each utterance is prefaced " \
               "with tags {}, or {}. Be precise. Avoid repeating the prompt.".format(l, r)


def get_gen_sys_msg(model_name):
    if "gemma" in model_name:
        l = "<|im_start|>user"
        r = "<|im_start|>assistant"
    else:
        l = "<|user|>"
        r = "<|assistant|>"
    return "You are a friendly assistant, who provides empathetic responses to the user. " \
            "The input contains previous turn of the dialog, where each utterance is prefaced " \
            "with tags {}, or {}. Be empathetic and precise. Make sure to give " \
            "responses that make dialogue flow. Avoid repeating the prompt.".format(l, r)


def calc_metrics(save_to, output_dir_base, base_model_id, adapter_id=None,
                 hf_key_path=None, is_local=False, is_test=False):
    login(Path(hf_key_path).read_text().strip())
    # mmlu
    model_dir = "{}/{}".format(output_dir_base, adapter_id)
    if is_local:
        tokenizer = AutoTokenizer.from_pretrained(
                base_model_id if adapter_id is None else model_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
                base_model_id if adapter_id is None else adapter_id)

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

    pipee = pipeline("text-generation", model=model, tokenizer=tokenizer,
                     max_new_tokens=42)
    sys_msg = get_sys_msg(base_model_id if adapter_id is None else adapter_id)
    eval_model = Mistral7B(pipe=pipee, tokenizer=tokenizer, sys_msg=sys_msg)

    if is_test:
        benchmark = MMLU(tasks=[MMLUTask.GLOBAL_FACTS], n_shots=5)
    else:
        benchmark = MMLU(n_shots=5)
    benchmark.evaluate(model=eval_model)
    print(benchmark.overall_score)
    print("Task-specific Scores: ", benchmark.task_scores)

    with open(save_to, 'w') as f:
        f.write(
            f"{benchmark.overall_score}\n\n{benchmark.task_scores}")
        print(f"MMLU Saved to {save_to}.")
    sl_time = 3
    print(f"Sleeping for {sl_time}s...")
    time.sleep(sl_time)

    # generate preds
    pipe_gen = pipeline("text-generation", model=model, tokenizer=tokenizer,
                        max_new_tokens=200)
    sys_msg = get_gen_sys_msg(base_model_id if adapter_id is None else adapter_id)
    test_df = get_ed_for_generation("test", tokenizer, sys_msg=sys_msg, tokenize=False,
                                    add_generation_prompt=True)
    if is_test:
        test_df = test_df.head(3).copy()

    gens = []
    for index, r in test_df.iterrows():
        out = pipe_gen(r["chat_templates"], return_full_text=False)[0]['generated_text']
        assert "~" not in out, f"Char ~ found in gen {index}: {out}"
        gens.append(out)
    test_df["gens"] = gens
    #import ipdb; ipdb.set_trace()
    if adapter_id is None:
        sv_nm = base_model_id.split('/')[1]
    else:
        if is_local:
            sv_nm = adapter_id
        else:
            sv_nm = adapter_id.split('/')[1]
    pth = f"{output_dir_base}/preds_x_{sv_nm}.txt"
    test_df.to_csv(pth, sep="~")
    print(f"Preds saved to {pth}. Freeing memory...")

    del tokenizer
    del model
    del pipee
    print(f"Sleeping for {sl_time}s...")
    time.sleep(sl_time)
    print("Done.")


def main(args: argparse.Namespace) -> None:
    if args.adapter == "none":
        calc_metrics(f"{args.base_dir}/mmlu_x_all_{args.base_model.split('/')[1]}.txt",
                     args.base_dir, args.base_model, hf_key_path=args.hf_key_path, 
                     is_local=args.is_local, is_test=args.is_test)
    else:
        if args.is_local:
            pth = args.adapter
        else:
            pth = args.adapter.split('/')[1]
        calc_metrics(f"{args.base_dir}/mmlu_x_{pth}.txt",
                     args.base_dir, args.base_model, adapter_id=args.adapter, 
                     hf_key_path=args.hf_key_path, 
                     is_local=args.is_local, is_test=args.is_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-bm", "--base_model", help="base model name")
    parser.add_argument("-a", "--adapter", help="adapter name", default="none")
    parser.add_argument("-d", "--base_dir", default="./results", 
            help="base dir with saved models")
    parser.add_argument("-k", "--hf_key_path", help="absolute path")
    parser.add_argument("-l", "--is_local", default=False, help="gets models from Hub", 
            action=argparse.BooleanOptionalAction)
    parser.add_argument("-t", "--is_test", default=False, help="run on low fidelity", 
            action=argparse.BooleanOptionalAction)
    main(parser.parse_args())
