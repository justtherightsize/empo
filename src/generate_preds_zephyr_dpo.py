import wandb

TEST = True
if TEST:
    wandb.init(mode="disabled")

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
from tqdm import tqdm
from src.ed_load import get_ed_for_generation
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from transformers import AutoTokenizer
from peft import PeftModel, PeftConfig
from datasets import Dataset

def generate_preds(base_model_id, model_id, output_dir_base):
    output_dir = output_dir_base + model_id

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # tokenizer = AutoTokenizer.from_pretrained(output_dir, padding_side="left")
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

    # When the adapter is saved in a named directory
    # model.load_adapter(output_dir, "train2")
    # model.set_adapter("train2")

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)

    sys_msg = "You are a friendly assistant, who provides empathetic responses to the user. " \
              "The input contains previous turn of the dialog, where the each utterance is prefaced " \
              "with tags <|user|>, or <|assistant|>. Be empathetic and precise. Make sure to give " \
              "responses that make dialogue flow. Avoid repeating the prompt."
    test_df = get_ed_for_generation("test", tokenizer, sys_msg=sys_msg, tokenize=False,
                                    add_generation_prompt=True)

    # def tprint(mdl):
    #     pipe = pipeline("text-generation", model=mdl, tokenizer=tokenizer, max_new_tokens=200)
    #     a = test_df.head(5)
    #     df = a.copy()
    #     # serie = df["chat_templates"].to_list()
    #     #
    #     # def data():
    #     #     for v in serie:
    #     #         yield v
    #     #
    #     # for out in pipe(data(), return_full_text=False, batch_size=8):
    #     #     print(out[0]["generated_text"])
    #
    #     for index, r in df.iterrows():
    #         print(pipe(r["chat_templates"], return_full_text=False)[0]["generated_text"])
    # tprint(model)


    # Generate replies and save to csv
    gens = []
    for index, r in tqdm(test_df.iterrows()):
        out = pipe(r["chat_templates"], return_full_text=False)[0]['generated_text']
        assert "~" not in out, f"Char ~ found in gen {index}: {out}"
        gens.append(out)
    test_df["gens"] = gens
    pth = f"results/preds_{model_id}.txt"
    test_df.to_csv(pth, sep="~")



def main(args: argparse.Namespace) -> None:
    base_model_id = args.base_model
    model_id = args.adapter #"zephyr-qlora-empathy3_dpo2"
    output_dir_base = args.base_dir #"./results/"
    generate_preds(base_model_id, model_id, output_dir_base)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu", default="1", help="not implemented")
    parser.add_argument("-bm", "--base_model", default="alignment-handbook/zephyr-7b-sft-lora",
                        help="base model name")
    parser.add_argument("-a", "--adapter", help="adapter name")
    parser.add_argument("-d", "--base_dir", default="./results/", help="base dir with saved models")

    main(parser.parse_args())
