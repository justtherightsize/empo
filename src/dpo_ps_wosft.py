import argparse
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import wandb
TEST = False
if TEST:
    wandb.init(mode="disabled")

from peft import LoraConfig, PeftModel
from alignment import DPOConfig
from trl import DPOTrainer
from src.ed_load import get_ed_for_dpo

from huggingface_hub import login
from pathlib import Path
login(Path('/home/xshared/.huggingface/mistral').read_text().strip())

from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from numpy import percentile


def train_dpo(base_model_id, model_id, output_dir_base, new_name):
    output_dir = output_dir_base + model_id
    dpo_output_dir = output_dir_base + model_id + "_" + new_name

    tokenizer = AutoTokenizer.from_pretrained(output_dir)

    # load datasets
    sys_msg = "You are a friendly assistant, who provides empathetic responses to the user. " \
              "The input contains previous turn of the dialog, where the each utterance is prefaced " \
              "with tags <|user|>, or <|assistant|>. Be empathetic and precise. Make sure to give " \
              "responses that make dialogue flow. Avoid repeating the prompt."

    train_dataset = get_ed_for_dpo("train", tokenizer, sys_msg=sys_msg, tokenize=False,
                                    add_generation_prompt=True)
    eval_dataset = get_ed_for_dpo("test", tokenizer, sys_msg=sys_msg, tokenize=False,
                                    add_generation_prompt=True)

    # find the p95 length of the prompt
    prompt_length = int(percentile([len(tokenizer(x)["input_ids"]) for x in train_dataset["prompt"]], 95))
    max_seq_length_chosen = int(percentile([len(tokenizer(x["prompt"] + x["chosen"])["input_ids"]) for x in train_dataset], 95))
    max_seq_length_rejected = int(percentile([len(tokenizer(x["prompt"] + x["rejected"])["input_ids"]) for x in train_dataset], 95))
    max_seq_length = max(max_seq_length_chosen, max_seq_length_rejected)

    # filter datasets to remove samples that are too long
    train_dataset = train_dataset.filter(lambda x: len(tokenizer(x["prompt"] + x["chosen"])["input_ids"]) <= max_seq_length)
    eval_dataset = eval_dataset.filter(lambda x: len(tokenizer(x["prompt"] + x["chosen"])["input_ids"]) <= max_seq_length)

    # Up the lengths to next multiple of 2
    prompt_length = ((prompt_length + 1) // 2) * 2
    max_seq_length = ((max_seq_length + 1) // 2) * 2

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        # load_in_4bit=True,
        quantization_config=bnb_config,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_cache=False
    )
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, output_dir, is_trainable=True)

    # Load the adapter a second time, with a different name, which will be our reference model.
    #   -> doesnt work, load the model twice
    # model.load_adapter(output_dir, adapter_name="reference")

    ref_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        # load_in_4bit=True,
        quantization_config=bnb_config,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_cache=False
    )
    ref_model.resize_token_embeddings(len(tokenizer))
    ref_model = PeftModel.from_pretrained(ref_model, output_dir, is_trainable=True)

    training_args = DPOConfig(
        output_dir=dpo_output_dir,               # directory to save and repository id
        num_train_epochs=1,                     # number of training epochs
        per_device_train_batch_size=8,         # batch size per device during training
        per_device_eval_batch_size=4,           # batch size for evaluation
        gradient_accumulation_steps=1,          # number of steps before performing a backward/update pass
        gradient_checkpointing=True,            # use gradient checkpointing to save memory
        optim="adamw_torch_fused",              # use fused adamw optimizer
        learning_rate=5e-5,                     # 10x higher LR than QLoRA paper
        max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
        warmup_ratio=0.1,                       # warmup ratio based on QLoRA paper
        lr_scheduler_type="cosine",             # use cosine learning rate scheduler
        logging_steps=25,                       # log every 25 steps
        save_steps=500,                         # when to save checkpoint
        save_total_limit=2,                     # limit the total amount of checkpoints
        evaluation_strategy="steps",            # evaluate every 1000 steps
        eval_steps=700,                         # when to evaluate
        bf16=True,                              # use bfloat16 precision
        tf32=True,                              # use tf32 precision
        push_to_hub=False,                      # push model to hub
        report_to="wandb",                # report metrics to tensorboard
    )

    dpo_args = {
        "beta": 0.1,                            # The beta factor in DPO loss. Higher beta means less divergence
        "loss_type": "sigmoid"                  # The loss type for DPO.
    }
    trainer = DPOTrainer(
                model,
                ref_model,
                # model_adapter_name="train2", -> doesnt work
                # ref_adapter_name="reference", -> doesnt work
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                max_length=max_seq_length,
                max_prompt_length=prompt_length,
                beta=dpo_args["beta"],
                loss_type=dpo_args["loss_type"],
                model_init_kwargs=None,
                ref_model_init_kwargs=None
            )

    trainer.train()
    trainer.save_model(dpo_output_dir)


def main(args: argparse.Namespace) -> None:
    train_dpo(args.base_model, args.adapter, args.base_dir, args.new_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu", default="1", help="not implemented")
    parser.add_argument("-bm", "--base_model", default="alignment-handbook/zephyr-7b-sft-lora",
                        help="base model name")
    parser.add_argument("-a", "--adapter", help="adapter name")
    parser.add_argument("-d", "--base_dir", default="./results/", help="base dir with saved models")
    parser.add_argument("-n", "--new_name", help="save name")

    main(parser.parse_args())
