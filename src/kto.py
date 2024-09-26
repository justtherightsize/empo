import argparse
import torch
import wandb
from peft import LoraConfig, PeftModel
from alignment import DPOConfig
from trl import DPOTrainer
from src.ed_load import get_ed_for_dpo
from huggingface_hub import login
from pathlib import Path
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from numpy import percentile


def train_kto(base_model_id, model_id, output_dir_base, new_name):
    output_dir = output_dir_base + model_id
    dpo_output_dir = output_dir_base + model_id + "_" + new_name
    config = wandb.config

    # import ipdb; ipdb.set_trace()
    tokenizer = AutoTokenizer.from_pretrained(output_dir)

    # load datasets
    sys_msg = "You are a friendly assistant, who provides empathetic responses to the user. " \
              "The input contains previous turn of the dialog, where the each utterance is prefaced " \
              "with tags <|user|>, or <|assistant|>. Be empathetic and precise. Make sure to give " \
              "responses that make dialogue flow. Avoid repeating the prompt."

    train_dataset = get_ed_for_dpo("train", tokenizer, sys_msg=sys_msg,
                                   tokenize=False, add_generation_prompt=True)
    eval_dataset = get_ed_for_dpo("test", tokenizer, sys_msg=sys_msg,
                                  tokenize=False, add_generation_prompt=True)

    # find the p95 length of the prompt
    prompt_length = int(percentile(
        [len(tokenizer(x)["input_ids"]) for x in train_dataset["prompt"]], 95))
    max_seq_length_chosen = int(percentile([len(tokenizer(x["prompt"] + x["chosen"])["input_ids"]) for x in train_dataset], 95))
    max_seq_length_rejected = int(percentile([len(tokenizer(x["prompt"] + x["rejected"])["input_ids"]) for x in train_dataset], 95))
    max_seq_length = max(max_seq_length_chosen, max_seq_length_rejected)

    if config.test_frac < 0.9999:
        train_dataset = train_dataset.select(
                        range(int(len(train_dataset) * config.test_frac)))
        eval_dataset = eval_dataset.select(
                      range(int(len(eval_dataset) * config.test_frac)))

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
        quantization_config=bnb_config,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_cache=False
    )
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, output_dir, is_trainable=True)

    # Load the adapter a second time, with a different name, which will be 
    # our reference model.
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
        output_dir=dpo_output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        learning_rate=5e-5,
        max_grad_norm=0.3,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=25,
        save_steps=500,
        save_total_limit=2,
        evaluation_strategy="steps",
        eval_steps=700,
        bf16=True,
        tf32=True,
        push_to_hub=False,
        report_to="wandb"
    )
    dpo_args = {
        "beta": 0.1,  # Higher beta means less divergence
        "loss_type": "sigmoid"
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
    print(f"5.-----Saving DPO to: {dpo_output_dir}--------")
    return dpo_output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu", default="1", help="not implemented")
    parser.add_argument("-bm", "--base_model",
                        default="alignment-handbook/zephyr-7b-sft-lora",
                        help="base model name")
    parser.add_argument("-a", "--adapter", help="adapter name")
    parser.add_argument("-d", "--base_dir", default="./results/",
                        help="base dir with saved models")
    parser.add_argument("-n", "--new_name", help="save name")

    ARGS = parser.parse_args()
    train_kto(ARGS.base_model, ARGS.adapter, ARGS.base_dir, ARGS.new_name)
