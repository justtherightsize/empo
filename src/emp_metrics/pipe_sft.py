import time
import shutil
import glob
import os
import re
import argparse
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import pandas as pd
from typing import Tuple, List, Dict
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import wandb
from mmlu_all import run_mmlu
from src.emp_metrics.dpo_ps import train_dpo
from src.emp_metrics.run_metrics_on_saved_df import calc_metrics


def load_preprocess_ed(split: str = "test") -> Tuple[pd.DataFrame, List[str], List[str]]:
    dataset = load_dataset("empathetic_dialogues")
    dataset.set_format("pandas")

    test_df = dataset[split][:]
    keyword = 'hit:'
    test_df = test_df[~test_df['utterance'].str.contains(keyword)]
    test_df["prompt"] = test_df["prompt"].apply(lambda u: u.replace("_comma_", ","))
    test_df["utterance"] = test_df["utterance"].apply(lambda u: u.replace("_comma_", ","))

    df_group = test_df.groupby(["conv_id"])
    test_dialogs = []
    test_keys = []
    for name_of_group, contents_of_group in df_group:
        contents_of_group.sort_values("utterance_idx")
        test_dialogs.append(contents_of_group["utterance"].to_list())
        test_keys.append(name_of_group)

    return test_df, test_keys, test_dialogs


def dialog2chat(dialog: List[str], system_message: str = None, user_key: str = "user",
                           assistant_key: str = "assistant") -> List[List[Dict[str,str]]]:
    template = []
    if system_message is not None:
        template.append({"role": "system", "content": system_message})
    for j in range(len(dialog)):
        template.append(
            {"role": user_key, "content": dialog[j]} if j % 2 == 0 else
            {"role": assistant_key, "content": dialog[j]})
    return template


def get_ed_chats(split: str, tokenizer, **kwargs) -> pd.DataFrame:
    _, _, dialogs = load_preprocess_ed(split)
    dias = [dialog2chat(d) for d in dialogs]

    test_tok = [tokenizer.apply_chat_template(x, **kwargs) for x in dias]
    test_tok = [t.replace("<s> <|user|>", "<s>\n<|user|>") for t in test_tok]
    df_final = pd.DataFrame(test_tok, columns=['chat_templates'])
    return df_final


def run_sft():
    with wandb.init():
        config = wandb.config
        import torch
        model_id = config.model_id
        output_dir_base = config.output_dir_base
        args = argparse.Namespace()
        args.base_model = model_id
        args.base_dir = output_dir_base
        args.hf_key_path = config.hf_key_path
        args.is_local = config.is_local
        args.is_test = config.is_test
        args.is_mmlu_subset = config.is_mmlu_subset
        args.test_frac = config.test_frac
        if config.local_adapter != "none":
            run_wo_sft(config, args)
            return
        
        tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="right",
                                                  padding=True)
        tokenizer.add_special_tokens({'pad_token': "[PAD]"})

        val_dataset = Dataset.from_pandas(
            get_ed_chats("validation", tokenizer, tokenize=False,
                         add_generation_prompt=False))
        train_dataset = Dataset.from_pandas(
            get_ed_chats("train", tokenizer, tokenize=False, add_generation_prompt=False))

        if config.test_frac < 0.9999:
            train_dataset = train_dataset.select(
                            range(int(len(train_dataset) * config.test_frac)))
            val_dataset = val_dataset.select(
                          range(int(len(val_dataset) * config.test_frac)))
        
        prev_ords = [int(match.group()) for d in os.listdir(output_dir_base) if
                     (match := re.search(r'\d+$', d))]
        output_dir = output_dir_base + model_id.split("/")[-1] + str(
                1 + max(prev_ords) if len(prev_ords) > 0 else 0)
        print(f"Output dir: {output_dir}")

        response_template = "\n<|assistant|>"
        instruction_template = "\n<|user|>"
        collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template,
                                                   response_template=response_template,
                                                   tokenizer=tokenizer,
                                                   mlm=False)

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            attn_implementation=config.attn_implementation,
            device_map="auto",
            use_cache=False)
        model.resize_token_embeddings(len(tokenizer))

        lora_config = LoraConfig(
            r=config.r,
            target_modules=config.target_modules,
            bias=config.bias,
            task_type="CAUSAL_LM",
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            modules_to_save=["embed_tokens", "lm_head"]
        )

        training_arguments = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=config.per_device_train_batch_size,
            per_device_eval_batch_size=config.per_device_eval_batch_size,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            optim="paged_adamw_32bit",
            num_train_epochs=config.num_train_epochs,
            logging_steps=0.05,
            learning_rate=config.learning_rate,
            max_grad_norm=config.max_grad_norm,
            warmup_ratio=config.warmup_ratio,
            lr_scheduler_type="cosine",
            gradient_checkpointing=True,
            fp16=False,
            bf16=True,
            report_to="wandb",
            load_best_model_at_end=True,
            save_total_limit=1
        )

        trainer = SFTTrainer(
            model=model,
            args=training_arguments,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            peft_config=lora_config,
            packing=False,
            dataset_text_field="chat_templates",
            tokenizer=tokenizer,
            max_seq_length=config.cutoff_len,
            data_collator=collator,
        )

        train_result = trainer.train()
        trainer.save_model(output_dir)
        print(f"1.-----Saving SFT to: {output_dir}--------")
        del model
        del tokenizer
        del train_result
        del trainer
        checkpt_dirs = glob.glob(output_dir + "/checkpoint-*")
        for dir_path in checkpt_dirs:
            shutil.rmtree(dir_path)
        time.sleep(15)

        # run mmlu evaluation and preds generation
        args.adapter = output_dir.split("/")[2]
        run_eval(args, "sft")

        if config.dpo:
            args.dpo_name = config.dpo_name
            dpo_out_dir = run_dpo(args)
            args.adapter = dpo_out_dir
            # import ipdb; ipdb.set_trace()
            run_eval(args, "dpo")


def run_wo_sft(config, args):
    args.adapter = config.local_adapter
    args.test_frac = config.test_frac
    # run_eval(args, "sft")

    if config.dpo:
        args.dpo_name = config.dpo_name
        dpo_out_dir = run_dpo(args)
        args.adapter = dpo_out_dir
        # import ipdb; ipdb.set_trace()
        # time.sleep(15)
        # run_eval(args, "dpo")


def run_eval(args: argparse.Namespace, run_prefix: str = "") -> None:
    # import ipdb; ipdb.set_trace()
    args.run_pref = run_prefix
    mmlu_res = run_mmlu(args)  # ~>"path_preds", "path_mmlu"

    # run metrics on saved df
    calc_metrics(mmlu_res["path_preds"],
                 args.adapter if args.is_local else args.base_model,
                 ["bertscore", "epitome"], run_prefix)


def run_dpo(args: argparse.Namespace) -> str:
    dpo_out_dir = train_dpo(args.base_model, args.adapter, args.base_dir,
                            args.dpo_name)
    return dpo_out_dir


