import wandb
TEST = True
if TEST:
    wandb.init(mode="disabled")

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


import re
from accelerate import PartialState
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, pipeline, \
    LlamaTokenizer
from peft import LoraConfig
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import pandas as pd
from typing import Tuple, List, Dict
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from peft import PeftModel, PeftConfig


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
    df_final = pd.DataFrame(test_tok, columns=['chat_templates'])
    return df_final


model_id = "alignment-handbook/zephyr-7b-sft-lora"
output_dir_base = "./results/"
output_dir = output_dir_base + model_id.split("/")[-1] + str(1 + max(
    [int(match.group()) for d in os.listdir(output_dir_base) if (match := re.search(r'\d+$', d))]))
print(f"Output dir: {output_dir}")

# tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="right", padding=True, use_fast=False,
#                                           trust_remote_code=True, clean_up_tokenization_spaces=True)
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="right", padding=True)
tokenizer.add_special_tokens({'pad_token': "[PAD]"})

val_dataset = Dataset.from_pandas(
    get_ed_chats("validation", tokenizer, tokenize=False, add_generation_prompt=False))
train_dataset = Dataset.from_pandas(
    get_ed_chats("train", tokenizer, tokenize=False, add_generation_prompt=False))

if TEST:
    train_dataset = train_dataset.select(range(2048))
    val_dataset = val_dataset.select(range(48))

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
    attn_implementation= "sdpa",#"flash_attention_2",
    device_map="auto",#{"": PartialState().process_index},
    use_cache=False)
model.resize_token_embeddings(len(tokenizer))

lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
    lora_alpha=16,
    lora_dropout=0.1,
    modules_to_save=["embed_tokens", "lm_head"]
)

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    num_train_epochs=1,
    logging_steps=0.1,
    learning_rate=2e-4,
    max_grad_norm=0.3,
    warmup_ratio=0,
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
    max_seq_length=384,
    data_collator=collator,
)

train_result = trainer.train()
trainer.save_model(output_dir)
# model.save_pretrained(output_dir, safe_serialization=False, save_embedding_layers=True)
del(model)
del(tokenizer)

# inference ----------------------------------------------
# inference_model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     quantization_config=quantization_config,
#     trust_remote_code=True
# )
# inference_model.config.use_cache = False
#
# inference_tokenizer = AutoTokenizer.from_pretrained(output_dir)
# inference_model.resize_token_embeddings(len(inference_tokenizer))
#
# config = PeftConfig.from_pretrained(output_dir)
# inference_model = PeftModel.from_pretrained(inference_model, output_dir)
#
# pipe = pipeline("text-generation",
#                 model=inference_model,
#                 tokenizer=inference_tokenizer,
#                 max_new_tokens=200,
# )
#
# prompt = """<|system|>
# You are a friendly assistant, who provides empathetic responses to the user. The input contains previous turn of the dialog, where the each utterance is prefaced with tags <|user|>, or <|assistant|>. Be empathetic and precise. Make sure to give responses that make dialogue flow. Avoid repeating the prompt.</s>
#
# <|user|>
# i'm so excited because i'm finally going to visit my parents next month! I didn't see them for 3 years</s>
#
# <|assistant|>"""
#
# print(pipe(prompt)[0]['generated_text'])
