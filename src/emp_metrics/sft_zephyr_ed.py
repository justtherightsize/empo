import os
from dataclasses import dataclass, field
from typing import Optional
from accelerate import PartialState
# settings for telamon
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch

from transformers import AutoTokenizer, HfArgumentParser, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer
import pandas as pd

from joan_utils import START_OF_TURN, END_OF_TURN, convert_to_dataset, format_chat

# import wandb
# wandb.init(mode="disabled")


@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """
    per_device_train_batch_size: Optional[int] = field(default=6)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=4)
    learning_rate: Optional[float] = field(default=2e-4)
    max_grad_norm: Optional[float] = field(default=0.3)
    weight_decay: Optional[int] = field(default=0.001)
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=8)
    max_seq_length: Optional[int] = field(default=2048)
    model_name: Optional[str] = field(
        # default="TinyLlama/TinyLlama-1.1B-Chat-v0.1",
        default="alignment-handbook/zephyr-7b-sft-lora",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        }
    )
    dataset_name: Optional[str] = field(
        # default="stingning/ultrachat",
        default="data/empathy_datasets/empathetic_dialogues",
        metadata={"help": "The preference dataset to use."},
    )
    subset: Optional[bool] = field(
        default=True,
        metadata={"help": "Use subset of data"},
    )
    fp16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables fp16 training."},
    )
    bf16: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables bf16 training."},
    )
    packing: Optional[bool] = field(
        default=True,
        metadata={"help": "Use packing dataset creating."},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )
    use_flash_attention_2: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables Flash Attention 2."},
    )
    optim: Optional[str] = field(
        default="paged_adamw_32bit",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: str = field(
        # default="constant",
        default="cosine",
        metadata={"help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"},
    )
    max_steps: int = field(default=-1, metadata={"help": "How many optimizer update steps to take"})
    warmup_ratio: float = field(default=0, metadata={"help": "Fraction of steps to do a warmup for"})
    save_steps: int = field(default=0.1, metadata={"help": "Save checkpoint every X updates steps."})
    logging_steps: int = field(default=0.1, metadata={"help": "Log every X updates steps."})
    output_dir: str = field(
        default="./results/zephyr-qlora-empathy2",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

# Load the GG model - this is the local one, update it to the one on the Hub
model_id = script_args.model_name

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    quantization_config=quantization_config, 
    attn_implementation="sdpa" if not script_args.use_flash_attention_2 else "flash_attention_2",
    device_map={"": PartialState().process_index},
    use_cache=False
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token_id = tokenizer.eos_token_id
# tokenizer.add_special_tokens({'additional_special_tokens': [START_OF_TURN, END_OF_TURN]})
# model.resize_token_embeddings(len(tokenizer))

lora_config = LoraConfig(
    r=script_args.lora_r,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
    lora_alpha=script_args.lora_alpha,
    lora_dropout=script_args.lora_dropout
)

train_dataset = convert_to_dataset(pd.read_csv(f"{script_args.dataset_name}/train.csv"))
val_dataset = convert_to_dataset(pd.read_csv(f"{script_args.dataset_name}/val.csv"))

if script_args.subset:
    train_dataset = train_dataset.select(range(128))
    val_dataset = val_dataset.select(range(36))
    
train_dataset = train_dataset.map(lambda x: {"text": format_chat(x["chat"])}).remove_columns("chat")
val_dataset = val_dataset.map(lambda x: {"text": format_chat(x["chat"])}).remove_columns("chat")

# print("An example from the dataset")
# print("#" * 100)
# print(train_dataset["text"][0])

training_arguments = TrainingArguments(
    output_dir=script_args.output_dir,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    optim=script_args.optim,
    num_train_epochs=3,
    # save_steps=script_args.save_steps,
    logging_steps=script_args.logging_steps,
    learning_rate=script_args.learning_rate,
    max_grad_norm=script_args.max_grad_norm,
    # max_steps=script_args.max_steps,
    warmup_ratio=script_args.warmup_ratio,
    lr_scheduler_type=script_args.lr_scheduler_type,
    gradient_checkpointing=script_args.gradient_checkpointing,
    fp16=script_args.fp16,
    bf16=script_args.bf16,
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
    packing=script_args.packing,
    dataset_text_field="text",
    tokenizer=tokenizer,
    max_seq_length=script_args.max_seq_length,
)

train_result = trainer.train()
# trainer.save_state()
trainer.save_model(script_args.output_dir)
metrics = train_result.metrics
print("metrics ------------------------------------------")
print(metrics)
# max_train_samples = training_args.max_train_samples if training_args.max_train_samples is not None else len(train_dataset)
# metrics["train_samples"] = min(max_train_samples, len(train_dataset))
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)

# trainer.save_model(script_args.output_dir)

