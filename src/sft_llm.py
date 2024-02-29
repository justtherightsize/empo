from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig
from accelerate import PartialState
# Load model directly
import torch
from utils import *
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
from transformers import AutoTokenizer, HfArgumentParser, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments


@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """
    per_device_train_batch_size: Optional[int] = field(default=4)
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
        default=None,
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        }
    )
    dataset_name: Optional[str] = field(
        default="stingning/ultrachat",
        metadata={"help": "The preference dataset to use."},
    )
    subset: Optional[bool] = field(
        default=False,
        metadata={"help": "Use subset of data"},
    )
    fp16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables fp16 training."},
    )
    bf16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables bf16 training."},
    )
    packing: Optional[bool] = field(
        default=False,
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
        default="constant",
        metadata={"help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"},
    )
    max_steps: int = field(default=-1, metadata={"help": "How many optimizer update steps to take"})
    warmup_ratio: float = field(default=0, metadata={"help": "Fraction of steps to do a warmup for"})
    save_steps: int = field(default=0.1, metadata={"help": "Save checkpoint every X updates steps."})
    logging_steps: int = field(default=0.1, metadata={"help": "Log every X updates steps."})
    output_dir: str = field(
        default="./results",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )

if __name__ == "__main__":
    # "meta-llama/Llama-2-7b-hf"
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    model_name = script_args.model_name
    compute_dtype = getattr(torch, "float16")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

    if "gemma" in model_name:
        model = AutoModelForCausalLM.from_pretrained(
                                            model_name, use_cache=False, 
                                            quantization_config=quant_config, 
                                            device_map={"": PartialState().process_index},
                                            attn_implementation="sdpa" if not script_args.use_flash_attention_2 else "flash_attention_2",
                                            )
    else: 
        model = AutoModelForCausalLM.from_pretrained(
                                            model_name, use_cache=False, 
                                            quantization_config=quant_config, 
                                            device_map="auto",
                                            attn_implementation="sdpa" if not script_args.use_flash_attention_2 else "flash_attention_2",
                                            )
    # model.config.pretraining_tp = 1
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.add_special_tokens({'additional_special_tokens': [START_OF_TURN, END_OF_TURN]})
    model.resize_token_embeddings(len(tokenizer))

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
        print("Selecting a susbet")
        train_dataset = train_dataset.select(range(500))
        val_dataset = val_dataset.select(range(100))
        
    train_dataset = train_dataset.map(lambda x: {"text": format_chat(x["chat"])}).remove_columns("chat")
    val_dataset = val_dataset.map(lambda x: {"text": format_chat(x["chat"])}).remove_columns("chat")

    print("An example from the dataset")
    print("#" * 100) 
    print(train_dataset["text"][0])
    # TODO: make that configurable
    
    # "../results/sft_llama2"
    training_args = TrainingArguments(
        output_dir=script_args.output_dir,
        num_train_epochs=4,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        optim="paged_adamw_32bit",
        save_total_limit=1,
        save_steps=script_args.save_steps,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=script_args.logging_steps,
        learning_rate=1.4e-5,
        weight_decay=0.001,
        fp16=script_args.fp16,
        bf16=script_args.bf16,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        report_to="wandb"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=script_args.max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
        packing=script_args.packing,
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)

