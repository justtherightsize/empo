import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from accelerate import PartialState
# settings for telamon
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch

from transformers import AutoTokenizer, HfArgumentParser, AutoModelForCausalLM, BitsAndBytesConfig, \
    TrainingArguments, \
    AutoConfig, pipeline

from peft import prepare_model_for_kbit_training, PeftModel, PeftConfig

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
        default="./results",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

model_id = script_args.model_name
output_dir = "./results/zephyr-qlora-empathy3"
checkpoint_id = "checkpoint-556"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    trust_remote_code=True
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(output_dir)
model.resize_token_embeddings(len(tokenizer))

config = PeftConfig.from_pretrained(output_dir)
model = PeftModel.from_pretrained(model, output_dir)

# https://discuss.huggingface.co/t/having-trouble-loading-a-fine-tuned-peft-model-codellama-13b-instruct-hf-base/52880
# https://discuss.huggingface.co/t/peft-model-from-pretrained-load-in-8-4-bit/47199/6
# model = prepare_model_for_kbit_training(model)

def fix_model_folder_with_incorrect_vocab_size(model_folder: Path, pad_vocab_size_to_multiple_of: int = 64):
    """If you got bitten by the bug https://github.com/huggingface/transformers/issues/25729, and your model got saved with an incorrect vocab size, this will fix it."""
    config_file = model_folder / "config.json"
    config = AutoConfig.from_pretrained(config_file)
    vocab_size: int = config.vocab_size
    # is vocab_size an int, and is it bigger than 0?
    assert isinstance(vocab_size, int) and vocab_size > 0

    if vocab_size % pad_vocab_size_to_multiple_of != 0:
        print(f"vocab_size ({vocab_size}) is not a multiple of {pad_vocab_size_to_multiple_of}. Fixing it...")
        # fix the vocab_size:
        config.vocab_size = vocab_size + (pad_vocab_size_to_multiple_of - vocab_size % pad_vocab_size_to_multiple_of)
        # save the config file back to the checkpoint folder:
        config.save_pretrained(model_folder)
        print(f"Fixed the vocab_size to {config.vocab_size}")

pipe = pipeline("text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=200,
)

prompt = """<|system|>
You are a friendly assistant, who provides empathetic responses to the user. The input contains previous turn of the dialog, where the each utterance is prefaced with tags <|user|>, or <|assistant|>. Be empathetic and precise. Make sure to give responses that make dialogue flow. Avoid repeating the prompt.</s>

<|user|>
i'm so excited because i'm finally going to visit my parents next month! I didn't see them for 3 years</s>

<|assistant|>"""

print(pipe(prompt)[0]['generated_text'])
