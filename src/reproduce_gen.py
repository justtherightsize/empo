from peft import PeftModel
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from huggingface_hub import login


# HF login: you have to be logged in and agree to the license of the base
# model: https://huggingface.co/alignment-handbook/zephyr-7b-sft-full
hf_key = "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"
login(hf_key)

# Load tokenizer either from remote
adapter_id = "justtherightsize/zephyr-7b-sft-full124_d270"
base_model_id = "alignment-handbook/zephyr-7b-sft-full"
tokenizer = AutoTokenizer.from_pretrained(adapter_id)

# Prepare dialog and convert to chat template
sys_msg = "You are a friendly assistant, who provides empathetic responses to the user. " \
            "The input contains previous turn of the dialog, where each utterance is prefaced " \
            "with tags <|user|>, or <|assistant|>. Be empathetic and precise. " \
            "Make sure to give responses that make dialogue flow. Avoid repeating the prompt. " \
            "Please respond creatively and expressively to make the responses longer. You can offer advice."

dialog = ["Yeah about 10 years ago I had a horrifying experience. It was 100% their fault but they hit the water barrels and survived. They had no injuries but they almost ran me off the road.", 
        "Did you suffer any injuries?", 
        "No I wasn't hit. It turned out they were drunk. I felt guilty but realized it was his fault."]

dwroles = [{"role": "system", "content": sys_msg}]
for j in range(len(dialog)):
    dwroles.append(
        {"role": "user", "content": dialog[j]} if j % 2 == 0 else
        {"role": "assistant", "content": dialog[j]})
template = tokenizer.apply_chat_template(dwroles, tokenize=False, add_generation_prompt=True)

# Load the big model first & resize embeds, load PEFT model
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
model = PeftModel.from_pretrained(model, adapter_id)

# Instantiate generation pipeline
pipe_gen = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Generate the response
out = pipe_gen(template, return_full_text=False, max_new_tokens=500)[0]['generated_text']
print(out)
