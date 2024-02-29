import torch
import pandas as pd
from tqdm import tqdm
from transformers import pipeline
from argparse import ArgumentParser
import logging
import os
from pathlib import Path

from utils import get_current_timestamp


TIMESTAMP = get_current_timestamp()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"../logs/{TIMESTAMP}.log"),
        logging.StreamHandler()
    ]
)


parser = ArgumentParser()
parser.add_argument("--model_name", dest="model_name", required=True, type=str)
parser.add_argument("--dataset_path", dest="dataset_path", required=True, type=str)
parser.add_argument("--input_field_name", dest="input_field_name", required=True, type=str)
parser.add_argument("--batching", dest="batching", action='store_true', help="Enable or not batching")
parser.add_argument("--batch_size", dest="batch_size", default=8, type=int)


if __name__ == "__main__":
    args = parser.parse_args()
    model_name = args.model_name
    dataset_path = args.dataset_path
    input_field_name = args.input_field_name

    logging.info(f"Using llama model: {model_name}")
    text_generator = pipeline("text-generation", model=model_name, torch_dtype=torch.float16, device_map="auto", return_full_text=False)


    df = pd.read_csv(f"{dataset_path}")
    df["generated_text"] = None

    system_prompt = """You're are a helpful Assistant, who provides empathetic responses to the requests from the speaker. 
        The input contains previous turn of the dialog, where the each utterance is encapsulated with tags <speaker>, or <listener>.
        Be empathetic and precise. Make sure to give responses that make dialogue flow. Keep it short.
        """
    

    for index, row in tqdm(df.iterrows(), desc="Running inference..."):
        prior_dialog = row[input_field_name]

        llama_prompt_template = f"""[INST] <<SYS>>
        {system_prompt}
        <</SYS>>

        {prior_dialog} [/INST]
        """

        if index < 1:
            logging.info("Llama prompt template example...")
            logging.info(llama_prompt_template)
            
        output = text_generator(llama_prompt_template, 
                        max_new_tokens=100)

        generated_text = output[0]["generated_text"]

        df.at[index, 'generated_text'] = generated_text
    
    short_model_name = "-".join(model_name.split("/")[-1].split("-")[:-1]) # "meta-llama/Llama-2-13b-chat-hf" => Llama-2-13b-chat
    dir_path = os.path.dirname(dataset_path)
    file_name = Path(dataset_path).stem
    df.to_csv(f'{dir_path}/{short_model_name}_{file_name}_generated_text.csv', index=False)