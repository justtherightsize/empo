import torch
import pandas as pd
from tqdm import tqdm
from transformers import pipeline
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--model_name", dest="model_name", required=True, type=str)
parser.add_argument("--root_path", dest="root_path", default="../data/empathy_datasets/eqt_human_msplit", type=str)

if __name__ == "__main__":
    args = parser.parse_args()
    model_name = args.model_name
    root_path = args.root_path

    print(f"Using llama model: {model_name}")
    text_generator = pipeline("text-generation", model=model_name, torch_dtype=torch.float16, device_map="auto", return_full_text=False)


    df = pd.read_csv(f"{root_path}/subset.csv")
    df["generated_text"] = None

    system_prompt = """You're are a helpful Assistant, who provides empathetic responses to the requests from the speaker. 
        The input contains previous turn of the dialog, where the each utterance is encapsulated with tags <speaker>, or <listener>.
        Be empathetic and precise. Make sure to give responses that make dialogue flow. Keep it short.
        """
    
    for index, row in tqdm(df.iterrows(), desc="Running inference..."):
        prior_dialog = row["prior_dialog"]

        llama_prompt_template = f"""[INST] <<SYS>>
        {system_prompt}
        <</SYS>>

        {prior_dialog} [/INST]
        """

        if index < 1:
            print("Prompt example")
            print("#" * 100)
            print("#" * 100)
            print(llama_prompt_template)
            print("#" * 100)
            print("#" * 100)

        output = text_generator(llama_prompt_template, 
                        max_new_tokens=100)

        generated_text = output[0]["generated_text"]

        df.at[index, 'generated_text'] = generated_text
    
    short_model_name = "-".join(model_name.split("/")[-1].split("-")[:-1]) # "meta-llama/Llama-2-13b-chat-hf" => Llama-2-13b-chat
    df.to_csv(f'{root_path}/{short_model_name}_subset_generated_text.csv', index=False)