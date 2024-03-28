from itertools import chain
from pathlib import Path
from typing import Union, Tuple, Any, List, Dict

import pandas
import pandas as pd
from datasets import load_dataset, Dataset


# offline ED
# ED_PTH = 'data/empathy_datasets/empathetic_dialogues'
# df = pd.read_csv(Path(ED_PTH, 'test.csv'))

def load_preprocess_ed(split: str = "test") -> Tuple[pandas.DataFrame, List[str], List[str]]:
    """
   ED from HuggingFace. Unescapes the commas. Groups the dialogs by keys. Sorts the groups by utterance_idx.
   @param split: one of {tran, val, test}
   @return: the dataframe, keys of dilogs, utterances of dialogs
   """
    dataset = load_dataset("empathetic_dialogues")
    dataset.set_format("pandas")

    test_df = dataset[split][:]
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


def get_progressive_chunks(dialog: List[str], system_message: str = None, user_key: str = "user",
                           assistant_key: str = "assistant") -> List[List[Dict[str,str]]]:
    """
    Extract progressively longer chunks of a dialog. Return them in a chat template as in
    https://huggingface.co/docs/transformers/main/en/chat_templating
    @param dialog: List of ordered utterances. Author1: even indices
    @param system_message: Prepends the message as a system promt. Use for generation only.
    @param user_key: key for the even idx prompts
    @param assistant_key: key for the odd idx prompts
    @return: chunks of len 2..n in the chat template
    """
    assert len(dialog) >= 1, "Empty dialog."

    chunks = []
    for i in range(0, len(dialog)):
        template = []
        if system_message is not None:
            template.append({"role": "system", "content": system_message})
        for j in range(0, i+1):
            template.append(
                {"role": user_key, "content": dialog[j]} if j % 2 == 0 else
                {"role": assistant_key, "content": dialog[j]})
        chunks.append(template)
    return chunks


def get_chunked_dataset(chunks: List[List[Dict[str,str]]], **kwargs) -> Dataset:
    """
    Produce a Dataset class off the list of chunks
    @param chunks: chunks in the chat template https://huggingface.co/docs/transformers/main/en/chat_templating
    @return: Dataset of chunks
    """
    dataset = Dataset.from_dict({"chat": chunks})
    dataset = dataset.map(lambda x: {
        "formatted_chat": tokenizer.apply_chat_template(x["chat"], **kwargs)})
    return dataset


# Usage
if __name__ == '__main__':
    df, keys, dialogs = load_preprocess_ed("test")
    print(f"{keys[0]}: {''.join(dialogs[0])}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    checkpoint = "HuggingFaceH4/zephyr-7b-beta"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    # Example 1: single message
    messages = ["How many helicopters can a human eat in one sitting?"]
    system_message = "You are a friendly chatbot who always responds in the style of a pirate"
    dialog_chunks = get_progressive_chunks(messages)
    tokenized_chat = tokenizer.apply_chat_template(dialog_chunks[0], tokenize=True, add_generation_prompt=True,
                                                   return_tensors="pt")
    print(tokenizer.decode(tokenized_chat[0]))

    # Example 2: multiple chunks
    system_message2 = "You're are a helpful Assistant, who provides empathetic responses to the requests from the " \
                     "speaker. The input contains previous turn of the dialog, where the each utterance is " \
                     "identified by the tags user or assistant. Be empathetic and precise. Make sure to give " \
                     "responses that make dialogue flow. Keep it short."
    dc2 = list(chain.from_iterable(get_progressive_chunks(value) for value in dialogs[0:2]))
    ds = get_chunked_dataset(dc2, tokenize=False, add_generation_prompt=False)
    print(ds['formatted_chat'][0])


