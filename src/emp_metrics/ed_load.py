from itertools import chain
from pathlib import Path
from typing import Union, Tuple, Any, List, Dict

import pandas as pd
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

# offline ED
# ED_PTH = 'data/empathy_datasets/empathetic_dialogues'
# df = pd.read_csv(Path(ED_PTH, 'test.csv'))
from src.emp_metrics.emotions import get_opposite_ed_keys


def load_preprocess_ed(split: str = "test") -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
   ED from HuggingFace. Unescapes the commas. Groups the dialogs by keys. Sorts the groups by utterance_idx.
   @param split: one of {train, validation, test}
   @return: the dataframe, keys of dilogs, utterances of dialogs
   TODO: remove whole dialogs which contain the malformed keyword
   """
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
        if len(contents_of_group) < 2:
            continue
        contents_of_group.sort_values("utterance_idx")
        test_dialogs.append(contents_of_group["utterance"].to_list())
        test_keys.append(name_of_group[0])

    return test_df, test_keys, test_dialogs


def get_progressive_chunks(dialog: List[str], system_message: str = None, user_key: str = "user",
                           assistant_key: str = "assistant") -> List[List[Dict[str,str]]]:
    """
    Extract progressively longer chunks of a dialog. Return them in a chat template as in
    https://huggingface.co/docs/transformers/main/en/chat_templating
    @param dialog: List of ordered utterances. Author1: even indices
    @param system_message: Prepends the message as a system prompt. Use for generation only.
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


def dialog2chat(dialog: List[str], system_message: str = None, user_key: str = "user",
                           assistant_key: str = "assistant") -> List[List[Dict[str,str]]]:
    """
    Return given dialog in a chat template as in
    https://huggingface.co/docs/transformers/main/en/chat_templating
    @param dialog: List of ordered utterances. Author1: even indices
    @param system_message: Prepends the message as a system prompt. Use for generation only.
    @param user_key: key for the even idx prompts
    @param assistant_key: key for the odd idx prompts
    @return: dialog in the chat template
    """
    assert len(dialog) >= 1, "Empty dialog."

    template = []
    if system_message is not None:
        # template.append({"role": "system", "content": system_message})
        raise NotImplemented()
    for j in range(len(dialog)):
        template.append(
            {"role": user_key, "content": dialog[j]} if j % 2 == 0 else
            {"role": assistant_key, "content": dialog[j]})
    return template


def get_ed(split: str, tokenizer, **kwargs) -> Dataset:
    """Pipeline for getting an non-tokenized ED split in the chat template. Whole dialogs are used.
    @param split: one of {train, validation, test}
    @param tokenizer: hf tokenizer
    @return: Dataset in chat template
    """
    _, _, dialogs = load_preprocess_ed(split)
    dias = [dialog2chat(d) for d in dialogs]


    test_tok = [tokenizer.apply_chat_template(x, **kwargs) for x in dias]
    df_final = pd.DataFrame(test_tok, columns=['chat_templates'])

    dataset = Dataset.from_list(df_final['chat_templates'].apply(
        lambda x: tokenizer(x, return_length=True)).to_list())
    return dataset


def prep4generation(dialogs, sys_msg=None, user_key: str = "user", assistant_key: str = "assistant"):
    odd_dias = []
    gen_targets = []
    prevs = []
    system_message = {"role": "system", "content": sys_msg}
    for d in dialogs:
        if len(d) % 2 == 0 and d[-1]["role"] == assistant_key:
            odd_dias.append(d[:-1] if sys_msg is None else [system_message] + d[:-1])
            gen_targets.append(d[-1]["content"])
            prevs.append(d[-2]["content"])
        elif len(d) % 2 == 1 and d[-1]["role"] == user_key and len(d) >= 3:
            odd_dias.append(d[:-2] if sys_msg is None else [system_message] + d[:-2])
            gen_targets.append(d[-2]["content"])
            prevs.append(d[-3]["content"])
        else:
            print(len(d))
            # raise ValueError(d)
    return odd_dias, gen_targets, prevs


def get_ed_for_generation(split: str, tokenizer, sys_msg=None, **kwargs) -> pd.DataFrame:
    """Pipeline for getting an non-tokenized ED split in the chat template for generation.
    @param split: one of {train, validation, test}
    @param tokenizer: hf tokenizer
    @return: Dataset in chat template
    """
    _, _, dialogs = load_preprocess_ed(split)
    dias = [dialog2chat(d) for d in dialogs]
    odd_dias, gen_targets, prevs = prep4generation(dias, sys_msg)

    test_tok = [tokenizer.apply_chat_template(x, **kwargs) for x in odd_dias]
    df_final = pd.DataFrame(test_tok, columns=['chat_templates'])
    df_final['gen_targets'] = gen_targets
    df_final['prevs'] = prevs
    return df_final


def get_ed_for_dpo(split: str, tokenizer, sys_msg=None, **kwargs) -> Dataset:
    """Pipeline for getting an non-tokenized ED split in the chat template for dpo.
    @param split: one of {train, validation, test}
    @param tokenizer: hf tokenizer
    @return: Dataset in chat template
    """
    df, keys, dialogs = load_preprocess_ed(split)
    k_dias = {k: dialog2chat(d) for k,d in zip(keys, dialogs)}

    odd_dias, chosen, _ = prep4generation(k_dias.values(), sys_msg)
    chosen = [[{"role": "assistant", "content": c}] for c in chosen]

    k_msg = {k:c for k,c in zip(keys, chosen)}
    opposite_keys = get_opposite_ed_keys(df)
    opposite_keys2 = get_opposite_ed_keys(df)
    opposite_keys3 = get_opposite_ed_keys(df)

    # idk why i put this here... 
    # bad = [r for r in opposite_keys.values() if r not in k_msg] 

    k_rejected = {k: k_msg[r] for k,r in opposite_keys.items() if r in k_msg}
    k_rejected2 = {k: k_msg[r] for k,r in opposite_keys2.items() if r in k_msg}
    k_rejected3 = {k: k_msg[r] for k,r in opposite_keys3.items() if r in k_msg}

    assert len(keys) == len(chosen) and len(keys) == len(odd_dias)
    prefs = [{"prompt": tokenizer.apply_chat_template(p, tokenize=False),
              "chosen": tokenizer.apply_chat_template(c, tokenize=False),
              "rejected": tokenizer.apply_chat_template(k_rejected[r], tokenize=False)}
             for p,c,r in zip(odd_dias, chosen, keys) if r in k_rejected]
    prefs2 = [{"prompt": tokenizer.apply_chat_template(p, tokenize=False),
              "chosen": tokenizer.apply_chat_template(c, tokenize=False),
              "rejected": tokenizer.apply_chat_template(k_rejected2[r], tokenize=False)}
             for p,c,r in zip(odd_dias, chosen, keys) if r in k_rejected2]
    prefs3 = [{"prompt": tokenizer.apply_chat_template(p, tokenize=False),
              "chosen": tokenizer.apply_chat_template(c, tokenize=False),
              "rejected": tokenizer.apply_chat_template(k_rejected3[r], tokenize=False)}
             for p,c,r in zip(odd_dias, chosen, keys) if r in k_rejected3]

    allprefs = [] # put them all in one batch as in the hf docs
    for a,b,c in zip(prefs, prefs2, prefs3):
        allprefs.append(a)
        allprefs.append(b)
        allprefs.append(c)
    res = Dataset.from_pandas(pd.DataFrame(data=allprefs))
    return res


def get_ed_for_kto(split: str, tokenizer, sys_msg=None, **kwargs) -> Dataset:
    """Pipeline for getting an non-tokenized ED split
    in the chat template for kto.
    @param split: one of {train, validation, test}
    @param tokenizer: hf tokenizer
    @return: Dataset in chat template
    """
    df, keys, dialogs = load_preprocess_ed(split)
    k_dias = {k: dialog2chat(d) for k,d in zip(keys, dialogs)}

    odd_dias, chosen, _ = prep4generation(k_dias.values(), sys_msg)
    chosen = [[{"role": "assistant", "content": c}] for c in chosen]

    k_msg = {k:c for k,c in zip(keys, chosen)}
    opposite_keys = get_opposite_ed_keys(df)

    bad = [r for r in opposite_keys.values() if r not in k_msg]

    k_rejected = {k: k_msg[r] for k,r in opposite_keys.items() if r in k_msg}

    assert len(keys) == len(chosen) and len(keys) == len(odd_dias)
    prefs = [{"prompt": tokenizer.apply_chat_template(p, tokenize=False),
              "completion": tokenizer.apply_chat_template(c, tokenize=False),
              "label": True}
             for p, c, r in zip(odd_dias, chosen, keys) if r in k_rejected]
    rejes = [{"prompt": tokenizer.apply_chat_template(p, tokenize=False),
              "completion": tokenizer.apply_chat_template(
                  k_rejected[r], tokenize=False),
              "label": False}
             for p, c, r in zip(odd_dias, chosen, keys) if r in k_rejected]
    kto_data = []
    for pr, rj in zip(prefs, rejes):
        kto_data.append(pr)
        kto_data.append(rj)
    res = Dataset.from_pandas(pd.DataFrame(data=kto_data))
    # import ipdb; ipdb.set_trace()
    return res


def get_ed_chats(split: str, tokenizer, **kwargs) -> pd.DataFrame:
    """Pipeline for getting an non-tokenized ED split in the chat template. Whole dialogs are used.
    @param split: one of {train, validation, test}
    @param tokenizer: hf tokenizer
    @return: Dataset in chat template
    """
    _, _, dialogs = load_preprocess_ed(split)
    dias = [dialog2chat(d) for d in dialogs]

    test_tok = [tokenizer.apply_chat_template(x, **kwargs) for x in dias]
    df_final = pd.DataFrame(test_tok, columns=['chat_templates'])
    return df_final


def get_ed_chat_format(split: str):
    """Get non-tokenized ED split in chat template as dictionaries. Whole dialogs are used.
    @param split: one of {train, validation, test}
    @return: Dataset in chat template
    """
    _, _, dialogs = load_preprocess_ed(split)
    dias = [dialog2chat(d) for d in dialogs]
    return dias


# Usage
if __name__ == '__main__':
    _df, _keys, _dialogs = load_preprocess_ed("test")
    print(f"{_keys[0]}: {''.join(_dialogs[0])}")

    _checkpoint = "HuggingFaceH4/zephyr-7b-beta"
    _tokenizr = AutoTokenizer.from_pretrained(_checkpoint)

    # # Example 1: single message: progressive chunks
    # _messages = ["How many helicopters can a human eat in one sitting?"]
    # _system_message = "You are a friendly chatbot who always responds in the style of a pirate"
    # _dialog_chunks = get_progressive_chunks(_messages)
    # _tokenized_chat = _tokenizr.apply_chat_template(_dialog_chunks[0], tokenize=True, add_generation_prompt=True,
    #                                                return_tensors="pt")
    # print(_tokenizr.decode(_tokenized_chat[0]))
    #
    # # Example 2: multiple progressive chunks
    # system_message2 = "You're are a helpful Assistant, who provides empathetic responses to the requests from the " \
    #                  "speaker. The input contains previous turn of the dialog, where the each utterance is " \
    #                  "identified by the tags user or assistant. Be empathetic and precise. Make sure to give " \
    #                  "responses that make dialogue flow. Keep it short."
    # dc2 = list(chain.from_iterable(get_progressive_chunks(value) for value in _dialogs[0:2]))
    # ds = get_dataset(dc2, _tokenizr, tokenize=False, add_generation_prompt=False)
    # print(ds['formatted_chat'][0])
    #
    # # Example 3: whole dialog
    # d3 = [dialog2chat(value) for value in _dialogs[0:2]]
    # ds3 = get_dataset(d3, _tokenizr, tokenize=False, add_generation_prompt=False)
    # print(ds3['formatted_chat'][0])
    # print(ds3['formatted_chat'][1])

    # Example 4: entire splits, whole dialogs
    ed_test = get_ed("validation", _tokenizr, tokenize=False, add_generation_prompt=False)
    print(ed_test['formatted_chat'][0])
    # ed_train = get_ed("train", _tokenizr, tokenize=False, add_generation_prompt=False)
    # print(ed_train['formatted_chat'][0])

    import matplotlib.pyplot as plt
    import seaborn as sns
    ed_df = ed_test.to_pandas()
    s = ed_df['formatted_chat'].str.len()
    s.describe()
    sns.histplot(s, log_scale=True)
    plt.show()


