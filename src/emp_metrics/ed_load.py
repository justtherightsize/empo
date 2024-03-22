from pathlib import Path
from typing import Union, Tuple, Any, List

import pandas
import pandas as pd
from datasets import load_dataset


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


# Usage
if __name__ == '__main__':
    df, keys, dialogs = load_preprocess_ed("test")
    print(f"{keys[0]}: {''.join(dialogs[0])}")
