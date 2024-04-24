from typing import Dict

import numpy as np

from src.emp_metrics.ed_load import load_preprocess_ed


def get_opposite_emotions(method:str) -> Dict[str, str]:
    if method == "plutchik_original":
        return {"sentimental": "",
                "afraid": "angry",
                "proud": "",
                "faithful": "",
                "terrified": "",
                "joyful": "sad",
                "angry": "afraid",
                "sad": "joyful",
                "jealous": "",
                "grateful": "",
                "prepared": "",
                "embarrassed": "",
                "excited": "",
                "annoyed": "",
                "lonely": "",
                "ashamed": "",
                "guilty": "",
                "surprised": "anticipating",
                "nostalgic": "",
                "confident": "",
                "furious": "",
                "disappointed": "",
                "caring": "",
                "trusting": "disgusted",
                "disgusted": "trusting",
                "anticipating": "surprised",
                "anxious": "",
                "hopeful": "",
                "content": "",
                "impressed": "",
                "apprehensive": "annoyed",
                "devastated": ""}
    if method == "plutchik_1":
        return {"sentimental": "apprehensive",
                "afraid": "angry",
                "proud": "ashamed",
                "faithful": "jealous",
                "terrified": "furious",
                "joyful": "sad",
                "angry": "afraid",
                "sad": "joyful",
                "jealous": "faithful",
                "grateful": "disgusted",
                "prepared": "anxious",
                "embarrassed": "confident",
                "excited": "devastated",
                "annoyed": "apprehensive",
                "lonely": "caring",
                "ashamed": "proud",
                "guilty": "proud",
                "surprised": "anticipating",
                "nostalgic": "hopeful",
                "confident": "confident",
                "furious": "terrified",
                "disappointed": "impressed",
                "caring": "lonely",
                "trusting": "disgusted",
                "disgusted": "trusting",
                "anticipating": "surprised",
                "anxious": "content",
                "hopeful": "nostalgic",
                "content": "anxious",
                "impressed": "disappointed",
                "apprehensive": "annoyed",
                "devastated": "excited"}
    else:
        raise NotImplemented


def get_opposite_ed_keys(split:str, emo_method:str="plutchik_1", sampling:str="random") -> dict[
    str, str]:
    df, keys, _ = load_preprocess_ed(split)

    # get the keys of opposite dialogs
    oposite_keys = {}
    for key in get_opposite_emotions(emo_method).keys():
        df_emotion = df[df["context"] == key]
        df_opposite = df[df["context"] == get_opposite_emotions(emo_method)[key]]

        em_keys = df_emotion["conv_id"].unique()
        op_keys = df_opposite["conv_id"].unique()

        if len(em_keys) > len(op_keys):
            op_keys = np.concatenate((op_keys, op_keys[:len(em_keys) - len(op_keys)]))
        elif len(em_keys) < len(op_keys):
            op_keys = op_keys[:len(em_keys)]

        if sampling =="random":
            random_permutation = np.random.permutation(op_keys)
        else:
            raise NotImplemented

        for ek,op in zip(em_keys, random_permutation):
            oposite_keys[ek] = op

    return oposite_keys


if __name__ == '__main__':
    get_opposite_ed_keys("test")
    print("Done.")
