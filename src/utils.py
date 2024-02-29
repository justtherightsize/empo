from collections import defaultdict
import datetime
from datasets import Dataset

START_OF_TURN = '<start_of_turn>'
END_OF_TURN = '<end_of_turn>'

def get_current_timestamp():
    return str(datetime.datetime.now()).replace(" ", "_").replace(".", ":")

def format_chat(chat):
    text = ""
    for turn in chat:
        text += START_OF_TURN + turn['role'] + '\n ' + turn['content'] + END_OF_TURN + '\n'
    return text

def convert_to_dataset(df):
    convToTurns = defaultdict(list)
    for _, row in df[df["level"] == 1].iterrows():
        conv_id = row["conv_id"]
        convToTurns[conv_id].append({"role": "user", "content": row["sp1"]})
        convToTurns[conv_id].append({"role": "assistant", "content": row["sp2"]})
    
    return Dataset.from_dict({"chat": list(convToTurns.values())})