import pandas as pd
import torch
from datasets import load_dataset, Dataset

test = [[{'role': 'user', 'content': 'hello, how are you?'},
        {'role': 'assistant', 'content': 'I am good, thanks, and you?'},
        {'role': 'user', 'content': "what time is it?"},
        # {'role': 'assistant', 'content': "It's 3 o'clock"}
         ]]



from transformers import AutoTokenizer
checkpoint = 'bienpr/Llama-2-7b-chat-hf'
tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding='max_length', padding_side="right", max_length=128)
tokenizer.add_special_tokens({'pad_token':'[PAD]'})

test_tok = [tokenizer.apply_chat_template(x, tokenize=False) for x in test]


df_final = pd.DataFrame(test_tok, columns=['template_formatted_conversation_turns'])



import torch
from datasets import Dataset
from trl import DataCollatorForCompletionOnlyLM

dataset = Dataset.from_list(df_final['template_formatted_conversation_turns'].apply(lambda x: tokenizer(x, return_length=True)).to_list())
response_template = '[/INST]'
instruction_template = '[INST]'
collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template, response_template=response_template, tokenizer=tokenizer)


dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                         collate_fn=collator,
                                         batch_size=1)
for batch in dataloader:
    print(batch)
    break