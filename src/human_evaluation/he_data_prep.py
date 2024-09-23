import itertools
import pandas as pd
import numpy as np
import re


def process_he(data_path):
    data = pd.read_csv(data_path, delimiter='~', index_col=0)

    # stratify and subsample to 32*4 samples
    data_he = data.groupby('emotions', group_keys=False).apply(lambda x: x.sample(4, random_state=42))
    # 7 additional samples to make the amount of sample divisible by batch sizes 
    additional = data[(~data.index.isin(data_he.index)) & (data["emotions"].isin(['sad', 'surprised', 'proud']))][0:7]
    data_he = pd.concat([data_he, additional], axis=0)
    
    to_replace = [r'</s>\n<\|assistant\|>\n', r'</s>\n<\|user\|>\n']
    replace_w = [r'\nListener: ', r'\nSpeaker: ']

    # remove the system part of the prompt and the last speaker guide
    sys_prompt_end = data_he.iloc[0]["chat_templates"].find("</s>")
    tidy_he = data_he["chat_templates"].str[sys_prompt_end:]

    # replace the prompts with the tidy output
    tidy_he = tidy_he.replace(to_replace, replace_w, regex=True)

    data_he["chat_cut"] = tidy_he.replace(replace_w, '\n', regex=True).str[1:-1] 
    
    # append the last part of the context    
    tidy_he = tidy_he.str[1:-2] + " (Response): " + data_he["gens"]
    data_he["tidy_chat"] = tidy_he

    return data_he.sample(frac=1, random_state=42).reset_index()
    

def sample_data(data, model_name:str, batch_indices):
    data["model"] = model_name
    batch_blocks = [pd.DataFrame(data.loc[sample_index]) for sample_index in batch_indices]
    return batch_blocks
    

def create_blocks(batch_indices, batch_size=16, total_size=64):
    index_bounds, multiples = np.arange(batch_size), np.arange(total_size, step=batch_size)

    samples = [np.concatenate([(index_bounds + batch_size*i)*s for i, s in enumerate(sample) if s])
               if sum(sample) > 0 else [] for sample in batch_indices]
    return samples


def prepare_batches(data_sets, names, batch_size, total_size):
    total_batches = int(total_size/batch_size)

    # there are total_batches! = 6  permutations, thus we need three extra to annotate each sample*model three times
    batch_indexes_perm = np.array(list(itertools.permutations(np.arange(total_batches))))

    # thus we use series of samples from each of the data_sets unmixed
    batch_indexes_same = (np.ones((total_batches, total_batches))*np.arange(total_batches)).T

    perm_indices = np.array([create_blocks((batch_indexes_perm == i)*1,
                             batch_size, total_size) for i in range(len(data_sets))])

    same_indices = np.array([create_blocks([(batch_indexes_same[i] == i)*1],
                             batch_size, total_size) for i in range(len(data_sets))])
    
    batch_perm, batch_same = [[] for _ in range(total_batches)], [[], [], []]
 
    for i in range(len(data_sets)):
        for x in range(data_sets[0].shape[0]//total_size):
            batch_perm[i] += sample_data(data_sets[i], names[i], perm_indices[i] + x*total_size) 
            batch_perm[i] += [sample_data(data_sets[l], names[l], same_indices[l] + x*total_size)[0]
                              for l in range(len(data_sets))]

    batch_complete = [pd.concat([batch[i] for batch in batch_perm], axis=0).sort_index().drop_duplicates()
                      for i in range(len(batch_perm[0]))]

    return batch_complete