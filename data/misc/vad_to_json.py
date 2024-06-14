import json

file = "/home/alahnala/research/empathy-generation/data/misc/NRC-VAD-Lexicon.txt"

vad_dict = {}

with open(file, 'r') as f:
    for l in f:
        parts = l.split('\t')
        vad_dict[parts[0]] = [float(item.strip()) for item in parts[1:]] 



with open("/home/alahnala/research/empathy-generation/data/misc/VAD.json", 'w') as f:
    json.dump(vad_dict, f)