# Automated metrics for measuring Emapthy


## Epitome and Diff-Epitome

The [diff_epitome.py](src/emp_metrics/diff_epitome.py) contains the implementation for the Epitome and Diff-Epitome metrics 
from https://github.com/ondrejsotolar/EmpGPT-3. In the paper, the authors claim, that diff-epitome aligns the best with
human evaluation.   

To run this, you first need to download the fine-tuned models for Epitome, as it is a model-based metric.

### Downloading the Epitome models

Do not version the 1GB models in git. Download the models using terminal into the checkpoints/epitome_checkpoints or other directory. The addresse is also in the paper's git repo listed above.

```bash
pip install gdown
gdown --folder --id 1PXqmv-MZ1uphHvV81htuAhid2uKGGeGd
```

### Install requirements
In addition to gdown you probably need the following libraries. Otherwise, it's standard pytorch and HuggingFace stuff.   
```bash
pip install parlai
```

### Loading and inference example

[diff_epitome.py](src/emp_metrics/diff_epitome.py) contains a main method that serves as the demo on how to run the metric. It's setup in a batch mode: you need to load a list of examples and it will evaluate all of them + give you averages. 

