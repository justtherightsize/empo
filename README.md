# Empathy Generation


## Detection Pipeline

Pipeline for creating models that we may use as reward functions.


### Environment

```bash
# create nlp-transfer environment
conda env create -f data/nlp-transfer.yml;
conda activate nlp-transfer;
```

### Train and run adapters

Script: [src/run_detection_adapter.py](src/run_detection_adapter.py)


#### Basic usage
Basic options:

    -d, --data_dir DATA_DIR
                    Directory to the dataset where there is a train.csv, test.csv, val.csv
                    Example: data/empathy_datasets/wassa_essay
    
    -tc, --text_column TEXT_COLUMN
                    Name of the text column in the data csvs
                    Example: essay

    -lc, --label_column LABEL_COLUMN
                    Column name of the task labels 
                    Example: empathy
    
    -tt, --task_type TASK_TYPE
                    Specify whether the task is classification or regression
                   

Example:

```bash
python src/train_adapter.py -d data/empathy_datasets/wassa_essay -tc essay -lc empathy -tt regression -dout train_output/train_adapter_output;
```

#### Train setups

There are two general ways we use this:
1. **Target task only**: Train an adapter an adapter for --model for a target task. 
    Example: see [Basic Usage](#basic-usage)
2. **Source task to target task**: This creates an adapter composition by loading a pre-trained adapter and adding a new adapter for the target task stacked on the pre-trained adapter. To do this, pass in `-lpa` to indicate you want to load a pre-trained adapter. Then, you have to pass in the path/name of the pre-trained adapter, which is done in one of two ways, depending on which kind of pre-trained adapter you want to load: 
    1. **An adapter that you have pre-trained**: Use the option `-a {adapter_path}`. 
        * The basic usage example trains an adapter for the empathy regression target task in the wassa dataset which is saved at `train_output/train_adapter_output/wassa_essay_distress/final_adapter`, which you would pass in as `adapter_path` if you want to use it for a downstream target task.

        Example:
        ```bash
        python scripts/train_adapter.py -d data/empathy_datasets/wassa_essay -tc essay -lc distress -tt regression -dout train_output/train_adapter_output/wassa_essay_empathy---wassa_essay_distress -lpa -a train_output/train_adapter_output/wassa_essay_distress/final_adapter;
        ```
        
    2. **Adapter from adapter hub**: Use the option `--aux {adapter_path}`. 
        * An example `adapter_path` is "AdapterHub/roberta-base-pf-anli_r3".

        Example:
        ```bash
        python scripts/train_adapter.py -d data/empathy_datasets/wassa_essay -tc essay -lc empathy -tt regression -dout train_output/train_adapter_output/AdapterHub/roberta-base-pf-anli_r3---wassa_essay_empathy -lpa --aux AdapterHub/roberta-base-pf-anli_r3;
        ```







