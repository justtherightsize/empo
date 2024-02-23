import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)
import argparse
from glob import glob
from shutil import rmtree
import traceback
import pandas as pd
import torch
import os
import numpy as np
# os.environ["CUDA_VISIBLE_DEVICES"]='0'
from transformers import AutoTokenizer,AutoModelForSequenceClassification,EvalPrediction,TrainingArguments,EarlyStoppingCallback
from adapters import AutoAdapterModel,AdapterTrainer,AdapterConfig
from datetime import datetime
from transformers.trainer_callback import TrainerCallback
from scipy import stats
from sklearn import metrics
from collections import Counter, defaultdict
from operator import itemgetter
from tabulate import tabulate
import adapters.composition as ac
from pprint import pprint
from typing import Optional
from transformers.trainer_utils import has_length
from torch.utils.data import RandomSampler
import torch.fx
from sklearn.model_selection import train_test_split
from copy import deepcopy
import math


#torch.set_default_device('cpu')

adpt_options = {'comsense/siqa@ukp': {'name': 'comsense/siqa@ukp',
            'model': 'roberta-base',
            'source': None,
            'config': 'pfeiffer'},
            'comsense/winogrande@ukp': {'name': 'comsense/winogrande@ukp',
            'model': 'roberta-base',
            'source': None,
            'config': 'pfeiffer'},
            'comsense/hellaswag@ukp': {'name': 'comsense/hellaswag@ukp',
            'model': 'roberta-base',
            'source': None,
            'config': 'pfeiffer'},
            'comsense/cosmosqa@ukp': {'name': 'comsense/cosmosqa@ukp',
            'model': 'roberta-base',
            'source': None,
            'config': 'pfeiffer'},
            'comsense/csqa@ukp': {'name': 'comsense/csqa@ukp',
            'model': 'roberta-base',
            'source': None,
            'config': 'pfeiffer'},
            'nli/cb@ukp': {'name': 'nli/cb@ukp',
            'model': 'roberta-base',
            'source': None,
            'config': 'pfeiffer'},
            'nli/multinli@ukp': {'name': 'nli/multinli@ukp',
            'model': 'roberta-base',
            'source': None,
            'config': 'pfeiffer'},
            'AdapterHub/roberta-base-pf-copa': {'name': 'AdapterHub/roberta-base-pf-copa',
            'model': 'roberta-base',
            'source': 'hf',
            'config': None},
            'AdapterHub/roberta-base-pf-ud_en_ewt': {'name': 'AdapterHub/roberta-base-pf-ud_en_ewt',
            'model': 'roberta-base',
            'source': 'hf',
            'config': None},
            'AdapterHub/roberta-base-pf-ud_deprel': {'name': 'AdapterHub/roberta-base-pf-ud_deprel',
            'model': 'roberta-base',
            'source': 'hf',
            'config': None},
            'AdapterHub/roberta-base-pf-emo': {'name': 'AdapterHub/roberta-base-pf-emo',
            'model': 'roberta-base',
            'source': 'hf',
            'config': None},
            'AdapterHub/roberta-base-pf-pmb_sem_tagging': {'name': 'AdapterHub/roberta-base-pf-pmb_sem_tagging',
            'model': 'roberta-base',
            'source': 'hf',
            'config': None},
            'AdapterHub/roberta-base-pf-anli_r3': {'name': 'AdapterHub/roberta-base-pf-anli_r3',
            'model': 'roberta-base',
            'source': 'hf',
            'config': None},
            'argument/ukpsent@ukp': {'name': 'argument/ukpsent@ukp',
            'model': 'roberta-base',
            'source': None,
            'config': 'pfeiffer'},
            'SALT-NLP/pfadapter-roberta-base-qnli-combined-value': {'name': 'SALT-NLP/pfadapter-roberta-base-qnli-combined-value',
            'model': 'roberta-base',
            'source': 'hf',
            'config': None}}

class RegressionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, label2idx=None):
        self.encodings = encodings
        self.labels = labels if not label2idx else [label2idx[l] for l in labels]
        self.num_labels = len(set(labels)) if not label2idx else len(label2idx)

    def __getitem__(self, idx):
        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)

class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, label2idx):
        self.encodings = encodings
        self.labels = [label2idx[l] for l in labels]
        self.num_labels = len(label2idx)

    def __getitem__(self, idx):
        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

adjustment_increment_divisors = {}
for t in ['conv_Emotion', 'conv_EmotionalPolarity', 'conv_Empathy']:
    adjustment_increment_divisors[t] = 3
adjustment_increment_divisors['wassa_essay_empathy'] = 6
adjustment_increment_divisors['wassa_essay_distress'] = 8
adjustment_increment_divisors['condolence_empathy_rating'] = 2


def reduce_train_data(train):
    adjusted_df = deepcopy(train)
    trg_task = args.data_dir.split('/')[-1] + '_' + args.label_column
    if trg_task in adjustment_increment_divisors:
        divisor = adjustment_increment_divisors[trg_task]
        for idx, row in train.iterrows():
            adjusted_label = adjust(row[args.label_column], divisor)
            adjusted_df.loc[idx, args.label_column] = adjusted_label

    label_counts = adjusted_df[args.label_column].value_counts()
    indices_of_single_item = []
    for l, count in label_counts.items():
        # print(l, count)
        if count == 1:
            idx = adjusted_df[adjusted_df[args.label_column] == l].index[0]
            indices_of_single_item.append(idx)
    # adjusted_df[~adjusted_df.index.isin(indices_of_single_item)]
    tmp = adjusted_df[~adjusted_df.index.isin(indices_of_single_item)]
    # print(tmp)
    tmp, _ = train_test_split(tmp, stratify=tmp[args.label_column], test_size=1-args.train_data_proportion)
    train = pd.concat([tmp, adjusted_df.loc[indices_of_single_item]])
    # train = train.loc[adjusted_df.index]
    # print(train[args.label_column].value_counts())
    # quit()
    return train


def run_adjustments(pred_df, models, divisor):
    adjusted_df = deepcopy(pred_df)
    models = [model for model in models if model in pred_df.columns]

    for idx, row in pred_df.iterrows():
        actual = row['actual']
        actual = adjust(row['actual'], divisor)
        adjusted_df.loc[idx, 'actual'] = actual
        for model in models:
            model_pred = row[model]
            model_pred = adjust(model_pred, divisor)

            adjusted_df.loc[idx, model] = model_pred


    pred_df = deepcopy(adjusted_df)
    return pred_df






def adjust(model_pred, divisor):
    remainder = model_pred - math.floor(model_pred)

    l = [i/divisor for i in range(0,divisor+1)]
    l2 = [abs(remainder-i) for i in l]
    l2.index(min(l2))
    model_pred = math.floor(model_pred) + l[l2.index(min(l2))]
    model_pred = round(model_pred, 1)
    return model_pred

def make_transformers_dataset(args):
    '''
    train, test, and dev should already be determined
    '''

    train = pd.read_csv(os.path.join(args.data_dir, 'train.csv'), index_col=0)
    if args.train_data_proportion != 1:
        train = reduce_train_data(train)


    val = pd.read_csv(os.path.join(args.data_dir, 'val.csv'), index_col=0)
    test = pd.read_csv(os.path.join(args.data_dir, 'test.csv'), index_col=0)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # tokenizer = AutoTokenizer.from_pretrained(args.model, token=ACCESS_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    # TODO: add special tokens

    train_text = list(train[args.text_column])
    val_text = list(val[args.text_column])
    test_text = list(test[args.text_column])

    for tset in [train_text, val_text, test_text]:
        for i in range(len(tset)):
            tset[i] = tset[i].replace('</speaker>', '').replace('</prompt>', '').replace('</listener>', '').replace('</target>', '').replace('<speaker>', '<s').replace('<listener>', '<l').replace('<emotion>', '<e').replace('<prompt>', '<p').replace('<target>', '<t')

    if args.task_type == 'regression':
        # I think this is just for the wassa labels...
        if 'unknown' in list(train[args.label_column]):
            train = train.drop(train[train[args.label_column] == 'unknown'].index)
            val = val.drop(val[val[args.label_column] == 'unknown'].index)
            test = test.drop(test[test[args.label_column] == 'unknown'].index)


        train_labels = [float(item) for item in train[args.label_column]]
        val_labels = [float(item) for item in val[args.label_column]]
        test_labels = [float(item) for item in test[args.label_column]]

        training_data = RegressionDataset(tokenizer(train_text, truncation=True, padding=True),
                                        train_labels)

        validation_data = RegressionDataset(tokenizer(val_text, truncation=True, padding=True),
                                        val_labels)

        test_data = RegressionDataset(tokenizer(test_text, truncation=True, padding=True),
                                        test_labels)
        label2idx = None

    elif args.task_type == 'classification':
        if train[args.label_column].dtype == 'int64':
            train_labels = [int(item) for item in train[args.label_column]]
            val_labels = [int(item) for item in val[args.label_column]]
            test_labels = [int(item) for item in test[args.label_column]]
            label2idx = {l:int(i) for i,l in enumerate(sorted(list(set(train_labels))))}
        elif type(train[args.label_column].values[0]) == str:
            train_labels = list(train[args.label_column])
            val_labels = list(val[args.label_column])
            test_labels = list(test[args.label_column])
            label2idx = {l:i for i,l in enumerate(sorted(list(set(train_labels))))}
        else:
            train_labels = list(train[args.label_column])
            val_labels = list(val[args.label_column])
            test_labels = list(test[args.label_column])
        
        training_data = ClassificationDataset(tokenizer(train_text, truncation=True, padding=True),
                                        train_labels, label2idx)

        validation_data = ClassificationDataset(tokenizer(val_text, truncation=True, padding=True),
                                        val_labels, label2idx)

        test_data = ClassificationDataset(tokenizer(test_text, truncation=True, padding=True),
                                        test_labels, label2idx)

    
    return training_data,validation_data,test_data,label2idx,[train.index,val.index,test.index]

def get_problem_type(args):
    if args.task_type in ['classification']:
        problem_type = 'single_label_classification'
    elif args.task_type == 'regression':
        problem_type='regression'
    else:
        print("problem type not implemented")
        return None
    return problem_type

def classification_metrics(p: EvalPrediction):
    '''
    import numpy as np
    from transformers import EvalPrediction
    from sklearn.metrics import accuracy_score, hamming_loss, balanced_accuracy_score, f1_score, precision_score, recall_score
    '''
    preds = np.argmax(p.predictions, axis=1)

    return {"accuracy":metrics.accuracy_score(list(p.label_ids), list(preds)),
            "hamming_loss":metrics.hamming_loss(p.label_ids, preds),
            "balanced_accuracy_score":metrics.balanced_accuracy_score(p.label_ids, preds),
            "f1_score":metrics.f1_score(p.label_ids, preds, average="macro"),
            "precision_score":metrics.precision_score(p.label_ids, preds, average="macro"),
            "recall_score":metrics.recall_score(p.label_ids, preds, average="macro")
            }

def regression_metrics(p: EvalPrediction):
    preds = p.predictions[:, 0]
    pr, pval = stats.pearsonr(p.label_ids, preds)
    return {"pearsonr": pr, #"pval": pval,
            "mean_squared_error": metrics.mean_squared_error(p.label_ids, preds),
            "mean_absolute_error": metrics.mean_absolute_error(p.label_ids, preds),
            "median_absolute_error": metrics.median_absolute_error(p.label_ids, preds),
#             "mean_absolute_percentage_error": mean_absolute_percentage_error(p.label_ids, preds),
            "r2_score": metrics.r2_score(p.label_ids, preds)}

class MyLog(TrainerCallback):
    """
    A bare :class:`~transformers.TrainerCallback` that just prints the logs.
    """
    def __init__(self, logging_dir='logs/', task_type='classification'):
        self.logfile = os.path.join(logging_dir, f"{self.now().replace(' ', '_')[:-3]}_")
        suffix = 0
        while os.path.exists(self.logfile + '-' + str(suffix) + '.log'):
            suffix += 1
        self.logfile += '-' + str(suffix) + '.log'
        self.metric_headers = self.get_metric_headers(task_type)
        print(self.now(), "\t".join(self.metric_headers).strip(),  sep='\t', file=open(self.logfile, "a"))
    
    def log_experiment_args(self, experiment_args):
        str_out_list = ["\n", "== Experiment Args =="]
        for k,v in experiment_args.items():
            str_out_list.append(f"{k}\t{v}")
        str_out = '\n'.join(str_out_list)
        print(str_out,  file=open(self.logfile, "a")) 

    def print2log(self, str_out):
        print(str_out,  file=open(self.logfile, "a"))

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        metrics_str = '\t'.join([f"{metrics[h]}" for h in self.metric_headers])
        print(self.now(), metrics_str, sep='\t', file=open(self.logfile, "a"))

    def log_test_metrics(self, metrics):
        h_list = []
        m_list = []
        for h in self.metric_headers:
            test_metric = h.replace('eval_', '')
            if test_metric in metrics:
                m_list.append(f"{metrics[test_metric]}")
                h_list.append(f"test_{test_metric}")
            else:
                m_list.append("-")
                h_list.append("-")
        print(self.now(), "\t".join(h_list).strip(), sep='\t', file=open(self.logfile, "a"))
        print(self.now(), "\t".join(m_list).strip(), sep='\t', file=open(self.logfile, "a"))

    def now(self):
        date = datetime.now()        
        return date.strftime("%d-%m-%Y %H:%M:%S")
    
    def get_metric_headers(self,task_type):
        classification_headers = ['eval_loss',
                        'eval_accuracy',
                        'eval_hamming_loss',
                        'eval_balanced_accuracy_score',
                        'eval_f1_score',
                        'eval_precision_score',
                        'eval_recall_score',
                        'eval_runtime',
                        'eval_samples_per_second',
                        'eval_steps_per_second',
                        'epoch']
        
        regression_headers = ['eval_loss',
                            'eval_pearsonr',
                            'eval_mean_squared_error',
                            'eval_mean_absolute_error',
                            'eval_median_absolute_error',
                            'eval_r2_score',
                            'eval_runtime',
                            'eval_samples_per_second',
                            'eval_steps_per_second',
                            'epoch']
        if task_type == 'classification':
            return classification_headers
        elif task_type == 'regression':
            return regression_headers

def classification_report(y_true, y_pred,tablefmt ='latex_booktabs'):
    out = metrics.classification_report(y_true, y_pred,output_dict=True, digits=4, zero_division=0)
    l_counts = Counter(y_true)
    maj_baseline = sorted(l_counts.items(), key=itemgetter(1), reverse=True)[0][1] / sum(l_counts.values())
    
    cls_metric_dict = defaultdict(lambda:[])
    keys = list(out.keys())
    for i, l in enumerate(keys):
        if l == 'accuracy':
            break
        cls_metric_dict['label'].append(l)
        for item, val in out[l].items():
            val = val*100 if item != 'support' else val
            cls_metric_dict[item].append(val)

    for label in ['macro avg', 'weighted avg']:
        cls_metric_dict['label'].append(label)
        for item in ['precision', 'recall', 'f1-score', 'support']:
            val = out[label][item]*100 if item != 'support' else out[label][item]
            cls_metric_dict[item].append(val)

    for label, val_ in [('Maj class %', maj_baseline), ('accuracy', out['accuracy'])]:
        cls_metric_dict['label'].append(label)
        for item in ['precision', 'recall', 'f1-score', 'support']:
            val = val_*100 #if item == 'f1-score' else None
            # val = 0 if item == 'support' else val
            cls_metric_dict[item].append(val)

    return cls_metric_dict, tabulate(pd.DataFrame(cls_metric_dict), headers='keys',tablefmt=tablefmt,showindex=False)        

def regr_to_cls_report(y_true, y_pred,tablefmt ='latex_booktabs'):
    actual_classes = [int(value) for value in y_true]
    predicted_classes = [int(round(value)) for value in y_pred]

    miny = min(actual_classes)
    maxy = max(actual_classes)
    for i, val in enumerate(predicted_classes):
        if val < miny:
            predicted_classes[i] = miny
        elif val > maxy:
            predicted_classes[i] = maxy
    
    cls_report_dict, _ = classification_report(actual_classes, predicted_classes)

    return cls_report_dict, tabulate(pd.DataFrame(cls_report_dict), headers='keys',tablefmt=tablefmt,floatfmt=".2f",showindex=False)


def main(args):

    """ == Initialize logger for this run == """
    logger = MyLog(logging_dir=args.logging_dir, task_type=args.task_type)    

    """ == Prepare dataset == """
    training_data,val_data,test_data,label2idx,split_indexes = make_transformers_dataset(args)
    

    num_labels=len(label2idx) if args.task_type in ['classification'] else 1
    kwargs = {'problem_type':get_problem_type(args),'num_labels':num_labels}
    if args.task_type != 'regression':
        kwargs['label2id'] = label2idx
        kwargs['id2label'] = {v:k for k,v in label2idx.items()}

    """ == Initialize model from pretrained LLM == """
    # model = AutoAdapterModel.from_pretrained(args.model, **kwargs, token=ACCESS_TOKEN)
    model = AutoAdapterModel.from_pretrained(args.model, **kwargs)

    adapter_name = None
    if args.load_pretrained_adapter:
        if args.aux:
            config = AdapterConfig.load(adpt_options[args.aux]['config'])# if adpt['config'] else None
            adapter_name = model.load_adapter(adpt_options[args.aux]['name'], source=adpt_options[args.aux]['source'], config=config)
        else:
            adapter_name = model.load_adapter(args.adapter_path, with_head=args.with_adapter_head, model_name=args.model)

    """ == Add adapter to be trained == """
    model.add_adapter(args.task_name, config=args.adapter_type)
    pprint(model.adapters_config.__dict__)
    print(model.adapter_summary())
    pprint(model.__dict__)
    model.train_adapter(args.task_name)
    model.add_classification_head(args.task_name, num_labels=num_labels)

    if args.load_pretrained_adapter:
        model.active_adapters = ac.Stack(adapter_name, args.task_name)
    else:
        model.set_active_adapters(args.task_name)

    """ == Set training arguments == """
    training_args =  TrainingArguments(
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        output_dir=args.output_dir,
        load_best_model_at_end=args.load_best_model_at_end,
        save_strategy=args.save_strategy,
        evaluation_strategy=args.evaluation_strategy,
        logging_steps=args.logging_steps,
        eval_steps=args.logging_steps,
        save_steps=args.logging_steps,
        logging_dir=args.logging_dir,
        disable_tqdm=False,
        metric_for_best_model=args.metric_for_best_model,
        seed=args.RANDOM_SEED,
        no_cuda=args.cpu,
        report_to='none',
    )
    if args.cpu:
        model.to('cpu')

    """ == Initialize trainer and train == """
    callbacks = [logger] # logs eval metrics
    if args.early_stopping_patience:
        # stops training early if no improvement after x steps
        early_stopping_patience = EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)
        callbacks.append(early_stopping_patience)

    trainer = CustomAdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=training_data,
        eval_dataset=val_data,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    trainer.train()


    """ == Save final adapter. Saves best checkpoint if load_best_model_at_end is True == """
    model.save_adapter(os.path.join(args.output_dir, 'final_adapter'), args.task_name)
    if args.cleanup:
        print("cleaning up checkpoint dirs")
        checkpoint_dirs = glob(os.path.join(args.output_dir, 'checkpoint*'))
        for dir in checkpoint_dirs:
            rmtree(dir)



    """ == Run inference on test set == """
    print("Predicting on test set.")
    predictions = trainer.predict(test_data)
    gold_standard = predictions.label_ids

    if args.task_type != 'regression':            
        test_predictions = np.argmax(predictions.predictions, axis=1)
    else:
        test_predictions = predictions.predictions[:, 0]


    """ == Save Predictions == """
    df_out = {'idx':split_indexes[-1], 'predicted':test_predictions, 'actual':gold_standard}
    df_out = pd.DataFrame(df_out)
    df_out.to_csv(os.path.join(args.output_dir, 'predictions.csv'))    

    """ == Log test prediction metrics == """
    prediction_metrics = compute_metrics(predictions)
    logger.log_test_metrics(prediction_metrics)

    if args.task_type == 'regression' and args.regression_cls:
        metric_dict, metric_table = regr_to_cls_report(df_out['actual'],df_out['predicted'],tablefmt='tsv')
        header = "\n== Classification results when rounding regression values ==\n"
        logger.print2log(f"{header}{metric_table}")

    """ == Log experiment information == """
    logger.log_experiment_args(vars(args))

    return

class CustomAdapterTrainer(AdapterTrainer):
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None
        return RandomSampler(self.train_dataset, num_samples=min(5000, len(self.train_dataset)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='python run_detection_adapter.py',
                    description='Give a csv file for training an adapter')
    parser.add_argument('-d', '--data_dir', help="Directory to the dataset where there is a train.csv, test.csv, val.csv")
    parser.add_argument('-a', '--adapter_path', default="train_adapter_output_TEST/conv_EmotionalPolarity/final_adapter", help="Directory where the adapter is. Should contain adapter_config.json, head_config.json, pytorch_adapter.bin, pytorch_model_head.bin.") # default for testing
    parser.add_argument('-tc', '--text_column')
    parser.add_argument('-lc', '--label_column')
    parser.add_argument('-tt', '--task_type')
    # parser.add_argument('-tok', '--tokenizer', default='roberta-base')
    parser.add_argument('-m', '--model', default='roberta-base')
    parser.add_argument('-tn', '--task_name', default=None)
    # training arguments
    parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float)
    parser.add_argument('--warmup_steps', default=1000, type=int) # set to 1000
    parser.add_argument('-te', '--num_train_epochs', default=5, type=int) # set to 5
    parser.add_argument('-tbs', '--per_device_train_batch_size', default=10, type=int)
    parser.add_argument('-vbs', '--per_device_eval_batch_size', default=10, type=int)
    parser.add_argument('-dout', '--output_dir', default='train_adapter_output') # set to train_adapter_output
    parser.add_argument('--load_best_model_at_end', default=True)
    parser.add_argument('--save_strategy', default='steps')
    parser.add_argument('--evaluation_strategy', default='steps')
    parser.add_argument('--logging_steps', default=100, type=int)
    parser.add_argument('--logging_dir', default='logs')
    parser.add_argument('--metric_for_best_model', default=None)
    parser.add_argument('--RANDOM_SEED', default=35, type=int)
    parser.add_argument('--early_stopping_patience', default=0, type=int)
    parser.add_argument('--regression_cls', action='store_true')
    parser.add_argument('--with_adapter_head', action='store_true', default=False)
    parser.add_argument('-at', '--adapter_type', default='seq_bn', help='Can be: seq_bn, double_seq_bn, par_bn, scaled_par_bn, seq_bn_inv, double_seq_bn_inv, compacter, compacter++, prefix_tuning, prefix_tuning_flat, lora, ia3, mam, unipelt, prompt_tuning')
    parser.add_argument('--cpu', action='store_true', default=False)
    parser.add_argument('-lpa', '--load_pretrained_adapter', action='store_true', default=False)
    parser.add_argument('-aux', '--aux', default=None, type=str)
    parser.add_argument('--train_data_proportion', default=1, type=float)
    parser.add_argument('--cleanup', action='store_true', default=False, help="Setting this will clean up all the checkpoint directories after the final_adapter is saved")
    # redo both 
    #   

    
    args = parser.parse_args()

    

    """ == task name will be adapter name == """
    if not args.task_name:
        data_dir =  args.data_dir[:-1] if args.data_dir[-1] == '/' else args.data_dir
        datasetname = data_dir.split('/')[-1]
        args.task_name = f"{datasetname}_{args.label_column}"
        print(args.task_name)
    
    """ == checkpoints, final adapter, logs, and test predictions will all go to output dir == """
    # if args.output_dir == 'train_adapter_output':
    args.output_dir = f'{args.output_dir}/{args.task_name}'
    if args.logging_dir == 'logs':
        args.logging_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(args.logging_dir, exist_ok=True)

    """ == this metric is used to determine which checkpoint was the best """
    if not args.metric_for_best_model:
        args.metric_for_best_model = "eval_pearsonr" if args.task_type == 'regression' else "eval_f1_score"

    """ == sets up which metrics function to use for experiment == """
    compute_metrics = classification_metrics if args.task_type in ['classification'] else regression_metrics

    """ == results will be the same for every run with the same exact parameters == """
    from transformers.trainer_utils import set_seed
    set_seed(args.RANDOM_SEED)

    """ == run experiment == """
    main(args)




    
