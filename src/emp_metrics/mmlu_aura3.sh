#!/bin/bash

export PYTHONPATH="."

echo "Starting background processes...=========================="

CUDA_VISIBLE_DEVICES=0 /home/xsotolar/python/envs/311/bin/python ./src/emp_metrics/mmlu_all.py -k /home/xsotolar/.huggingface/mistral -l --no-is_test -bm "alignment-handbook/zephyr-7b-sft-full" -a "zephyr-7b-sft-full122" 


echo "Script has finished.==============================="
