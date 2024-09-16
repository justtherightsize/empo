#!/bin/bash

export PYTHONPATH="."

echo "Starting background processes...=========================="

CUDA_VISIBLE_DEVICES=0 nice /home/xsotolar/python/envs/311/bin/python ./src/emp_metrics/pipe_arun.py -sa sft_7 -ss 2  


echo "Script has finished.==============================="
