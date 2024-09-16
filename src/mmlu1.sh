#!/bin/bash

export PYTHONPATH="."

echo "Starting background processes..."

CUDA_VISIBLE_DEVICES=0 /home/xshared/devel/envs/311/bin/python \
  /home/xshared/devel/empathy-generation/src/emp_metrics/mmlu_all.py \
  -bm mistralai/Mistral-7B-v0.1 -a alignment-handbook/zephyr-7b-sft-qlora &

CUDA_VISIBLE_DEVICES=1 /home/xshared/devel/envs/311/bin/python \
  /home/xshared/devel/empathy-generation/src/emp_metrics/mmlu_all.py \
  -bm HuggingFaceH4/zephyr-7b-beta &

pid1=$!
pid2=$!

echo "First background process PID: $pid1"
echo "Second background process PID: $pid2"

wait $pid1
echo "First background process has completed."

wait $pid2
echo "Second background process has completed."

echo "Script has finished."
