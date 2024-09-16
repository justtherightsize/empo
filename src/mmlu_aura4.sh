#!/bin/bash

export PYTHONPATH="."

echo "Starting background processes...=========================="

CUDA_VISIBLE_DEVICES=0 nice /home/xsotolar/python/envs/311/bin/python ./src/emp_metrics/mmlu_all.py -k /home/xsotolar/.huggingface/mistral -l --no-is_test -bm "alignment-handbook/zephyr-7b-sft-full" -a "zephyr-7b-sft-full123" &

CUDA_VISIBLE_DEVICES=1 nice /home/xsotolar/python/envs/311/bin/python ./src/emp_metrics/mmlu_all.py -k /home/xsotolar/.huggingface/mistral -l --no-is_test -bm "alignment-handbook/zephyr-7b-sft-full" -a "zephyr-7b-sft-full124" &

pid1=$!
pid2=$!

echo "First background process PID: $pid1"
echo "Second background process PID: $pid2"

wait $pid1
echo "First background process has completed."

wait $pid2
echo "Second background process has completed."

echo "Script has finished.==============================="
