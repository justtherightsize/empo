python src/emp_metrics/run_empathy_eval.py -f data/empo/preds_dlrxxx_zephyr-7b-sft-full.txt -m diversity > data/results/TREE_preds_dlrxxx_zephyr-7b-sft-full.txt;
python src/emp_metrics/run_empathy_eval.py -f data/empo/preds_dlrxxx_zephyr-7b-sft-full122.txt -m diversity > data/results/TREE_preds_dlrxxx_zephyr-7b-sft-full122.txt;


python src/emp_metrics/run_empathy_eval.py -f data/empo/preds_dlrxxx_zephyr-7b-sft-full122.txt -m diversity -hu > data/results/TREE_human.txt;