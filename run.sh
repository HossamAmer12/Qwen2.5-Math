
# BoN
# hub dataset repo
# export DATASET_ID="HuggingFaceH4/Llama-3.2-1B-Instruct-best-of-N-completions"
# # config to evaluate
# export DATASET_CONFIG="HuggingFaceH4_MATH-500--T-0.8--top_p-1.0--n-1024--max_tokens-2048--bsz-8--seed-0--agg_strategy-last"
# # preds@N to evaluate
# export VOTING_N="1 2 4 16 32 64 128 256"

# # Beam search
# # hub dataset repo
# export DATASET_ID="HuggingFaceH4/Llama-3.2-1B-Instruct-beam-search-completions"
# # config to evaluate
# export DATASET_CONFIG="HuggingFaceH4_MATH-500--T-0.8--top_p-1.0--n-merged--m-4--iters-40--look-0--seed-0--agg_strategy-last"
# # preds@N to evaluate
# export VOTING_N="4 16 64 256"

# # Run the evaluation script
# python evaluation/evaluate_hf.py \
#     --dataset_id $DATASET_ID \
#     --dataset_config $DATASET_CONFIG \
#     --voting_n $VOTING_N



# Best of N you created for LLama-3.2-1B
# hub dataset repo
# export DATASET_ID="HuggingFaceH4/Llama-3.2-1B-Instruct-best-of-N-completions"

# Llama 1B 10 examples
# export DATASET_ID="/home/hossamamer/TTC_workspace/search-and-learn/data/meta-llama/Llama-3.2-1B-Instruct/best_of_n_completions.jsonl"

# TinyLlama
# export DATASET_ID="/home/hossamamer/TTC_workspace/my_repos/search-and-learn/data/home/k00925509/model_weights/tinyllama/best_of_n_completions.jsonl"

# TinyLlama 1B math code
# export DATASET_ID="/home/hossamamer/TTC_workspace/my_repos/search-and-learn/data/home/hossamamer/.cache/huggingface/hub/models--TinyLlama--TinyLlama_v1.1_math_code/snapshots/698ef988e06730a38eca552cdf86e99c08118df5/best_of_n_completions.jsonl"

# LLama-3.2-1B 500 examples BoN
# export DATASET_ID="/home/hossamamer/TTC_workspace/my_repos/search-and-learn/data/meta-llama/Llama-3.2-1B-Instruct/best_of_n_completions.jsonl"
# export DATASET_ID="sample_best_of_n_json_basic/best_of_n_completions_all.jsonl"

# Llama-3.2-1B 500 examples Pass@k
# export DATASET_ID="/home/hossamamer/TTC_workspace/my_repos/search-and-learn/data/meta-llama/Llama-3.2-1B-Instruct/best_of_n_completions.jsonl"
# export DATASET_ID="sample_best_of_n_json_basic/best_of_n_completions_all_pass.jsonl"

# TinyLlama checkpoint-100 500 examples Pass@k
# export DATASET_ID="/home/hossamamer/TTC_workspace/my_repos/search-and-learn/data/home/m00918254/TTC-checkpoints/tinyllama-math-code-checkpoint-100/best_of_n_completions.jsonl"

# TinyLlama checkpoint-200 500 examples Pass@k
# export DATASET_ID="/home/hossamamer/TTC_workspace/my_repos/search-and-learn/data/home/m00918254/TTC-checkpoints/tinyllama-math-code-checkpoint-200/best_of_n_completions_200.jsonl"

# TinyLlama checkpoint-300 500 examples Pass@k
export DATASET_ID="/home/hossamamer/TTC_workspace/my_repos/search-and-learn/data/home/m00918254/TTC-checkpoints/tinyllama-math-code-checkpoint-300/best_of_n_completions_300.jsonl"

export DATASET_ID="/home/hossamamer/TTC_workspace/my_repos/search-and-learn/data/home/m00918254/TTC-checkpoints/tinyllama-sft-prm800/from-checkpoint-20000/output/dvts_completions.jsonl"

# config to evaluate
export DATASET_CONFIG="HuggingFaceH4_MATH-500--T-0.8--top_p-1.0--n-1024--max_tokens-2048--bsz-8--seed-0--agg_strategy-last"
# preds@N to evaluate
export VOTING_N="1 2 4 16 32 64 128 256"

# Run the evaluation script
python evaluation/evaluate_hf.py \
    --dataset_id $DATASET_ID \
    --dataset_config $DATASET_CONFIG \
    --voting_n $VOTING_N

