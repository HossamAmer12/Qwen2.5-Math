
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
export DATASET_ID="/home/hossamamer/TTC_workspace/search-and-learn/data/meta-llama/Llama-3.2-1B-Instruct/best_of_n_completions.jsonl"


# export DATASET_ID="/home/hossamamer/TTC_workspace/my_repos/search-and-learn/data/home/k00925509/model_weights/tinyllama/best_of_n_completions.jsonl"

# config to evaluate
export DATASET_CONFIG="HuggingFaceH4_MATH-500--T-0.8--top_p-1.0--n-1024--max_tokens-2048--bsz-8--seed-0--agg_strategy-last"
# preds@N to evaluate
export VOTING_N="1 2 4 16 32 64"

# Run the evaluation script
python evaluation/evaluate_hf.py \
    --dataset_id $DATASET_ID \
    --dataset_config $DATASET_CONFIG \
    --voting_n $VOTING_N

