#! /bin/bash

# Usage: bash script/preprocess/preprocess_train.sh

set -e

python -m src.runner.preprocessor \
    --data_split "train_processed" \
    --data_file_path "data/bird/train/train_processed.json" \
    --database_root_dir "data/bird/train/train_databases" \
    --save_root_dir "data/preprocessed/bird_task" \
    --lsh_threshold 0.5 \
    --lsh_signature_size 128 \
    --lsh_n_gram 3 \
    --lsh_top_k 10 \
    --edit_similarity_threshold 0.3 \
    --embedding_similarity_threshold 0.6 \
    --n_parallel_processes 16 \
    --max_dataset_samples -1 \
    --max_example_length 50 \
    "$@"
