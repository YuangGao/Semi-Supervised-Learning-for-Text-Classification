#!/bin/bash

# Array of different n_labeled values to test
N_LABELED_VALUES=(10 200 2500)

# Base command
BASE_CMD="python -u code/normal_train.py --gpu 0 --data-path ./data/ag_news_csv/ --batch-size 8 --epochs 20"

# Run experiments for each n_labeled value
for n_labeled in "${N_LABELED_VALUES[@]}"
do
    echo "Running experiment with n_labeled = $n_labeled"
    $BASE_CMD --n-labeled $n_labeled
    echo "----------------------------------------"
done

echo "All experiments completed." 