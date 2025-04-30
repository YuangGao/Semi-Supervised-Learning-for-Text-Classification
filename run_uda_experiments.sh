#!/bin/bash

# Array of different n_labeled values to test
N_LABELED_VALUES=(200 2500 10)

# Base command for MixText
BASE_CMD="python -u ./code/train_uda.py --data-path ./data/imdb_csv/ --val-iteration 300 --batch-size 8 --epochs 20 --lambda-u 0.9 --T 0.9 --rampup-length 5"

# Run experiments for each n_labeled value
for n_labeled in "${N_LABELED_VALUES[@]}"
do
    echo "Running UDA experiment with n_labeled = $n_labeled"
    $BASE_CMD --n-labeled $n_labeled
    echo "----------------------------------------"
done

echo "All UDA experiments completed." 