#!/bin/bash

# Array of different n_labeled values to test
N_LABELED_VALUES=(10 200 2500)

# Base command for MixText
BASE_CMD="python ./code/train.py --gpu 0 --data-path ./data/ag_news_csv/ --batch-size 4 --batch-size-u 8 --epochs 20 --val-iteration 1000 --lambda-u 1 --T 0.5 --alpha 16 --mix-layers-set 7 9 12 --lrmain 0.000005 --lrlast 0.0005"

# Run experiments for each n_labeled value
for n_labeled in "${N_LABELED_VALUES[@]}"
do
    echo "Running MixText experiment with n_labeled = $n_labeled"
    $BASE_CMD --n-labeled $n_labeled
    echo "----------------------------------------"
done

echo "All MixText experiments completed." 