# Semi-Supervised Learning for Text Classification

This repository contains code for implementing semi-supervised learning approaches for text classification, including MixText and BERT-based models.

## Getting Started

These instructions will help you set up and run the code for semi-supervised text classification.

### Requirements
* Python 3.6 or higher
* PyTorch >= 2.4.1
* Transformers >= 4.46.3
* Fairseq >= 0.12.2
* Other dependencies listed in `environment.yml`

### Environment Setup

We recommend using conda to set up the environment. First, install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/download/), then create and activate the environment:

```bash
# Create environment from environment.yml
conda env create -f environment.yml
```

### Code Structure
```
|__ data/
        |__ yahoo_answers_csv/ --> Datasets for Yahoo Answers
            |__ classes.txt --> Classes for Yahoo Answers dataset
            |__ train.csv --> Original training dataset
            |__ test.csv --> Original testing dataset

|__ code/
        |__ read_data.py --> Code for reading and preprocessing datasets
        |__ normal_bert.py --> BERT baseline model implementation
        |__ normal_train.py --> Training code for BERT baseline
        |__ mixtext.py --> MixText model implementation
        |__ train.py --> Training code for MixText model
```

### Dataset Preparation

The code supports several text classification datasets available at [this Google Drive folder](https://drive.google.com/drive/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M?resourcekey=0-TLwzfR2O-D2aPitmn5o9VQ). 

To use a dataset:
1. Download the desired dataset from the Google Drive folder
2. Extract the contents to the `data/` directory
3. The dataset should include:
   - `train.csv`: Training data
   - `test.csv`: Testing data
   - `classes.txt`: Class labels (for some datasets)

### Training Models

#### Training BERT Baseline
To train the BERT baseline model using only labeled data:
```bash
python code/normal_train.py --gpu 0,1 --n-labeled 10 --data-path ./data/yahoo_answers_csv/ --batch-size 8 --epochs 20
```

#### Training MixText Model
To train the MixText model using both labeled and unlabeled data:
```bash
python code/train.py --gpu 0,1 --n-labeled 10 --data-path ./data/yahoo_answers_csv/ --batch-size 4 --batch-size-u 8 --epochs 20 --val-iteration 1000 --lambda-u 1 --T 0.5 --alpha 16 --mix-layers-set 7 9 12 --lrmain 0.000005 --lrlast 0.0005
```

### Command Line Arguments

#### For BERT Baseline (`normal_train.py`):
- `--gpu`: GPU IDs to use (comma-separated)
- `--n-labeled`: Number of labeled samples per class
- `--data-path`: Path to dataset directory
- `--batch-size`: Batch size for training
- `--epochs`: Number of training epochs

#### For MixText (`train.py`):
- `--gpu`: GPU IDs to use (comma-separated)
- `--n-labeled`: Number of labeled samples per class
- `--data-path`: Path to dataset directory
- `--batch-size`: Batch size for labeled data
- `--batch-size-u`: Batch size for unlabeled data
- `--epochs`: Number of training epochs
- `--val-iteration`: Validation frequency
- `--lambda-u`: Weight for unlabeled loss
- `--T`: Temperature parameter
- `--alpha`: Mixup alpha parameter
- `--mix-layers-set`: Layers to apply mixup
- `--lrmain`: Learning rate for main model
- `--lrlast`: Learning rate for last layer