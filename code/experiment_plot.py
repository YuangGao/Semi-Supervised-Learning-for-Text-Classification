import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
import os
import numpy as np
import glob

# Set the style without LaTeX
plt.style.use(['science', 'no-latex'])

def load_experiment_data(exp_dir):
    """Load training metrics from an experiment directory."""
    metrics_path = os.path.join(exp_dir, 'training_metrics.csv')
    return pd.read_csv(metrics_path)

def find_experiment_dir(base_dir, n_labeled, model_type='bert'):
    """Find experiment directory based on pattern matching."""
    if model_type == 'bert':
        pattern = f"{base_dir}/exp_*_nlabeled{n_labeled}_epochs20_bs8"
    elif model_type == 'mixtext':
        pattern = f"{base_dir}/exp_*_nlabeled{n_labeled}_unlabeled5000_epochs20_bs4_mixTrue_method0"
    else:  # uda
        pattern = f"{base_dir}/exp_*_nlabeled{n_labeled}_unlabeled5000_epochs20_bs8"
    
    matching_dirs = glob.glob(pattern)
    if matching_dirs:
        return matching_dirs[0]  # Return the first matching directory
    return None

def plot_training_curves(bert_data, mixtext_data, uda_data, dataset_name, save_dir='plots'):
    """Plot training curves comparing BERT, MixText, and UDA models."""
    # Create a 3x2 grid of subplots with smaller size
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    
    # Define colors for consistent plotting
    colors = {'BERT': '#1f77b4', 'MixText': '#ff7f0e', 'UDA': '#2ca02c'}
    
    # Plot for each labeled data size
    for idx, n_labeled in enumerate(['10', '200', '2500']):
        # Get data for all models
        bert_exp = bert_data[n_labeled]
        mixtext_exp = mixtext_data[n_labeled]
        uda_exp = uda_data[n_labeled]
        epochs = np.arange(len(bert_exp))
        
        # Left column: Training Loss
        ax_loss = axes[idx, 0]
        ax_loss.plot(epochs, bert_exp['train_loss'], 
                    label='BERT', color=colors['BERT'], marker='o', markersize=3)
        ax_loss.plot(epochs, mixtext_exp['train_loss'], 
                    label='MixText', color=colors['MixText'], marker='s', markersize=3)
        ax_loss.plot(epochs, uda_exp['train_loss'], 
                    label='UDA', color=colors['UDA'], marker='^', markersize=3)
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Training Loss')
        ax_loss.set_title(f'{dataset_name} - Training Loss (n={n_labeled})')
        ax_loss.legend()
        
        # Right column: Validation Accuracy
        ax_acc = axes[idx, 1]
        ax_acc.plot(epochs, bert_exp['val_acc'], 
                   label='BERT', color=colors['BERT'], marker='o', markersize=3)
        ax_acc.plot(epochs, mixtext_exp['val_acc'], 
                   label='MixText', color=colors['MixText'], marker='s', markersize=3)
        ax_acc.plot(epochs, uda_exp['val_acc'], 
                   label='UDA', color=colors['UDA'], marker='^', markersize=3)
        ax_acc.set_xlabel('Epoch')
        ax_acc.set_ylabel('Validation Accuracy')
        ax_acc.set_title(f'{dataset_name} - Validation Accuracy (n={n_labeled})')
        ax_acc.legend()
    
    plt.tight_layout()
    
    # Create plots directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save only PNG with descriptive filename
    filename = f'{dataset_name.lower()}_bert_mixtext_uda_comparison.png'
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

def process_dataset(dataset_name):
    """Process and plot results for a specific dataset."""
    # Base directories for experiments
    bert_base_dir = f'experiments_bert_{dataset_name.lower()}'
    mixtext_base_dir = f'experiments_mixtext_{dataset_name.lower()}'
    uda_base_dir = f'experiments_uda_{dataset_name.lower()}'
    
    # Load experiment data
    bert_data = {}
    mixtext_data = {}
    uda_data = {}
    n_labeled_sizes = ['10', '200', '2500']
    
    # Find and load BERT experiments
    for n_labeled in n_labeled_sizes:
        bert_dir = find_experiment_dir(bert_base_dir, n_labeled, 'bert')
        if not bert_dir:
            print(f"Warning: Could not find BERT experiment directory for {dataset_name} with n={n_labeled}")
            return
        try:
            bert_data[n_labeled] = load_experiment_data(bert_dir)
        except FileNotFoundError:
            print(f"Warning: Could not find BERT data file in {bert_dir}")
            return
    
    # Find and load MixText experiments
    for n_labeled in n_labeled_sizes:
        mixtext_dir = find_experiment_dir(mixtext_base_dir, n_labeled, 'mixtext')
        if not mixtext_dir:
            print(f"Warning: Could not find MixText experiment directory for {dataset_name} with n={n_labeled}")
            return
        try:
            mixtext_data[n_labeled] = load_experiment_data(mixtext_dir)
        except FileNotFoundError:
            print(f"Warning: Could not find MixText data file in {mixtext_dir}")
            return
    
    # Find and load UDA experiments
    for n_labeled in n_labeled_sizes:
        uda_dir = find_experiment_dir(uda_base_dir, n_labeled, 'uda')
        if not uda_dir:
            print(f"Warning: Could not find UDA experiment directory for {dataset_name} with n={n_labeled}")
            return
        try:
            uda_data[n_labeled] = load_experiment_data(uda_dir)
        except FileNotFoundError:
            print(f"Warning: Could not find UDA data file in {uda_dir}")
            return
    
    # Create comparison plots
    plot_training_curves(bert_data, mixtext_data, uda_data, dataset_name, save_dir='plots')

def main():
    # Process each dataset
    datasets = ['ag_news', 'dbpedia', 'yahoo']
    
    for dataset in datasets:
        print(f"Processing {dataset} dataset...")
        process_dataset(dataset)
        print(f"Completed processing {dataset}. Output saved as '{dataset.lower()}_bert_mixtext_uda_comparison.png'")

if __name__ == '__main__':
    main()
