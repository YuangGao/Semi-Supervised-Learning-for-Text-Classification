import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
import os
import numpy as np
import glob

# Set the style without LaTeX and adjust font sizes
plt.style.use(['science', 'no-latex'])
plt.rcParams.update({
    'font.size': 8,          
    'axes.titlesize': 8,    
    'axes.labelsize': 8,     
    'xtick.labelsize': 7,    
    'ytick.labelsize': 7,    
    'legend.fontsize': 7,    
    'figure.dpi': 300,       
    'figure.figsize': (6, 7) 
})

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
    
    matching_dirs = sorted(glob.glob(pattern))
    if matching_dirs:
        return matching_dirs[-1]  # Return the last matching directory
    return None

def plot_training_curves(bert_data, mixtext_data, uda_data, dataset_name, save_dir='plots'):
    """Plot training curves comparing BERT, MixText, and UDA models."""
    # Create a 2x3 grid of subplots
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # Define colors for consistent plotting
    colors = {'BERT': '#1f77b4', 'MixText': '#ff7f0e', 'UDA': '#2ca02c'}
    
    # Plot for each labeled data size
    for idx, n_labeled in enumerate(['10', '200', '2500']):
        # Get data for all models
        bert_exp = bert_data[n_labeled]
        mixtext_exp = mixtext_data[n_labeled]
        uda_exp = uda_data[n_labeled]
        epochs = np.arange(len(bert_exp))
        
        # First row: Training Loss
        ax_loss = axes[0, idx]
        ax_loss.plot(epochs, bert_exp['train_loss'], 
                    label='BERT', color=colors['BERT'], marker='o', markersize=2, linewidth=1)
        ax_loss.plot(epochs, mixtext_exp['train_loss'], 
                    label='MixText', color=colors['MixText'], marker='s', markersize=2, linewidth=1)
        ax_loss.plot(epochs, uda_exp['train_loss'], 
                    label='UDA', color=colors['UDA'], marker='^', markersize=2, linewidth=1)
        ax_loss.set_xlabel('Epoch', fontsize=12)
        ax_loss.set_ylabel('Training Loss', fontsize=12)
        ax_loss.set_title(f'{dataset_name} - Training Loss (n={n_labeled})', fontsize=12)
        ax_loss.legend(frameon=False)
        
        # Second row: Validation Accuracy
        ax_acc = axes[1, idx]
        ax_acc.plot(epochs, bert_exp['val_acc'], 
                   label='BERT', color=colors['BERT'], marker='o', markersize=2, linewidth=1)
        ax_acc.plot(epochs, mixtext_exp['val_acc'], 
                   label='MixText', color=colors['MixText'], marker='s', markersize=2, linewidth=1)
        ax_acc.plot(epochs, uda_exp['val_acc'], 
                   label='UDA', color=colors['UDA'], marker='^', markersize=2, linewidth=1)
        ax_acc.set_xlabel('Epoch', fontsize=12)
        ax_acc.set_ylabel('Validation Accuracy', fontsize=12)
        ax_acc.set_title(f'{dataset_name} - Validation Accuracy (n={n_labeled})', fontsize=12)
        ax_acc.legend(frameon=False)
    
    plt.tight_layout(pad=0.5)
    
    # Create plots directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save only PNG with descriptive filename
    filename = f'{dataset_name.lower()}_bert_mixtext_uda_comparison.png'
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

def plot_best_test_accuracy(bert_data, mixtext_data, uda_data, dataset_name, save_dir='plots'):
    """Plot best test accuracy comparison across different numbers of labeled samples."""
    plt.figure(figsize=(6, 4))
    
    # Define colors for consistent plotting
    colors = {'BERT': '#1f77b4', 'MixText': '#ff7f0e', 'UDA': '#2ca02c'}
    
    # Prepare data points
    n_labeled = [10, 200, 2500]
    
    # Get best test accuracy for each model
    bert_acc = [bert_data[str(n)]['test_acc'].max() for n in n_labeled]
    mixtext_acc = [mixtext_data[str(n)]['test_acc'].max() for n in n_labeled]
    uda_acc = [uda_data[str(n)]['test_acc'].max() for n in n_labeled]
    
    # Plot lines with markers
    plt.plot(n_labeled, bert_acc, label='BERT', color=colors['BERT'], marker='o', markersize=4, linewidth=1.5)
    plt.plot(n_labeled, mixtext_acc, label='MixText', color=colors['MixText'], marker='s', markersize=4, linewidth=1.5)
    plt.plot(n_labeled, uda_acc, label='UDA', color=colors['UDA'], marker='^', markersize=4, linewidth=1.5)
    
    plt.xlabel('Number of Labeled Samples', fontsize=16)
    plt.ylabel('Best Test Accuracy', fontsize=16)
    plt.title(f'{dataset_name} - Best Test Accuracy Comparison', fontsize=16)
    plt.legend(frameon=False)
    
    # Set x-axis to log scale for better visualization
    plt.xscale('log')
    
    # Create plots directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the plot
    filename = f'{dataset_name.lower()}_best_test_accuracy_comparison.png'
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
    plot_best_test_accuracy(bert_data, mixtext_data, uda_data, dataset_name, save_dir='plots')

def main():
    # Process each dataset
    datasets = ['ag_news', 'dbpedia', 'yahoo', 'imdb']
    
    for dataset in datasets:
        print(f"Processing {dataset} dataset...")
        process_dataset(dataset)
        print(f"Completed processing {dataset}. Output saved as '{dataset.lower()}_bert_mixtext_uda_comparison.png'")

if __name__ == '__main__':
    main()
