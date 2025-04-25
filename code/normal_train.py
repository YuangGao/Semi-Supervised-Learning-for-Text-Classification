import argparse
import os
import random
import math
import csv
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from transformers import AdamW
from torch.autograd import Variable
from torch.utils.data import Dataset

from read_data import *
from normal_bert import ClassificationBert

parser = argparse.ArgumentParser(description='PyTorch Base Models')

parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=4, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--batch-size-u', default=24, type=int, metavar='N',
                    help='train batchsize')

parser.add_argument('--lrmain', '--learning-rate-bert', default=0.00001, type=float,
                    metavar='LR', help='initial learning rate for bert')
parser.add_argument('--lrlast', '--learning-rate-model', default=0.001, type=float,
                    metavar='LR', help='initial learning rate for models')

parser.add_argument('--gpu', default='0,1,2,3', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

parser.add_argument('--n-labeled', type=int, default=20,
                    help='Number of labeled data')
parser.add_argument('--val-iteration', type=int, default=200,
                    help='Number of labeled data')


parser.add_argument('--mix-option', default=False, type=bool, metavar='N',
                    help='mix option')
parser.add_argument('--train_aug', default=False, type=bool, metavar='N',
                    help='aug for training data')


parser.add_argument('--model', type=str, default='bert-base-uncased',
                    help='pretrained model')

parser.add_argument('--data-path', type=str, default='data/yahoo_answers_csv/',
                    help='path to data folders')

parser.add_argument('--output-dir', type=str, default='experiments',
                    help='Directory to save experiment results')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print("gpu num: ", n_gpu)

best_acc = 0


def main():
    global best_acc
    
    # Create output directory with experiment parameters in name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f"exp_{timestamp}_nlabeled{args.n_labeled}_epochs{args.epochs}_bs{args.batch_size}"
    experiment_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Save experiment arguments
    with open(os.path.join(experiment_dir, 'args.txt'), 'w') as f:
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")
    
    # Initialize CSV file for metrics
    metrics_file = open(os.path.join(experiment_dir, 'training_metrics.csv'), 'w')
    metrics_writer = csv.writer(metrics_file)
    metrics_writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_acc', 'test_loss', 'test_acc'])
    
    train_labeled_set, train_unlabeled_set, val_set, test_set, n_labels = get_data(
        args.data_path, args.n_labeled)
    labeled_trainloader = Data.DataLoader(
        dataset=train_labeled_set, batch_size=args.batch_size, shuffle=True)
    val_loader = Data.DataLoader(
        dataset=val_set, batch_size=512, shuffle=False)
    test_loader = Data.DataLoader(
        dataset=test_set, batch_size=512, shuffle=False)

    model = ClassificationBert(n_labels).to(device)
    model = nn.DataParallel(model)
    optimizer = AdamW(
        [
            {"params": model.module.bert.parameters(), "lr": args.lrmain},
            {"params": model.module.linear.parameters(), "lr": args.lrlast},
        ])

    criterion = nn.CrossEntropyLoss()
    test_accs = []
    
    # Initialize lists to store metrics
    train_losses = []
    val_losses = []
    val_accs = []
    test_losses = []
    test_accs_list = []

    print(f"Starting training with {args.epochs} epochs...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("Training...")
        
        # Train and get training loss
        train_loss = train(labeled_trainloader, model, optimizer, criterion, epoch)
        train_losses.append(train_loss)
        
        print("\nValidating...")
        # Validate
        val_loss, val_acc = validate(
            val_loader, model, criterion, epoch, mode='Valid Stats')
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch + 1} - Validation: acc={val_acc:.4f}, loss={val_loss:.4f}")

        # Test if validation accuracy improved
        test_loss, test_acc = None, None
        if val_acc >= best_acc:
            print("\nTesting (new best validation accuracy)...")
            best_acc = val_acc
            test_loss, test_acc = validate(
                test_loader, model, criterion, epoch, mode='Test Stats ')
            test_accs.append(test_acc)
            test_losses.append(test_loss)
            test_accs_list.append(test_acc)
            print(f"Epoch {epoch + 1} - Test: acc={test_acc:.4f}, loss={test_loss:.4f}")
            
            # Save model checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'test_accs': test_accs
            }
            torch.save(checkpoint, os.path.join(experiment_dir, f'checkpoint_epoch_{epoch}.pt'))

        # Write metrics to CSV
        metrics_writer.writerow([
            epoch,
            train_loss,
            val_loss,
            val_acc,
            test_loss if test_loss is not None else '',
            test_acc if test_acc is not None else ''
        ])
        print(f"Metrics saved to CSV for epoch {epoch + 1}")

    # Save final metrics
    metrics_file.close()
    
    # Save numpy arrays of metrics
    np.save(os.path.join(experiment_dir, 'train_losses.npy'), np.array(train_losses))
    np.save(os.path.join(experiment_dir, 'val_losses.npy'), np.array(val_losses))
    np.save(os.path.join(experiment_dir, 'val_accs.npy'), np.array(val_accs))
    np.save(os.path.join(experiment_dir, 'test_losses.npy'), np.array(test_losses))
    np.save(os.path.join(experiment_dir, 'test_accs.npy'), np.array(test_accs_list))

    print('\nTraining completed!')
    print('Best validation accuracy:', best_acc)
    print('Test accuracies:', test_accs)


def validate(valloader, model, criterion, epoch, mode):
    model.eval()
    with torch.no_grad():
        loss_total = 0
        total_sample = 0
        acc_total = 0
        correct = 0
        
        print(f"\nRunning {mode}...")
        total_batches = len(valloader)
        
        for batch_idx, (inputs, targets, length) in enumerate(valloader):
            if batch_idx % 10 == 0:  # Print every 10 batches
                print(f"Processing batch {batch_idx}/{total_batches}")
                
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            _, predicted = torch.max(outputs.data, 1)

            correct += (np.array(predicted.cpu()) ==
                        np.array(targets.cpu())).sum()
            loss_total += loss.item() * inputs.shape[0]
            total_sample += inputs.shape[0]

        acc_total = correct/total_sample
        loss_total = loss_total/total_sample

    return loss_total, acc_total


def train(labeled_trainloader, model, optimizer, criterion, epoch):
    model.train()
    total_loss = 0
    num_batches = 0

    for batch_idx, (inputs, targets, length) in enumerate(labeled_trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        print('epoch {}, step {}, loss {}'.format(
            epoch, batch_idx, loss.item()))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0


if __name__ == '__main__':
    main()
