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
from uda import UDA
from time import time

parser = argparse.ArgumentParser(description='PyTorch UDA')
parser.add_argument('--data-path', type=str, default='yahoo_answers_csv/')
parser.add_argument('--n-labeled', type=int, default=10)
parser.add_argument('--un-labeled', type=int, default=5000)
parser.add_argument('--val-iteration', type=int, default=1000)
parser.add_argument('--model', type=str, default='bert-base-uncased')
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch-size', type=int, default=4)
parser.add_argument('--batch-size-u', type=int, default=24)
parser.add_argument('--lambda-u', type=float, default=1.0)
parser.add_argument('--T', type=float, default=0.9, help='sharpening temperature')
parser.add_argument('--lrmain', type=float, default=5e-6)
parser.add_argument('--lrlast', type=float, default=5e-4)
parser.add_argument('--rampup-length', type=int, default=5,
                    help='length of rampup period for lambda_u')
parser.add_argument('--train_aug', default=False, type=bool, metavar='N',
                    help='augment labeled training data')
parser.add_argument('--output-dir', type=str, default='uda_exp/',
                    metavar='PATH',
                    help='Directory to save experiment results')
# parser.add_argument('--patience', type=int, default=5)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# n_gpu = torch.mps.device_count()
print("GPU num: ", n_gpu)

best_acc = 0
total_steps = 0
flag = 0

def main():
    global best_acc
    global total_steps
    global flag
    # Create output directory with experiment parameters in name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f"exp_{timestamp}_nlabeled{args.n_labeled}_unlabeled{args.un_labeled}_epochs{args.epochs}_bs{args.batch_size}"
    experiment_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Save experiment arguments
    with open(os.path.join(experiment_dir, 'args.txt'), 'w') as f:
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")
    
    # Initialize CSV file for metrics
    metrics_file = open(os.path.join(experiment_dir, 'training_metrics.csv'), 'w')
    metrics_writer = csv.writer(metrics_file)
    metrics_writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_acc', 'test_loss', 'test_acc', 'Lx', 'Lu'])
    
    # Read dataset and build dataloaders
    train_labeled_set, train_unlabeled_set, val_set, test_set, n_labels = get_data(
        args.data_path, args.n_labeled, args.un_labeled, model=args.model, train_aug=args.train_aug)

    labeled_trainloader = Data.DataLoader(train_labeled_set, batch_size=args.batch_size, shuffle=True)
    unlabeled_trainloader = Data.DataLoader(train_unlabeled_set, batch_size=args.batch_size_u, shuffle=True)
    val_loader = Data.DataLoader(val_set, batch_size=512, shuffle=False)
    test_loader = Data.DataLoader(test_set, batch_size=512, shuffle=False)

    # Define the model, set the optimizer
    model = UDA(
        num_labels=n_labels,
        model_name=args.model,
        lambda_u=args.lambda_u,
        T=args.T
    ).to(device)
    model = nn.DataParallel(model)

    # Optimizer & Scheduler
    optimizer = AdamW([
        {"params": model.module.bert.parameters(), "lr": args.lrmain},
        {"params": model.module.linear.parameters(), "lr": args.lrlast}
    ])

    scheduler = None
    #WarmupConstantSchedule(optimizer, warmup_steps=num_warmup_steps)
    
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    Lx_list = []
    Lu_list = []
    val_losses = []
    val_accs = []
    test_accs = []
    test_losses = []
    

    # Start training
    for epoch in range(args.epochs):
        start_time = time()
        print(f"Epoch {epoch + 1}/{args.epochs}")

        train_loss, Lx, Lu = train(labeled_trainloader, unlabeled_trainloader, model, optimizer,
              scheduler, criterion, epoch, n_labels, args.train_aug)
        train_losses.append(train_loss)
        Lx_list.append(Lx)
        Lu_list.append(Lu)

        val_loss, val_acc = validate(val_loader, model, criterion, epoch, mode='Valid Stats')
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # print(f"Epoch {epoch}: val_acc = {val_acc:.4f}, val_loss = {val_loss:.4f}")

        if val_acc >= best_acc:
            best_acc = val_acc
            test_loss, test_acc = validate(
                test_loader, model, criterion, epoch, mode='Test Stats')
            test_accs.append(test_acc)
            test_losses.append(test_loss)
            # print(f"Epoch {epoch}: test_acc = {test_acc:.4f}, test_loss = {test_loss:.4f}")
            
            # Save model checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'test_accs': test_accs
            }
            torch.save(checkpoint, os.path.join(experiment_dir, f'checkpoint_epoch_{epoch}.pt'))

        end_time = time()
        print(f"Epoch {epoch} completed in {(end_time - start_time)/60:.2f} mins")
        
        # Write metrics to CSV
        metrics_writer.writerow([
            epoch,
            train_loss,
            val_loss,
            val_acc,
            test_loss if test_loss is not None else '',
            test_acc if test_acc is not None else '',
            Lx,
            Lu
        ])

    # Save final metrics
    metrics_file.close()
    
    # Save numpy arrays of metrics
    np.save(os.path.join(experiment_dir, 'train_losses.npy'), np.array(train_losses))
    np.save(os.path.join(experiment_dir, 'val_losses.npy'), np.array(val_losses))
    np.save(os.path.join(experiment_dir, 'val_accs.npy'), np.array(val_accs))
    np.save(os.path.join(experiment_dir, 'test_losses.npy'), np.array(test_losses))
    np.save(os.path.join(experiment_dir, 'test_accs.npy'), np.array(test_accs))
    np.save(os.path.join(experiment_dir, 'Lx_list.npy'), np.array(Lx_list))
    np.save(os.path.join(experiment_dir, 'Lu_list.npy'), np.array(Lu_list))


    print("Finished training!")
    print(f"Best val acc = {best_acc:.4f}")
    print('Test acc:')
    print(test_accs)

def train(labeled_trainloader, unlabeled_trainloader, model, optimizer, scheduler, criterion, epoch, n_labels, train_aug=False):
    global total_steps
    global flag

    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)
    model.train()

    total_loss = 0.0
    Lx_total = 0.0
    Lu_total = 0.0
    lambda_u = args.lambda_u * linear_rampup(epoch, args.rampup_length)
    print(f"Epoch {epoch}: lambda_u = {lambda_u}")
    
    # epoch_progress = (epoch + 1) / args.epochs  # For TSA

    for batch_idx in range(args.val_iteration):
        total_steps += 1
        
        # Load labeled batch
        try:
            inputs_x, targets_x, inputs_x_length = next(labeled_train_iter)
        except StopIteration:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x, inputs_x_length = next(labeled_train_iter)
        
        # Load unlabeled batch
        try:
            (inputs_u, inputs_u2, inputs_ori), (length_u, length_u2, length_ori) = next(unlabeled_train_iter)
        except StopIteration:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_u, inputs_u2, inputs_ori), (length_u, length_u2, length_ori) = next(unlabeled_train_iter)

        # Move data to device
        inputs_x, targets_x = inputs_x.to(device), targets_x.to(device)
        inputs_u = inputs_u.to(device)
        inputs_u2 = inputs_u2.to(device)
        inputs_ori = inputs_ori.to(device)
        
        # Create attention masks from lengths
        def create_mask(inputs, lengths):
            mask = torch.zeros_like(inputs)
            for i, length in enumerate(lengths):
                mask[i, :length] = 1
            return mask
        
        mask_x = create_mask(inputs_x, inputs_x_length)
        mask_u = create_mask(inputs_u, length_u)
        mask_u2 = create_mask(inputs_u2, length_u2)
        mask_ori = create_mask(inputs_ori, length_ori)

        # Forward pass for labeled data
        # sup_loss, logits_x = model(
        #     (inputs_x, mask_x), 
        #     labeled=True,
        #     targets=targets_x,
        #     epoch_progress=epoch_progress,
        #     tsa_schedule='linear'
        # )
        logits_x = model((inputs_x, mask_x), labeled=True, targets=targets_x)
        sup_loss = criterion(logits_x, targets_x)

        # Forward pass for unlabeled data
        with torch.no_grad():
            logits_ori = model((inputs_ori, mask_ori), labeled=False)
        
        logits_u = model((inputs_u, mask_u), labeled=False)
        logits_u2 = model((inputs_u2, mask_u2), labeled=False)
        
        # Compute unsupervised losses
        loss_orig_de = compute_kl_loss(logits_ori, logits_u, T=args.T)
        loss_orig_ru = compute_kl_loss(logits_ori, logits_u2, T=args.T)
        loss_de_ru = compute_kl_loss(logits_u, logits_u2, T=args.T)
        
        unsup_loss = (loss_orig_de + loss_orig_ru + loss_de_ru) / 3
        loss_total = sup_loss + lambda_u * unsup_loss

        # Backward pass
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        
        # Track losses
        total_loss += loss_total.item()
        Lx_total += sup_loss.item()
        Lu_total += unsup_loss.item()

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch} Step {batch_idx}: "
                  f"Loss={loss_total.item():.4f} "
                  f"(Lx={sup_loss.item():.4f} "
                  f"Lu={unsup_loss.item():.4f})")

    return total_loss / args.val_iteration, Lx_total / args.val_iteration, Lu_total / args.val_iteration

def validate(valloader, model, criterion, epoch, mode='Valid Stats'):
    model.eval()
    loss_total = 0.0
    correct = 0
    total_sample = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets, lengths) in enumerate(valloader):
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Create attention mask from lengths
            attention_mask = torch.zeros_like(inputs)
            for i, length in enumerate(lengths):
                attention_mask[i, :length] = 1
            
            # Forward pass
            logits = model((inputs, attention_mask), labeled=False).to(device)
            loss = criterion(logits, targets)
            
            # Calculate accuracy
            _, predicted = torch.max(logits.data, 1)
            correct += (predicted == targets).sum().item()
            total_sample += targets.size(0)
            loss_total += loss.item() * targets.size(0)
            
            # Print sample predictions for first batch
            if batch_idx == 0 and mode == 'Valid Stats':
                print("Sample predictions:")
                print("Predicted:", predicted[:10].cpu().numpy())
                print("Ground truth:", targets[:10].cpu().numpy())
    
    # Calculate metrics
    avg_loss = loss_total / total_sample
    accuracy = correct / total_sample
    
    print(f"{mode} - Epoch {epoch}: "
          f"Loss = {avg_loss:.4f}, "
          f"Accuracy = {accuracy:.4f}")
    
    return avg_loss, accuracy

def compute_kl_loss(p_logits, q_logits, T):
    p = F.softmax(p_logits / T, dim=1)
    q = F.log_softmax(q_logits / T, dim=1)
    return F.kl_div(q, p, reduction='batchmean')

def linear_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)

if __name__ == '__main__':
    main()