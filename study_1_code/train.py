import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '1' # Set the visible CUDA, here we use the second GPU

import sys
import subprocess  # Make sure this is at the top, before its first use
import click

# required packages
# "torch", "transformers", "numpy", "matplotlib", "pandas", "scikit-learn", "scipy", "tqdm"

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset, Subset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import pandas as pd
from tqdm import tqdm
import pickle as pkl
np.random.seed = 666

# our utility functions
from data_utils import *
from model_utils import *






def train_the_model(train_mode, dataset, with_prompt, device):
    assert train_mode in ['finetune', 'transfer']
    assert dataset in ['huggingface', 'kaggle', 'combined']

    if with_prompt:
        pmt = 'with_prompt'
    else:
        pmt = 'no_prompt'
        
    data_path_huggingface = './datasets/huggingface_task2/'
    data_path_kaggle = './datasets/kaggle_task2/'

    model_save_path = f'./checkpoints/{train_mode}-{dataset}-{pmt}.pt'
    
    # Load training dataset
    train_dataframe_huggingface = pd.read_csv(os.path.join(data_path_huggingface, 'train.csv'))    
    train_dataframe_kaggle = pd.read_csv(os.path.join(data_path_kaggle, 'train.csv'))

    if dataset == 'huggingface':
        train_dataframe = train_dataframe_huggingface
    elif dataset == 'kaggle':
        train_dataframe = train_dataframe_kaggle
    elif dataset == 'combined':
        train_dataframe = pd.concat([train_dataframe_huggingface, train_dataframe_kaggle], axis=0) # combined dataset

    print('Training Samples: ', train_dataframe.shape[0])

    # Prepare dataset object for training
    train_dataset = dataframe_to_dataset(train_dataframe, with_prompt)

    # training data loader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    # some hyper-parameters
    total_epochs = 20
    num_labels = 20
    
    # create the model
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=20)  
    model.to(device)
    model.train()

    if train_mode == 'finetune':
        initial_lr = 2e-5
        minimum_lr = 1e-6
        lr_decay = 0.8
        optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr)
    elif train_mode == 'transfer':
        initial_lr = 1e-2
        minimum_lr = 1e-6
        lr_decay = 0.5
        model.eval()
        model.classifier.train()
        optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=initial_lr)

    best_correlation = 0
    for epoch in range(total_epochs):  # Number of epochs
    
        # Update learning rate
        new_lr = max(minimum_lr, initial_lr*(lr_decay**max(0,epoch-5)) ) # reduce learning rate after 5 epochs
        update_learning_rate(optimizer, new_lr)
    
        # Training Loop
        for step, batch in enumerate(tqdm(train_loader)):

            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
            outputs = model(**inputs)  #.loss.backward()
            
            # compute loss
            loss_classification = outputs.loss
            loss = loss_classification # only use classification loss
    
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    # save model
    torch.save(model.state_dict(),  model_save_path)



# Command Line Arguments
@click.command()
@click.option('--train_mode', help='train mode either [transfer] or [finetune]',  required=True)
@click.option('--dataset', help='dataset either [huggingface] or [kaggle] or [combined]',  required=True)
@click.option('--cuda', help='cuda device [0] or [1]', default=0, type=int, required=False)
@click.option('--with_prompt', help='', is_flag=True)
def main(**kwargs):
    train_mode = kwargs['train_mode']
    dataset = kwargs['dataset']
    with_prompt = kwargs['with_prompt']
    cuda = kwargs['cuda']

    print(f'CUDA: {cuda}')
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{cuda}' # Set the visible CUDA, here we use the second GPU
    device = check_device()
    
    print(f'Training Mode {train_mode}, Dataset {dataset}, with_prompt {with_prompt}')


    train_the_model(train_mode, dataset, with_prompt, device)
    

if __name__ == "__main__":
    main()






