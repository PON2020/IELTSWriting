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


# Initialize the tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Encoding and decoding functions for ordinal regression
def encode_score(score):
    return int(score * 2)

def decode_score(encoded_score):
    return encoded_score / 2.0

# Function to tokenize and encode the dataset
def encode_data(df, with_prompt=True):
    input_ids = []
    attention_masks = []
    for _, row in df.iterrows():
        if with_prompt:
            text = row['Question'] + " [SEP] " + row['Essay']
        else:
            text = row['Essay']
        encoded = tokenizer.encode_plus(
            text=text,
            truncation=True,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)

# Create a PyTorch dataset
class EssayDataset(Dataset):
    def __init__(self, input_ids, attention_masks, labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_masks[idx], self.labels[idx]
    
def dataframe_to_dataset(dataframe, with_prompt=True):
    # encode scores (multiple by 2, to make half points integers)
    dataframe['Encoded_Scores'] = dataframe['Overall'].apply(encode_score)
    # encode data
    input_ids, attention_masks = encode_data(dataframe, with_prompt)
    # convert scores to labels
    labels = torch.tensor(dataframe['Encoded_Scores'].values)
    # create the dataset object
    dataset = EssayDataset(input_ids, attention_masks, labels)
    return dataset    

