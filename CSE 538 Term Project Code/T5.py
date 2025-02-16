# Team Members: Zhiwei Lin, Sai Chaddha
# Description: This file trains a T5 model for classification on toxic and non-toxic conversation data from huggingface.  The output is generating either 'positive' (toxic) or 'negative' (non-toxic) depending on the input sentence.  After training the model it saves the model and tokenizer to a directory.  We try to balance the batches by weighting them to make sure they have the same amount of toxic and non-toxic data.
# We are running the system from Google Colab with a T4 or A100 GPU.

# We submitted the code on time; however, we weren't aware that we needed to cite our code since the project assignment didn't include instructions to do so.
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler
from transformers import T5Tokenizer, T5ForConditionalGeneration,AdamW, get_linear_schedule_with_warmup
import time
import datetime
import random
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import optuna
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import TPESampler
import argparse
import os 


from extract_data import *
from normalized import *

# require bleow package downloaded 

# !pip install optuna 


### Convert data to input_id and masked attention
# Vector Semantics [II. Semantics]
class TokenData(Dataset):
    def __init__(self, texts, targets, tokenizer, text_max_len=100, target_max_len = 2):

        self.texts = texts
        self.tokenizer = tokenizer
        self.text_max_len = text_max_len
        self.target_max_len = target_max_len
        self.targets = targets

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        text = " ".join(text.split())  
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.text_max_len,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        outputs = self.tokenizer.encode_plus(
            str(self.targets[idx]),
            add_special_tokens=True,
            max_length=self.target_max_len,
            truncation=False,
            return_attention_mask=True,
            return_tensors='pt'
        )

        body_ids = inputs['input_ids'].squeeze().clone().detach().long()
        body_mask = inputs['attention_mask'].squeeze().clone().detach().long()
        target_ids = outputs['input_ids'].squeeze().clone().detach().long()
        target_mask = outputs['attention_mask'].squeeze().clone().detach().long()


        return {
            'body_ids': body_ids,
            'body_mask': body_mask,
            'target_ids': target_ids,
            'target_mask': target_mask,
        }

# BEGIN[Github]['https://seekinginference.com/applied_nlp/t5-class.html']'How to deal with inbalanced classification in T5'
### helper function to count target distribution inside tensor data sets
def target_count(tensor_dataset):
    count0 = 0
    count1 = 0
    total = []
    for i in tensor_dataset:
        # for positive
        if torch.all(torch.eq(i['target_ids'], torch.tensor([1465, 1]))):
            count0 += 1
        # for negative
        elif torch.all(torch.eq(i['target_ids'], torch.tensor([2841, 0]))):
            count1 += 1
    total.append(count0)
    total.append(count1)
    return torch.tensor(total)


### prepare weighted sampling for imbalanced classification
def create_sampler(target_tensor, tensor_dataset):
    class_sample_count = target_count(tensor_dataset)
    weight = 1. / class_sample_count.float()
    new_batch = []
    for i in tensor_dataset:
        if torch.all(torch.eq(i['target_ids'], torch.tensor([1465, 1]))):
            new_batch.append(0)
        elif torch.all(torch.eq(i['target_ids'], torch.tensor([2841, 0]))):
            new_batch.append(1)
    samples_weight = torch.tensor([weight[t] for t in new_batch])
    sampler = torch.utils.data.WeightedRandomSampler(weights=samples_weight,
                                                     num_samples=len(samples_weight),
                                                     replacement=True)
    return sampler
# END[Github]

### train T5 model
def train(model, train_dataloader, val_dataloader, optimizer ,scheduler, device, epochs):
    # Classification [IV. Applications]
    training_stats = []

    for epoch in range(epochs):
        model.train()
        train_total_loss = 0

        for step, batch in enumerate(train_dataloader):
            b_input_ids = batch["body_ids"].to(device)
            b_input_mask = batch["body_mask"].to(device)
            b_target_ids = batch["target_ids"].to(device)
            b_target_mask = batch["target_mask"].to(device)

            optimizer.zero_grad()

            outputs = model(
                input_ids=b_input_ids,
                attention_mask=b_input_mask,
                labels=b_target_ids,
                decoder_attention_mask=b_target_mask
            )
            loss = outputs.loss
            train_total_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()

        avg_train_loss = train_total_loss / len(train_dataloader)

        model.eval()
        total_valid_loss = 0

        with torch.no_grad():
            for step, batch in enumerate(val_dataloader):
                b_input_ids = batch["body_ids"].to(device)
                b_input_mask = batch["body_mask"].to(device)
                b_target_ids = batch["target_ids"].to(device)
                b_target_mask = batch["target_mask"].to(device)

                outputs = model(
                    input_ids=b_input_ids,
                    attention_mask=b_input_mask,
                    labels=b_target_ids,
                    decoder_attention_mask=b_target_mask
                )
                loss = outputs.loss
                total_valid_loss += loss.item()

        avg_val_loss = total_valid_loss / len(val_dataloader)

        training_stats.append({
            'epoch': epoch + 1,
            'Train Loss': avg_train_loss,
            'Validation Loss': avg_val_loss
        })

        print(f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.5f}, Validation Loss: {avg_val_loss:.5f}")

    return training_stats

def main(file_path, input_type, epochs, learning_rate, batch_size):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Applying LMs [III. Language Modeling]
    model = T5ForConditionalGeneration.from_pretrained('t5-base').to(device)
    # Tokenization [I. Syntax]
    tokenizer = T5Tokenizer.from_pretrained('t5-base')

    train_data, val_data, _ = extract_data(file_path) # extract dataset 
    # train_data = train_data[0:10000]
    # val_data = val_data[0:100]

    # convert target 0 (non-toxic) to negative and target 1 (toxic) postive for text to text generation 
    mapping = {0: 'negative', 1: 'positive'}
    train_data['target'] = train_data['target'].map(mapping)
    val_data['target'] = val_data['target'].map(mapping)

    # two types of input text - first one is only question, which is used for base model
    # second one is both question and response by bot, which is used for improved model
    if input_type == 'question':
        train_texts = train_data['conversation'].apply(lambda x: x[0]['content'])
        val_texts = val_data['conversation'].apply(lambda x: x[0]['content'])
    elif input_type == 'question_and_response':
        train_texts = train_data['conversation'].apply(lambda x: x[0]['content'] + " " + x[1]['content'])
        val_texts = val_data['conversation'].apply(lambda x: x[0]['content'] + " " + x[1]['content'])

    # normalize the data before training
    train_texts = [normalizeString(text) for text in train_texts]
    val_texts = [normalizeString(text) for text in val_texts]

    # help model the know the task by adding classifying the toxicity of: 
    task_prefix = "classify the toxicity of:"

    train_texts = [f"{task_prefix} {text}" for text in train_texts]
    val_texts = [f"{task_prefix} {text}" for text in val_texts]

    train_targets = train_data['target'].tolist()
    val_targets = val_data['target'].tolist()

    # convert data into vectors (input_ids and masked attention)
    train_dataset = TokenData(train_texts, train_targets, tokenizer)
    val_dataset = TokenData(val_texts, val_targets, tokenizer)
    
    # helps to adjust the inbalanced weight
    train_sampler = create_sampler(target_count(train_dataset), train_dataset)

  
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True) 

    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    total_steps = len(train_dataloader) * epochs
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # training 
    training_stats = train(model, train_dataloader, val_dataloader,optimizer , scheduler, device, epochs)
    # save the trained model
    output_dir = f'./model_T5_{input_type}_save/'
    os.makedirs(output_dir, exist_ok=True)

    model_to_save = model.module if hasattr(model, 'module') else model  
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process training parameters.')
    parser.add_argument('-file', '--file_path', type=str, required=True, help='The path to the file to be processed')
    parser.add_argument('-input', '--input_type', type=str, required=True, choices=['question', 'question_and_response'],
                        help='The type of input, either "question" or "question_and_response"')
    parser.add_argument('-epochs', '--epochs', type=int, default=5, help='Number of training epochs, default is 5')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-5, help='Learning rate, default is 1e-5')
    parser.add_argument('-bs', '--batch_size', type=int, default=64, help='Batch size, default is 64')

    args = parser.parse_args()
    main(args.file_path, args.input_type, args.epochs, args.learning_rate, args.batch_size)