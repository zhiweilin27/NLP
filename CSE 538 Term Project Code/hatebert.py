# Team Members: Zhiwei Lin, Sai Chaddha
# Description: This file trains a HateBERT model on our dataset, being the conversation dataset from huggingface. After training the model it saves the model and tokenizer to a directory.  This performs binomial classification on the data, depending on whether a given input is toxic (1) or nontoxic (0).
# We are running the system from Google Colab with a T4 or A100 GPU.

# We submitted the code on time; however, we weren't aware that we needed to cite our code since the project assignment didn't include instructions to do so.

from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import re
import unicodedata
from io import open
import torch
import torch.nn.functional as F
import numpy as np
import tqdm
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import f1_score
import argparse
from sklearn.utils.class_weight import compute_class_weight


from extract_data import *
from normalized import *

# Vector Semantics [II. Semantics]
class tokenData(Dataset):
    def __init__(self, texts, targets, tokenizer, max_len):

        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.targets = targets

    def __len__(self):

        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        text = " ".join(text.split())
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.targets[idx], dtype=torch.long)
        }

# Begin[Colab][https://colab.research.google.com/github/Ankur3107/colab_notebooks/blob/master/classification/BERT_Fine_Tuning_Sentence_Classification_v2.ipynb#scrollTo=RqfmWwUR_Sox]
# 'how to fine tuning BERT model' 
# There are generic code, but I did look up this webiste for comfirmation, as it teaches how to implement BERT step by step. 
def train(model, train_dataset, val_dataset, learning_rate, batch_size, epochs, weight_tensor=None):
    train_loader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=batch_size
    )

    validation_loader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=batch_size
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss(weight = weight_tensor)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    for epoch_num in range(epochs):
        total_acc_train = 0
        total_loss_train = 0
        total_train_samples = 0

        model.train()
        for traindata in tqdm(train_loader):
            input_id = traindata['ids'].to(device)
            mask = traindata['mask'].to(device)
            target = traindata['targets'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_id, attention_mask=mask)
            logits = outputs.logits
            loss = criterion(logits, target)

            loss.backward()
            optimizer.step()

            acc = (logits.argmax(dim=1) == target).sum().item()
            total_acc_train += acc
            total_loss_train += loss.item()
            total_train_samples += target.size(0)

            avg_train_loss = total_loss_train / total_train_samples
            avg_train_acc = total_acc_train / total_train_samples

        total_acc_val = 0
        total_loss_val = 0
        total_val_samples = 0

        preds = []
        targets = []

        model.eval()
        with torch.no_grad():
            for valdata in validation_loader:
                input_id = valdata['ids'].to(device)
                mask = valdata['mask'].to(device)
                target = valdata['targets'].to(device)

                outputs = model(input_ids=input_id, attention_mask=mask)
                logits = outputs.logits
                batch_loss = criterion(logits, target)
                total_loss_val += batch_loss.item()

                predicted_labels = logits.argmax(dim=1)
                acc = (predicted_labels == target).sum().item()
                total_acc_val += acc
                total_val_samples += target.size(0)

                preds.extend(predicted_labels.cpu().numpy())
                targets.extend(target.cpu().numpy())

        avg_val_loss = total_loss_val / total_val_samples
        avg_val_acc = total_acc_val / total_val_samples

        f1 = f1_score(targets, preds, average='macro')

        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {avg_train_loss:.3f} '
            f'| Train Accuracy: {avg_train_acc:.3f} '
            f'| Val Loss: {avg_val_loss:.3f} '
            f'| Val Accuracy: {avg_val_acc:.3f} '
            f'| Val F1 Score: {f1:.3f}')
# END[Colab]
def main(file_path, input_type, epochs, lr, batch_size):
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_data, val_data, _ = extract_data(file_path)
    
    if input_type == 'question':
        raw_train = train_data['conversation'].apply(lambda x: x[0]['content']) # question
        train_target = train_data['target']
        
        raw_val = val_data['conversation'].apply(lambda x: x[0]['content']) # question
        val_target = val_data['target']
    elif input_type == 'response':
        raw_train = train_data['conversation'].apply(lambda x: x[1]['content']) # response
        train_target = train_data['target']
        
        raw_val = val_data['conversation'].apply(lambda x: x[1]['content']) # response
        val_target = val_data['target']
    else:
        raise ValueError('The input can only be "question" or "response".') 

    class_labels = train_target.unique()
    class_weights = compute_class_weight('balanced', classes=class_labels, y=train_target)

    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)    
    train_lines = []
    val_lines = []

    for line in raw_train:
        train_lines.append(normalizeString(line))
    
    for line in raw_val:
        val_lines.append(normalizeString(line))
    
    # Tokenization [I. Syntax]
    tokenizer = AutoTokenizer.from_pretrained("GroNLP/hateBERT")
    # Masked Language Modeling [III. Language Modeling]
    model = AutoModelForSequenceClassification.from_pretrained("GroNLP/hateBERT" , num_labels = 2)

    train_dataset = tokenData(train_lines, train_target, tokenizer, max_len=100)
    val_dataset = tokenData(val_lines, val_target, tokenizer, max_len=100)
    
    lr = lr
    batch_size = batch_size
    epochs = epochs
    
    train(model, train_dataset,val_dataset, lr, batch_size, epochs, weight_tensor)


    output_dir = './model_hatebert_{}_save/'.format(input_type)

    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    model_to_save = model.module if hasattr(model, 'module') else model  
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process training parameters.')
    parser.add_argument('-file', '--file_path', type=str, required=True,
                        help='The path to the file to be processed')
    parser.add_argument('-input', '--input_type', type=str, required=True, choices=['question', 'response'],
                        help='The type of input, either "question" or "response"')
    parser.add_argument('-epochs', '--epochs', type=int, default=5,
                        help='Number of training epochs, default is 5')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-5,
                        help='Learning rate, default is 1e-6')
    parser.add_argument('-bs', '--batch_size', type=int, default=64,
                        help='Batch size, default is 64')
    
    args = parser.parse_args()
    
    main(args.file_path, args.input_type, args.epochs, args.learning_rate, args.batch_size)

 #example  
 #!python hatebert.py -file '/content/drive/MyDrive/Colab Notebooks/Project/train-00000-of-00006-4feeb3f83346a0e9.parquet' -input "response" -epochs 5 -lr 1e-5 -bs 64


