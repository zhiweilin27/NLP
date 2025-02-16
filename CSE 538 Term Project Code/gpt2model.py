# Team Members: Zhiwei Lin, Sai Chaddha
# Description: This file trains a GPT-2 model on our dataset, being the conversation dataset from huggingface. After training the model it saves the model and tokenizer to a directory. The user can then interact with the chatbot by running the chat() function.
# We are running the system from Google Colab with a T4 or A100 GPU.

# We submitted the code on time; however, we weren't aware that we needed to cite our code since the project assignment didn't include instructions to do so.

import itertools
import os
import random
import re
import unicodedata
from io import open
import torch
from torch import optim, nn
import torch.nn.functional as F
from torch.jit import script, trace
from torch.utils.data import Dataset, DataLoader,random_split, RandomSampler, SequentialSampler
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments,GPT2Config,AdamW, get_linear_schedule_with_warmup
import time
import datetime
import numpy as np
import tqdm
import argparse


from extract_data import *
from normalized import *

# Vector Semantics [II. Semantics]
class tokenData(Dataset):
    def __init__(self, filtered_pairs, tokenizer):
        self.X = filtered_pairs
        self.X_encoded = tokenizer(self.X, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
        self.input_ids = self.X_encoded['input_ids']
        self.attention_mask = self.X_encoded['attention_mask']

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx]

# Dialog (chatbots) [IV. Applications]
def chat():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_directory = '/content/GPT2model_save'

    tokenizer = GPT2Tokenizer.from_pretrained(save_directory)
    model = GPT2LMHeadModel.from_pretrained(save_directory).to(device)
    a = tokenizer.encode("<bot>:")[0]
    print("Chatbot is ready to talk to you! (type 'quit' to stop)")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            print("Chatbot: Goodbye!")
            break

        prompt = "<s> " + user_input + " <bot>:"
        generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
        generated = generated.to(device)
        sample_outputs = model.generate(
            generated,
            pad_token_id=tokenizer.eos_token_id,  # it generates a warning if i don't set it explicitly
            do_sample=True,
            top_k=50,
            max_length=100,
            top_p=0.95,
            num_return_sequences=1
        )
        start_index =(sample_outputs[0] == a).nonzero(as_tuple=True)[0].item()
        sample_outputs=sample_outputs[0][start_index+1:]
        response = tokenizer.decode(sample_outputs, skip_special_tokens=True)

        print("Chatbot:", response)


def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))

def main(FilePath, batch_size, epochs, learning_rate, ):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Start cleaning data")
    train_data, val_data, _ = extract_data(FilePath)

    train_data, train_targets = prompt_and_reponse(train_data)   # combine prompt and response
    val_data, train_targets  = prompt_and_reponse(val_data)

    train_lines = []
    val_lines = []
    for line in train_data:
        train_lines.append("<s> "+ normalizeString(line)+ " </s>")

    for line in val_data:
        val_lines.append("<s> "+ normalizeString(line)+ " </s>")

    # train_lines = train_lines[:30000]
    # val_lines = val_lines[:3000]
    
    # Tokenization [I. Syntax]
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<s>', eos_token='</s>', pad_token='<pad>')

    tokenizer.add_tokens(["<bot>:"])  
    configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)

    # Generative Language Modeling [III. Language Modeling]
    model = GPT2LMHeadModel.from_pretrained('gpt2', config=configuration)

    model.resize_token_embeddings(len(tokenizer))

    model = model.to(device)
    print('Start tokenizing')
    train_dataset= tokenData(train_lines, tokenizer)
    val_dataset= tokenData(val_lines, tokenizer)
    print('End tokenizing')



    train_dataloader = DataLoader(
                train_dataset,
                sampler = RandomSampler(train_dataset),
                batch_size = batch_size
            )

    validation_dataloader = DataLoader(
                val_dataset,
                sampler = SequentialSampler(val_dataset),
                batch_size = batch_size
            )


    seed_val = 123

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


    warmup_steps = 1e2
    epsilon = 1e-8

    sample_every = 1000


    optimizer = AdamW(model.parameters(),lr = learning_rate, eps = epsilon)


    total_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = warmup_steps, num_training_steps = total_steps)


    total_t0 = time.time()

    training_stats = []
  #BEGIN[Colab][https://colab.research.google.com/drive/13dZVYEOMhXhkXWfvSMVM1TTtUDrT6Aeh?usp=sharing]"how to fine tuning GPT"
  # The code below were referenced to this webiste. 
  # There are generic code, but I did look up this webiste for detail implementation, as it teaches how to implement BERT step by step. 
    model = model.to(device)
    for epoch_i in range(0, epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        t0 = time.time()

        total_train_loss = 0

        model.train()

        for step, batch in enumerate(train_dataloader):

            b_input_ids = batch[0].to(device)
            b_labels = batch[0].to(device)
            b_masks = batch[1].to(device)

            model.zero_grad()

            outputs = model(  b_input_ids,
                              labels=b_labels,
                              attention_mask = b_masks,
                              token_type_ids=None
                            )

            loss = outputs[0]

            batch_loss = loss.item()
            total_train_loss += batch_loss

            if step % sample_every == 0 and not step == 0:

                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}. Loss: {:>5,}.   Elapsed: {:}.'.format(step, len(train_dataloader), batch_loss, elapsed))

                model.eval()

                sample_outputs = model.generate(
                                        bos_token_id=random.randint(1,30000),
                                        do_sample=True,
                                        top_k=50,
                                        max_length = 100,
                                        top_p=0.95,
                                        num_return_sequences=1
                                    )
                for i, sample_output in enumerate(sample_outputs):
                      print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))

                model.train()

            loss.backward()

            optimizer.step()

            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)

        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))

        # ========================================
        #               Validation
        # ========================================

        print("")
        print("Running Validation...")

        t0 = time.time()

        model.eval()

        total_eval_loss = 0
        nb_eval_steps = 0

        for batch in validation_dataloader:

            b_input_ids = batch[0].to(device)
            b_labels = batch[0].to(device)
            b_masks = batch[1].to(device)

            with torch.no_grad():

                outputs  = model(b_input_ids,
    #                            token_type_ids=None,
                                 attention_mask = b_masks,
                                labels=b_labels)

                loss = outputs[0]

            batch_loss = loss.item()
            total_eval_loss += batch_loss

        avg_val_loss = total_eval_loss / len(validation_dataloader)

        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    print("")
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
# END[Colab]

    output_dir = './gpt2model_save/'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_to_save = model.module if hasattr(model, 'module') else model  
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    chat()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train a GPT-2 model.')
    parser.add_argument('-file', '--file_path', type=str, required=True, help='Path to the training data file')
    parser.add_argument('-bs', '--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('-epochs', '--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('-lr', '--learning_rate', type=float, default=5e-4, help='Learning rate')

    args = parser.parse_args()

    main(args.file_path, args.batch_size, args.epochs, args.learning_rate)

   # example
   # !python gpt2model.py -file '/content/drive/MyDrive/Colab Notebooks/Project/train-00000-of-00006-4feeb3f83346a0e9.parquet' -bs 64 -epochs 5 -lr 5e-4



