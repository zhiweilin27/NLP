import re
import numpy as np
import collections
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import time  # this library just to record the time it takes to run entire .py file, does not impact code/algorithm at all.
from tabulate import tabulate # this library just to create table format for recording all combinations of weight decay and lr in 2.4, does not impact code/algorithm at all.
from a1_p1_lin_112845768 import wordTokenizer, spacelessBPELearn, spacelessBPETokenize

def data_import(file_path):
    
    ''' this function is to import the data set with only sentence (no text_id, no sent_id, and no label)
        the target word is marked with << >>, this feature is important later on to extract the target word in the sentence
        #input: file_path - path to the file 
        #output: data - list of sentences
    '''    
    
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            sentence = parts[-1]
            data.append(sentence)
        return data
def target_word_data(file_path, target_word='reason',split=3968):
    ''' this function is to import the train_set and dev_set for the target word.
       input: file_path - path to the file 
               target_word - the target word
               split - the line number at which to split data into train and dev set
       output: train_dict -  a dictionary containing line numbers (sentences) as keys and corresponding target labels in train data (data before splitting number) 
                test_ dict - a dictionary containing line numbers (sentences) as keys and corresponding target labels in test data (data after splitting number) 

                for example {0 : reason%1:26:00::} means sentence 0 contains the target word "reason" 
    '''
    train_dict = {}
    dev_dict = {}
    with open(file_path, 'r') as file:
        for line_number, line in enumerate(file):  
            parts = line.strip().split('\t')
            if len(parts) > 2:
                label = parts[2]
                if re.match(fr'{target_word}%\d:\d\d:\d\d::', label):
                    if line_number < split:
                        train_dict[line_number] = label
                    else:
                        dev_dict[line_number] = label
        return train_dict, dev_dict
        
    
def preprocessing(data,train_dict, dev_dict, vocabulary, tokenizer_type='word'):
    ''' this function is to import the data set with only sentence (no text_id, no sent_id, and no label)
    
    input :data - list of sentences
            train_dict -  a dictionary containing line numbers (sentences) as keys and corresponding target labels in train data 
            test_ dict - a dictionary containing line numbers (sentences) as keys and corresponding target labels in test data 
            tokenizer_type - type of tokenizer is used
            vocabulary - set of 500 most frequent vocabulary in train data

    output: train_set - the training set of features and labels
             dev_set - the dev set of features and labels
    
    for example: receptor has two classes (0,1) and a set of features (1503).
                 [    (0,1,0,0,1....),  [(0,0,0,1,......,0,0,...),        ] 
                                         (0,0,0,0,......,1,0,...),
                                         (......)] 
    
    '''
    #BEGIN[ChatGPT][https://chat.openai.com/auth/login] "get unique values in dictionary values and convert them to numbers"
    unique = {target_word: i for i, target_word in enumerate(sorted(set(train_dict.values())))}
    train_data = {key: unique[value] for key, value in train_dict.items()}
    #END[ChatGPT]
    train_index = list(train_data.keys())
    train_encoding = extractLexicalFeatures(data, train_index,vocabulary, tokenizer_type=tokenizer_type)

    unique = {target_word: i for i, target_word in enumerate(sorted(set(dev_dict.values())))}
    dev_data = {key: unique[value] for key, value in dev_dict.items()}
    dev_index = list(dev_data.keys())
    dev_encoding = extractLexicalFeatures(data, dev_index,vocabulary, tokenizer_type=tokenizer_type)

    train_labels = np.array(list(train_data.values()))
    dev_labels = np.array(list(dev_data.values()))
    
    train_set = [train_labels,train_encoding]
    dev_set = [dev_labels, dev_encoding]
    
    return train_set, dev_set

def vocabulary_set(tokens,start=0,end=3968,tokenizer_type ="word"):
    '''
    inputs: tokens – list of sentence in entire data set
            start - starting line number for train data
            end - ending line number for train data
    outputs:
            vocabulary - set of 500 most frequent vocabulary in train data
    '''
    data = ' '.join(tokens[start:end])                  
    data = re.sub("<<|>>","",data)                      
    
    if tokenizer_type == 'bpe':
        vocabulary = spacelessBPELearn(data,max_vocabulary=500)
        vocabulary.append('<unk>')
    else:
        words = wordTokenizer(data)
        words = collections.Counter(words).most_common(500)
        vocabulary = [word for word, _ in words]
        vocabulary.append('<unk>')
    return vocabulary

def extractLexicalFeatures(tokens, target, vocabulary,tokenizer_type="word"):
    '''     input: tokens – a list or string of words in order 
            (in my implementation, tokens is data (list of sentence in entire data set))
                   target - the index into the list of the target word 
            (in my implementation, target is the index of sentence that contains the target word, for example sentence 0 contains the target word "reason")
                   
                   tokenizer_type - the type of tokenizer to use: 'word' or 'bpe'

            output: features - a feature vector
    '''
    feature = None
    for i in target:
         #BEGIN[ChatGPT/Cheatograph][https://chat.openai.com/auth/login and https://cheatography.com/davechild/cheat-sheets/regular-expressions/] "Find regular expression for the word insdie << >> and get index of that word"

        target_sentence = re.sub(r'<<(\w+)>>', r'\1', tokens[i])
        target_sentence = re.split(r'(\W)|\s',target_sentence)
        target_sentence = [word for word in target_sentence if word]        
        target_word = re.findall(r'<<(\w+)>>', tokens[i])[0]
        #End[ChatGPT/Cheatograph]
        j = target_sentence.index(target_word)
#         print(target_word)
        word_before = target_sentence[:j]
#         print(word_before)
        word_after = target_sentence[j+1:]
#         print(word_after)
        if tokenizer_type == 'bpe':
            before_target =spacelessBPETokenize(' '.join(word_before), vocabulary)
            after_target = spacelessBPETokenize(' '.join(word_after), vocabulary)
        else:
            before_target = wordTokenizer(' '.join(word_before))
            after_target = wordTokenizer(' '.join(word_after))
#         print(before_target)
#         print(after_target)
        target_sentence = before_target +[target_word] + after_target
#         print(target_sentence)
#         print(target_word)
#         print(target_sentence)
#         print(i)
        prev_word = target_sentence[len(before_target)-1]
        next_word = target_sentence[len(before_target)+1]
#         print(target_word)
#         print(prev_word)
#         print(next_word)
        prev_one_hot = np.zeros(len(vocabulary))
        next_one_hot = np.zeros(len(vocabulary))
        mult_hot = np.zeros(len(vocabulary))

        if prev_word in vocabulary:
            index = vocabulary.index(prev_word)
            prev_one_hot[index] = 1
        else:
            prev_one_hot[-1] = 1 
            
        if next_word in vocabulary:
            index = vocabulary.index(next_word)
            next_one_hot[index] = 1
        else:
            next_one_hot[-1] = 1

        for i in range(len(target_sentence)):
            if target_sentence[i] in vocabulary:
                mult_hot[vocabulary.index(target_sentence[i])] = 1
            else:
                mult_hot[-1] = 1
        A = np.concatenate((prev_one_hot, next_one_hot, mult_hot))        
        if feature is None:
            feature = A.reshape(1, -1)  
        else:
            feature = np.vstack((feature, A)) 
    return feature

    
class LogisticRegression(torch.nn.Module):
    def __init__(self, num_feats, num_classes, dropout_rate):
        super(LogisticRegression, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)

        self.linear = nn.Linear(num_feats, num_classes,bias=True)
        
    def forward(self, X):
        newX = self.dropout(X)
        
        return self.linear(newX) 
  
    
def trainLogReg(train_set, dev_set=None, loss_curve=False, max_epoch=30, lr=0.1, weight_decay = 10,dropout_rate=0):
    '''Train a logistic regression model using the provided train_set. 
       (note, I already convert train_corpus(raw words) into train_set(feature set) 
   
       input: train_set - the training set of features and labels
             dev_set(optinal, if wanted to obtain loss curve of dev_set) - the dev set of features and labels
             loss_curve (default is False) -  Whether to print the loss curve.
             max_epoch - Maximum number of epochs for training
             lr -  Learning rate for optimization
             weight_decay -  Weight decay (L2 penalty) for regularization
             dropout_rate - Dropout rate

       outputs: model - trained logistic regression model.
                loss_curve - would print out loss_curve if wanted
        
   '''
    
    num_feats = train_set[1].shape[1]
    num_classes = len(set(train_set[0]))
    #BEGIN[ChatGPT][https://chat.openai.com/auth/login] "find number of sample for each classes and convert them into precentage"
    num_sample=collections.Counter(train_set[0])
    weight =[(count / len(train_set[0])) for key, count in num_sample.items()]
    #END[ChatGPT]
    loss_func = nn.CrossEntropyLoss(weight=torch.Tensor(weight))

    
    model = LogisticRegression(num_feats, num_classes,dropout_rate)
    sgd = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.99, weight_decay=weight_decay)
    X_train = torch.tensor(train_set[1], dtype=torch.float)

    y_train = torch.tensor(train_set[0], dtype=torch.long)
    
    if dev_set:
        X_dev = torch.tensor(dev_set[1], dtype=torch.float)
        y_dev = torch.tensor(dev_set[0], dtype=torch.long)
    
    
    train_losses = []
    val_losses = []
    for epoch in range(1, max_epoch + 1):
        model.train()

        y_pred = model(X_train)
#         print(y_pred)
#         print(y_train)
#         _, y_pred = torch.max(y_pred, 1)

        train_loss = loss_func(y_pred, y_train)
        
        sgd.zero_grad()
        train_loss.backward()
#         for name, param in model.named_parameters():
#             if param.requires_grad:
#                 print(f"{name} gradient: {param.grad}")
        sgd.step()
        
        train_losses.append(train_loss.item()) 
        if dev_set:
            model.eval()
            with torch.no_grad():
                outputs = model(X_dev)
#                 print(outputs)
#                 print(y_dev)
                val_loss = loss_func(outputs, y_dev)
                val_losses.append(val_loss.item()) 
    if loss_curve:
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Dev Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend()
        plt.show()

    return model

def crossVal(model, train_set,dev_set):
    '''
       input: model – the model to be tested
              train_set - the training set of features and outcomes
              dev_set - the dev set of features and outcomes
       output: model_accuracies - the dev set accuracies
    '''
    outputs = model(torch.tensor(dev_set[1], dtype=torch.float))

    _, predicted_dev = torch.max(outputs, 1)
#     print(predicted_dev)
    accuracies = f1_score(dev_set[0], predicted_dev.numpy(), average='macro')
    
    return accuracies


    
def Improvements(train_set, dev_set, lr, weight_decay,dropout_rate,max_epoch=100):
    '''
    I changed momentum parameter in SGD to 0.999 and turn on the Nesterov parameter, and change lr to 0.1 and epochs to 70 for 2.5. 
    input: train_set - the training set of features and labels
           dev_set - the dev set of features and labels
           max_epoch - Maximum number of epochs for training.
           lr -  Learning rate for optimization.
           dropout_rate - Dropout rate

    output: 
           accuracy - model f1 score
    '''

 
    
    num_feats = train_set[1].shape[1]
    num_classes = len(set(train_set[0]))
    
    num_sample=collections.Counter(train_set[0])
    weight =[(count / len(train_set[0])) for key, count in num_sample.items()]
    loss_func = nn.CrossEntropyLoss(weight=torch.Tensor(weight))

    
    model = LogisticRegression(num_feats, num_classes,dropout_rate)
    sgd = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.999, weight_decay=weight_decay, nesterov=True)

    
    X_train = torch.tensor(train_set[1], dtype=torch.float)
    y_train = torch.tensor(train_set[0], dtype=torch.long)
    
    X_dev = torch.tensor(dev_set[1], dtype=torch.float)
    y_dev = torch.tensor(dev_set[0], dtype=torch.long)
        
    for epoch in range(1, max_epoch + 1):
        model.train()
        y_pred = model(X_train)
        train_loss = loss_func(y_pred, y_train)

        sgd.zero_grad()
        train_loss.backward()
        sgd.step()
    outputs = model(torch.tensor(dev_set[1], dtype=torch.float))
    _, predicted_dev = torch.max(outputs, 1)
    accuracies = f1_score(dev_set[0], predicted_dev.numpy(), average='macro')
        
    return accuracies


def main(file_path):
    

    np.random.seed(0)
    torch.manual_seed(0)
    start_time = time.time()

    data = data_import(file_path)    # list of sentence in the data set
    train_data = data[:3968]         # list of train sentence 

    vocabulary_word = vocabulary_set(data,start=0,end=3968,tokenizer_type ="word")
    feature_word = extractLexicalFeatures(train_data, [0,1,-1],vocabulary_word, tokenizer_type='word')
    print('Checkpoint 2.1:\n')
    print(f'feature vector for the first 2 and last documents of the training set for word tokenizer is: \n {feature_word}')

    vocabulary_bpe = vocabulary_set(data,start=0,end=3968,tokenizer_type ="bpe")
    feature_bpe = extractLexicalFeatures(train_data, [0,1,-1], vocabulary_bpe, tokenizer_type='bpe')
    print(f'feature vector for the first 2 and last documents of the training set for bpe tokenizer is: \n {feature_bpe}')
    print('\nCheckpoint 2.2:\n NOTE: betas for word tokenization models would be shown \n')
    target_words = ['receptor', 'reduction', 'rate', 'reserve', 'reason', 'return']

    weight_decays = [0.001, 0.01, 0.1, 1, 10, 100]
    dropout_rates = [0, 0.1, 0.2, 0.5]
    tokenizer_types = ['word', 'bpe']
    data = data_import(file_path)
    data_bpe = {}
    results = {}  
    best_f1 = 0
    best_com = None
    best_token = None
    #BEGIN[ChatGPT][https://chat.openai.com/auth/login] "how to get average f1 score across all words for each set of weight decay and dropout rate"
    #(I was inpsired by ChatGPT in creating combination table, where column is L2 and row is dropout rate, with each entry is average f1 score across 6 words)
    for tokenizer_type in tokenizer_types:
        if tokenizer_type =="bpe": 
            vocabulary = vocabulary_bpe
        else:
            vocabulary = vocabulary_word
        results[tokenizer_type] = {}
        for i in weight_decays:
            for j in dropout_rates:
                results[tokenizer_type][(i, j)] = [] 

        for word in target_words:
            train_dict, dev_dict = target_word_data(file_path, target_word=word, split=3968)
            train_set, dev_set = preprocessing(data, train_dict, dev_dict, vocabulary, tokenizer_type=tokenizer_type)
            if tokenizer_type == "bpe": data_bpe[word] = {'train_set':train_set,'dev_set':dev_set}
            for i in weight_decays:
                for j in dropout_rates:
                    model = trainLogReg(train_set=train_set, dev_set=dev_set, loss_curve=False, max_epoch=35, lr=0.05, weight_decay=i, dropout_rate=j)
                    accuracy = crossVal(model, train_set, dev_set)
                    results[tokenizer_type][(i, j)].append(accuracy)

                    if i == 10 and j == 0 and tokenizer_type == 'word':
                        print(f"Model beta for {word} with a word tokenizer weight decay = 10 and dropout rate = 0 is: \n {model.linear.weight.detach().numpy()}")                        

        for i in weight_decays:
            for j in dropout_rates:
                accuracies = results[tokenizer_type][(i, j)]
                average_accuracy = sum(accuracies) / len(accuracies)
                results[tokenizer_type][(i, j)] = average_accuracy
                if average_accuracy > best_f1:
                    best_f1 = average_accuracy
                    best_com = [i,j]
                    best_token = tokenizer_type

        if tokenizer_type =='word': 
            print('\nCheckpoint 2.3:\n Note: F1 values for both types of tokenization models will be shown with weight decay = 10 and dropout rate = 0\n')
            print(f"Average F1 score across 6 words with word tokenizer weight decay = 10 and dropout rate = 0 is: {results['word'][(10, 0)]}.")
        else:
            print(f"Average F1 score across 6 words with bpe tokenizer weight decay = 10 and dropout rate = 0 is: {results['bpe'][(10, 0)]}.")

    for tokenizer_type in tokenizer_types:
        weight_decays = sorted(set(wd for (wd, _) in results[tokenizer_type].keys()))
        dropout_rates = sorted(set(dr for (_, dr) in results[tokenizer_type].keys()))

        table_data = [[tokenizer_type] + dropout_rates]  


        for wd in weight_decays:
            row = [wd]  
            for dr in dropout_rates:
                accuracy = results[tokenizer_type].get((wd, dr), 'N/A')
                formatted_accuracy = f"{accuracy:.2f}" if accuracy != 'N/A' else accuracy
                row.append(formatted_accuracy)
            table_data.append(row)
        if tokenizer_type =='word': print('\nCheckpoint 2.4:\n')

        print(tabulate(table_data[1:], headers=table_data[0], tablefmt="grid"))
    #END[ChatGPT]

    print(f'The best average F1 score across 6 words with a set of weight decay = {best_com[0]},dropout rate = {best_com[1]}, and tokenizer = {best_token} is: {best_f1} ')



    print('\nCheckpoint 2.5:\n')

    lr = 0.1
    last_table =[]
    for word in target_words:
        data = data_bpe[word]
        train_set = data["train_set"]
        dev_set = data["dev_set"]
        i = 0.01
        j = 0.1
        accuracy = Improvements(train_set, dev_set, lr, weight_decay=i,dropout_rate=j,max_epoch=70)
        last_table.append([word,i,j,accuracy])

    columns = ['Word', 'Weight Decay', 'Dropout Rate', 'F1 Score']  
    accuracies = [i[3] for i in last_table]
    average_accuracies = sum(accuracies)/len(accuracies)
    print('f1 scores for each word with a set of weight decay=0.01, dropout_rate=0.1, tokenizer=bpe, lr=0.1 and epochs=70, momentum=0.999, Nestero=True in 2.5 is:\n',tabulate(last_table, headers=columns, tablefmt="grid"))
    print("Average of best f1 scores across 6 target words is :",average_accuracies)


    end_time = time.time()
    total_time = end_time - start_time
    print('\nTotal time taken for entire part 2:', total_time,'\n')

    print('For 2.5, I used Nesterov momentum factor,0.999 and Nesterov=True, in the learning rate = 0.1 in SGD and changed the max epoch to 70.') 
    print('The reason for these changes is that I believe the original SGD converges too slowly for this logistic regression. So, using Nesterov momentum to accelerate the convergence, and also training the model longer with more epochs allows the model to learn more details in the train set, which means it gets closer to the minimum point')
    
if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    file_path = "a1_wsd_24_2_10.txt"
    main(file_path)
