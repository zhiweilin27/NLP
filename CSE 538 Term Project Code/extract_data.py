# Team Members: Zhiwei Lin, Sai Chaddha
# Description: This file extracts the data from the dataset file and returns the train, validation and test data.  It also contains a function that combines the prompt and response into a single string.
# We are running the system from Google Colab with a T4 or A100 GPU.

# We submitted the code on time; however, we weren't aware that we needed to cite our code since the project assignment didn't include instructions to do so.

import pandas as pd
from sklearn.model_selection import train_test_split

# data extraction

def extract_data(FilePath):
    df = pd.read_parquet(FilePath)
    df_english = df[df['language'] == 'English'].reset_index(drop=True)
    df_english = df_english[df_english['turn'] == 1].reset_index(drop=True)
    
    targets = []
    for idx, row in df_english.iterrows():
        if all(not value for value in row['openai_moderation'][0]['categories'].values()):
            targets.append(0)
        else:
            targets.append(1)
    df_english['target'] = targets
    dataset = df_english[['conversation', 'target']]

    train_data, temp_df = train_test_split(dataset, test_size=0.3, stratify=dataset['target'], random_state=123)
    val_data, test_data = train_test_split(temp_df, test_size=0.5, stratify=temp_df['target'], random_state=123)

    return train_data.reset_index(drop=True), val_data.reset_index(drop=True), test_data.reset_index(drop=True)

# combine question and response

def prompt_and_reponse(data):
    
    token = []
    target = []
    
    for idx, row in data.iterrows():
        user_content = row['conversation'][0]['content']
        
        assistant_content = row['conversation'][1]['content']
        combined_content = user_content + " <bot>: " + assistant_content 
        token.append(combined_content)
        target.append(row['target'])
    return token, target

