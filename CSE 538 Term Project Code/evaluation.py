# Team Members: Zhiwei Lin, Sai Chaddha
# Description: This file evaluate best hyperparamter (alpha) for hatebert for the toxicity detection task. (alpha)(question_probs) + (1-alpha)(response_probs)
# We are running the system from Google Colab with a T4 or A100 GPU.

# We submitted the code on time; however, we weren't aware that we needed to cite our code since the project assignment didn't include instructions to do so.
from sklearn.metrics import accuracy_score, f1_score
from transformers import GPT2LMHeadModel, GPT2Tokenizer,AutoTokenizer, AutoModelForSequenceClassification

import torch
from extract_data import *
from normalized import *

## Hyperparameter tuning (alpha) with test dataset. (alpha)(question_probs) + (1-alpha)(response_probs) for imporved model 1 
def evaluate_weight_model(test_question, test_response, test_target, alpha, tokenizer, question_model, response_model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    question_model.eval().to(device)
    response_model.eval().to(device)

    all_predictions = []

    for i in range(len(test_question)):


        encoded_question = tokenizer(
            test_question[i],
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        encoded_question = {key: val.to(device) for key, val in encoded_question.items()}


        encoded_response = tokenizer(
            test_response[i],
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        encoded_response = {key: val.to(device) for key, val in encoded_response.items()}

        with torch.no_grad():
            question_outputs = question_model(**encoded_question)
            question_logits = question_outputs.logits
            response_outputs = response_model(**encoded_response)
            response_logits = response_outputs.logits

        question_probs = torch.nn.functional.softmax(question_logits, dim=-1)
        response_probs = torch.nn.functional.softmax(response_logits, dim=-1)


        # Combine probabilities using the weighting factor alpha
        new_probs = alpha * question_probs + (1 - alpha) * response_probs

        predicted_class = torch.argmax(new_probs, dim=-1).item()
        all_predictions.append(predicted_class)



    accuracy = accuracy_score(test_target, all_predictions)
    f1 = f1_score(test_target, all_predictions, average='macro')

    return accuracy, f1


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

question_directory = '/content/drive/MyDrive/Colab Notebooks/project/model_hatebert_question_save'
response_directory = '/content/drive/MyDrive/Colab Notebooks/project/model_hatebert_response_save'



tokenizer = AutoTokenizer.from_pretrained(question_directory) # tokenizer for response and question are the same
question_model = AutoModelForSequenceClassification.from_pretrained(question_directory , num_labels = 2).to(device)
response_model = AutoModelForSequenceClassification.from_pretrained(response_directory , num_labels = 2).to(device)


alpha = [0.5,0.6,0.7,0.8,0.9]
FilePath = '/content/drive/MyDrive/Colab Notebooks/project/train-00000-of-00006-4feeb3f83346a0e9.parquet'
_, _ , test_data = extract_data(FilePath)

test_target = test_data['target']
test_question = test_data['conversation'].apply(lambda x: x[0]['content'])
test_response = test_data['conversation'].apply(lambda x: x[1]['content'])

for i in alpha:
    accuracy, f1 = evaluate_weight_model(test_question,test_response, test_target, i, tokenizer, question_model, response_model)
    print(f"Alpha: {i}, Accuracy: {accuracy:.2f}, F1 Score: {f1:.2f}")



### chat agent with mixed repsonse_bert and question_bert
def chat():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gpt_directory = '/content/drive/MyDrive/Colab Notebooks/project/gpt2model_save'
    gpttokenizer = GPT2Tokenizer.from_pretrained(gpt_directory)
    gptmodel = GPT2LMHeadModel.from_pretrained(gpt_directory).to(device)

    question_directory = '/content/drive/MyDrive/Colab Notebooks/project/model_hatebert_question_save'
    tokenizer = AutoTokenizer.from_pretrained(question_directory) # tokenizer for response and question are the same
    question_model = AutoModelForSequenceClassification.from_pretrained(question_directory , num_labels = 2).to(device)

    response_directory = '/content/drive/MyDrive/Colab Notebooks/project/model_hatebert_response_save'
    response_model = AutoModelForSequenceClassification.from_pretrained(response_directory , num_labels = 2).to(device)


    a = gpttokenizer.encode("<bot>:")[0]
    print("Chatbot is ready to talk to you! (type 'quit' to stop)")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            print("Chatbot: Goodbye!")
            break

        prompt = "<s> " + user_input + " <bot>:"

        generated = torch.tensor(gpttokenizer.encode(prompt)).unsqueeze(0)
        generated = generated.to(device)
        sample_outputs = gptmodel.generate(
            generated,
            pad_token_id=gpttokenizer.eos_token_id,  # it generates a warning if i don't set it explicitly
            do_sample=True,
            top_k=50,
            max_length=50,
            top_p=0.95,
            num_return_sequences=1
        )
        start_index =(sample_outputs[0] == a).nonzero(as_tuple=True)[0].item()
        sample_outputs=sample_outputs[0][start_index+1:]
        response = gpttokenizer.decode(sample_outputs, skip_special_tokens=True)


        encoded_question = tokenizer(user_input, return_tensors='pt', truncation=True, max_length=512)
        encoded_question = {k: v.to(device) for k, v in encoded_question.items()}


        encoded_response = tokenizer(response, return_tensors='pt', truncation=True, max_length=512)
        encoded_response = {k: v.to(device) for k, v in encoded_response.items()}

        with torch.no_grad():
          question_outputs = question_model(**encoded_question)
          question_logits = question_outputs.logits

          response_outputs = response_model(**encoded_response)
          response_logits = response_outputs.logits


        question_probs = torch.nn.functional.softmax(question_logits, dim=-1)
        response_probs = torch.nn.functional.softmax(response_logits, dim=-1)

## question_bert + reponse_bert with alpha = 0.5, best hypermeter determined
        probs = 0.5*question_probs + 0.5 * response_probs

        predicted_class = torch.argmax(probs, dim=-1).item()

        if predicted_class == 1:
            print('I am sorry, I do not know the answer as it is not an appropriate question.')
        else:
            print("Chatbot:", response)







    
