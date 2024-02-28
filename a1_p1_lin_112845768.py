import re
import collections
import os

#BEGIN(Cheatograph)[https://cheatography.com/davechild/cheat-sheets/regular-expressions/]"regular expression cheat sheet" (I referred this page for rules of regular expressions, as well as to textbook and slides)
def wordTokenizer(text):
    ''' 
        input: text, a single string to be word tokenized.

        output: result, a list of strings of all word tokens, in order, from the string
    '''
    contraction = r"(n't|N'T|'s|'S|'m|'M'|'re|'RE|'ve|'VE|'ll|'LL|'d|'D|')"
    punctuation = r'([^\w\s@$#]+)\1*'
    dot = r"(\d+\.\d+)"
    dash = r'([\d\w]+-[\d\w-]+)'
    whitespace = r'\s+'

    text = text.replace(r':)', " EMO1 ")
    text = text.replace(r':-)', " EMO2 ")
    text = text.replace(r':(', " EMO3 ")
    text = text.replace(r':-(', " EMO4 ")

    pattern = f"{contraction}|{dot}|{whitespace}|{dash}|{punctuation}"

    tokens = re.split(pattern, text)
    tokens = [token for token in tokens if token and not token.isspace()]
    result = []
    for token in tokens:
        modified_token = re.sub(r'(?<=([^\w\s]))(?!\1)([^\w\s])', r' \2', token)
        modified_token = modified_token.replace('EMO1', ":)")
        modified_token = modified_token.replace('EMO2', ":-)")
        modified_token = modified_token.replace('EMO3', ":(")
        modified_token = modified_token.replace('EMO4', ":-(")
        modified_token = re.split(r'\s+', modified_token)
        result.extend(modified_token)
    
    return result

#BEGIN(GitHub)(https://ethen8181.github.io/machine-learning/deep_learning/subword/bpe.html#Implementation-From-Scratch) " BPE Implementation From Scratch" (was inpired by couple of webistes, but most come from this webiste)
def spacelessBPELearn(docs, max_vocabulary=1000):
    ''' 
        input: docs, a list of strings to be used as the corpus for learning the BPE vocabulary

        output: final_vocabulary, a set of all members of the learned vocabulary
    '''    
    text = ''.join(c if 31< ord(c) < 127 else '?' for doc in docs for c in doc)
    vocabulary = text.strip().split()
    word_freq_dict = collections.defaultdict(int)
    V = list(set(text))
    V = [word for word in V if word.strip()]
    n = len(V)
    for word in vocabulary:
        word_freq_dict[' '.join(word)] += 1
    
    while len(V) < max_vocabulary:
        pairs = collections.defaultdict(int)
        for word, freq in word_freq_dict.items():
            chars = word.split()
            for i in range(len(chars)-1):
                pairs[chars[i], chars[i+1]] += freq
        
        if not pairs:
            break
#         if len(V) in {n, n + 1, n + 10, n + 100, n + 500}:
#             top_5 = sorted(pairs.items(), key=lambda x: x[1], reverse=True)[:5]
#             print(f"At iteration {len(V)-n} top five most frequent pairs are: {top_5}")
        
        best_pair = max(pairs, key=pairs.get)
        combined_pair = ''.join(best_pair)
        V.append(combined_pair)
        
        merged_dict = {}
        bigram = re.escape(' '.join(best_pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in word_freq_dict:
            w_out = p.sub(''.join(best_pair), word)
            merged_dict[w_out] = word_freq_dict[word]
        
        word_freq_dict = merged_dict
    
    return V
#END(GitHub)


#BEGIN[ChatGPT][https://chat.openai.com/auth/login] "Byte pairing tokenization python code example"
def spacelessBPETokenize(text, vocab):
    words = []  
    chars = list(text) 
    i = 0    
    while i < len(chars):
        if chars[i].isspace():
            i += 1
            continue
        for j in range(len(chars), i, -1):
            if ''.join(chars[i:j]) in vocab:
                words.append(''.join(chars[i:j]))
                i = j
                break
        else:
            words.append(chars[i])
            i += 1
    
    return words
#END[ChatGPT]


def main(text):
    with open(text, 'r',encoding="utf-8") as file:
        lines = file.readlines()

    first_5_doc = lines[:5]
    last_doc = lines[-1]
    content = ' '.join(first_5_doc + [last_doc])
    print('Checkpoint 1.1:\n')
    for i in {0,1,2,3,4,-1}:
        tokens = wordTokenizer(lines[i])
        print(tokens)
    print('\nCheckpoint 1.2:\n')

    vocab = spacelessBPELearn(''.join(lines))
    print("Final Vocabulary:", vocab)
    

    tokens = spacelessBPETokenize(lines[0], vocab)
    print("\nDoc 1:\n", tokens)
    tokens = spacelessBPETokenize(lines[1], vocab)
    print("\nDoc2:\n", tokens)
    tokens = spacelessBPETokenize(lines[2], vocab)
    print("\nDoc3:\n", tokens)
    tokens = spacelessBPETokenize(lines[3], vocab)
    print("\nDoc4:\n", tokens)
    tokens = spacelessBPETokenize(lines[4], vocab)
    print("\nDoc5:\n", tokens)
    tokens = spacelessBPETokenize(lines[-1], vocab)
    print("\nLast Doc:\n", tokens)
    
if __name__ == "__main__":
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    input_file = "a1_tweets.txt"
    main(input_file)
