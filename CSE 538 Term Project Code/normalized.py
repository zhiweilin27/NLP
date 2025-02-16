# Team Members: Zhiwei Lin, Sai Chaddha
# Description: This file contains the function that normalizes the string by converting it to ascii and removing any special characters.
# We are running the system from Google Colab with a T4 or A100 GPU.

import unicodedata
import re

# BEGIN[GitHub][https://gist.github.com/lewiis/c76fbbd6bd35101edefee3fad3ebe588]'Turn a Unicode string to plain ASCII'
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
# END[GitHub]
# Regular Expressions [I. Syntax]
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z0-9.!?<>:'/_%\[\],()\"-]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    s = re.sub(r'\s+([,.!?])', r'\1', s)
    s = re.sub(r"i'm", "i am", s)
    s = re.sub(r"\r", "", s)
    s = re.sub(r"he's", "he is", s)
    s = re.sub(r"she's", "she is", s)
    s = re.sub(r"it's", "it is", s)
    s = re.sub(r"that's", "that is", s)
    s = re.sub(r"what's", "that is", s)
    s = re.sub(r"where's", "where is", s)
    s = re.sub(r"here's", "here is", s)
    s = re.sub(r"how's", "how is", s)
    s = re.sub(r"\'ll", " will", s)
    s = re.sub(r"\'ve", " have", s)
    s = re.sub(r"\'re", " are", s)
    s = re.sub(r"\'d", " would", s)
    s = re.sub(r"\'re", " are", s)
    s = re.sub(r"won't", "will not", s)
    s = re.sub(r"can't", "cannot", s)
    s = re.sub(r"n't", " not", s)
    # s =  "<s> " + s +" </s>"
    return s
