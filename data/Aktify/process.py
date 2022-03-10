import pandas as pd
import numpy as np
import pandas as pd
import string
import nltk
import re


file_path = "./home_security_multi_intent.csv"
# fiel_path = "./credit_repair_multi_intent.csv"

raw_data = pd.read_csv(file_path)[["phrase", "intents"]]

def processing_data(raw_data):
    
    special_char = '\@_!-#$%^&*()<>?/.\|}{~:;[],"\"'
    emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    raw_utterance = [i.lower() for i in raw_data["phrase"]]
    raw_utterance1 = []

    for utt in raw_utterance:
        new_utt = ''.join((filter(lambda i: i not in special_char, utt)))
        new_utt = re.sub("[\n]"," ", new_utt)
        new_utt = re.sub(emoji_pattern, "", new_utt)
        new_utt = re.sub("[ ]+", " " , new_utt)
        raw_utterance1.append(new_utt)

    raw_intent = [i for i in raw_data["intents"]]
    intent_list = []
    chars = re.escape(string.punctuation)
    for sample_intent in raw_intent:
        sample_intent1 = ["_".join(re.sub(r'['+chars+']', " ", i).strip().split(" ")[1:]) 
                          for i in sample_intent.split(",")]
        intent_list.append(sample_intent1)

    intents = [",".join(i) for i in intent_list]
    raw_data["intents"] = intents
    raw_data["phrase"] = raw_utterance1
    
    return raw_data

raw_data = processing_data(raw_data)

raw_data = raw_data.sample(frac=1).reset_index(drop=True) # shuffle

test = raw_data.iloc[:200]
train = raw_data.iloc[200:]

test.to_csv("test.csv")
train.to_csv("train.csv")

unique = [x for x in set([j for i in raw_data["intents"] for j in i])]

train_unique = [x for x in set([j for i in train["intents"] for j in i])]

test_unique = [x for x in set([j for i in test["intents"] for j in i])]

print(f"Processing dataset{file_path} training data {len(train)}  testing data {len(test)}")
print(f"total label {len(unique)}, train label {len(train_unique)} test label {len(test_unique)}")

f = open(f"./vocab.txt", "w")
f.writelines([line + "\n" for line in sorted(unique)])
f.close()
