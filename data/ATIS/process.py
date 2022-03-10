import pandas as pd
import numpy as np
import pandas as pd
import string
import re


train_file_path = "./train.txt"
val_file_path = "./dev.txt"
test_file_path = "./test.txt"


def processing_data(path, csv_name):

    def processing(content):
        text = []
        slot = []
        intent = []
        vocab = []

        text_tmp = ""
        slot_tmp = []

        for i in content:
            temp = i.strip("\n")

            line = temp.split()
            if len(line) == 1:
                vocab.extend(temp.split("#"))
                intent.append(temp.split("#"))
            elif len(line) > 1:
                text_tmp += f"{line[0]} "
                slot_tmp.append(line[1])

            if len(temp) == 0:
                text.append(text_tmp.strip())
                slot.append(slot_tmp)
                text_tmp = ""
                slot_tmp = []

        if len(text_tmp) >0 or len(slot_tmp) >0:
                text.append(text_tmp.strip())
                slot.append(slot_tmp)

        vocab = sorted(list(set(vocab)))

        return text, slot, intent, vocab

    
    f = open(path, "r")
    content = f.readlines()

    text, slot, intent, vocab = processing(content)

    labels = [",".join(i) for i in intent]

    data = pd.DataFrame({
        "phrase":text,
        "intents":labels
        })

    data.to_csv(csv_name)

    print(f"Processing dataset{csv_name} data {len(data)}")

    return vocab


train_vocab = processing_data(train_file_path, "train.csv")
valid_vocab = processing_data(val_file_path, "val.csv")
test_vocab = processing_data(test_file_path, "test.csv")

train_vocab.extend(valid_vocab)
train_vocab.extend(test_vocab)
vocab = sorted(list(set(train_vocab)))

f = open(f"./vocab.txt", "w")
f.writelines([line + "\n" for line in sorted(vocab)])
f.close()