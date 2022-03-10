import pandas as pd
import numpy as np
import pandas as pd
import string
import re


train_file_path = "./train.tsv"
val_file_path = "./eval.tsv"
test_file_path = "./test.tsv"


def processing_data(path, csv_name):

    raw_data = pd.read_table(path, header = None)

    vocab = []
    labels = []
    for i in raw_data[2]:
        temp = re.findall("IN:[A-Z\_]+", i)
        vocab.extend(temp)
        label = [ l[3:].lower() for l in temp]
        labels.append(",".join(label))

    data = pd.DataFrame({
        "phrase":raw_data[0],
        "intents":labels
        })

    data.to_csv(csv_name)

    print(f"Processing dataset{csv_name} data {len(data)}")

    vocab = sorted(list(set(vocab)))
    vocab = [ l[3:].lower() for l in vocab]

    return vocab




train_unique = processing_data(train_file_path, "train.csv")
processing_data(val_file_path, "val.csv")
processing_data(test_file_path, "test.csv")


f = open(f"./vocab.txt", "w")
f.writelines([line + "\n" for line in sorted(train_unique)])
f.close()
