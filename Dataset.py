import ast
import json
import os
from torch.utils.data import Dataset
import torch
# from prepare import procssing_pos_tag
import pandas as pd

class multi_intent_dataset(Dataset):

    def __init__(self, path, data_type, tokenizer):
        self.path = path
        self.data_type = data_type
        self.tokenizer = tokenizer
        self.index2instance = None
        self.instance2index = None
        self.num_labels = None

        MAX_LENGTH = 64

        f = open(f"{self.path}/{self.data_type}.txt", "r")
        content = f.readlines()
        f.close()

        text, slot, intent = processing(content)

        # self.build_instance_map(intent)
        self.get_instance_map()

        # self.pos_tag_ids = torch.LongTensor(procssing_pos_tag(text, MAX_LENGTH, tokenizer))
        self.one_hot = self.one_hot_mapping(intent)
        self.input_data = self.encode_mi_file(text, MAX_LENGTH)

        print(f"{self.input_data.input_ids.size(0)} {self.data_type} data has been loaded!")

    def build_instance_map(self, labels):
        if not os.path.exists(f"{self.path}/vocab.txt") and self.data_type == "train":
            self.index2instance = list(set([j for i in labels for j in i]))
            print(self.index2instance)
            self.instance2index = {j: i for i, j in enumerate(self.index2instance)}
            f = open(f"{self.path}/vocab.txt", "w")
            f.writelines([line + "\n" for line in self.index2instance])
            f.close()

    def get_instance_map(self):
        try:
            f = open(f"{self.path}/vocab.txt", "r")
            self.index2instance = f.read().splitlines()
            self.num_labels = len(self.index2instance)
            self.instance2index = {j: i for i, j in enumerate(self.index2instance)}
            f.close()
        except FileNotFoundError:
            print("Vocab file not exist!")

    def __len__(self):
        return self.input_data.input_ids.size(0)

    def __getitem__(self, item):

        return {"input_ids": self.input_data["input_ids"][item],
         "attention_mask": self.input_data["attention_mask"][item],
         "intent": self.one_hot[item],
         # "pos_tag_ids": self.pos_tag_ids[item]
         }

    def encode_mi_file(self, text, MAX_LENGTH):
        tokenized_data = self.tokenizer(
            text,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        # print(tokenized_data.input_ids.size())

        assert tokenized_data.input_ids.size(1) == MAX_LENGTH
        return tokenized_data

    def essemble_data(self):
        assert self.input_data.input_ids.size(0) == self.one_hot.size(0)
        self.pos_tag_ids = torch.tensor(self.pos_tag_ids)

        input = [{"input_ids": self.input_data["input_ids"][i],
                  "attention_mask": self.input_data["attention_mask"][i],
                  "intent": self.one_hot[i],
                  "pos_tag_ids":self.pos_tag_ids[i]
                 } for i in range(self.one_hot.size(0))]

        return input


    def one_hot_mapping(self, labels):
        one_hot = []
        for label in labels:
            tmp = [0.] * len(self.instance2index)
            for i in label:
                tmp[self.instance2index[i]] = 1
            one_hot.append(tmp)

        return torch.FloatTensor(one_hot)

    def get_num_labels(self):
        return self.num_labels

def get_labels_vocab(config, path):
    try:
        f = open(f"{path}/vocab.txt", "r")
        index2instance = f.read().splitlines()
        config.num_classes = len(index2instance)
        f.close()
        return index2instance
    except FileNotFoundError:
        print("Vocab file not exist!")

def processing(content):
    text = []
    slot = []
    intent = []

    text_tmp = ""
    slot_tmp = []

    for i in content:
        temp = i.strip("\n")
        if len(temp) == 0:
            text.append(text_tmp.strip())
            slot.append(slot_tmp)
            text_tmp = ""
            slot_tmp = []

        line = temp.split()
        if len(line) == 1:
            intent.append(temp.split("#"))
        elif len(line) > 1:
            text_tmp += f"{line[0]} "
            slot_tmp.append(line[1])

    return text, slot, intent


def get_SNIPS_num_labels():
    return 7

def get_ATIS_num_labels():
    return 18



import os
from torch.utils.data import Dataset
import torch
# from prepare import procssing_pos_tag


class aktify_multi_intent_dataset(Dataset):

    def __init__(self, path, data_type, tokenizer):
        self.path = path
        self.data_type = data_type
        self.tokenizer = tokenizer
        self.index2instance = None
        self.instance2index = None
        self.num_labels = None

        MAX_LENGTH = 128

        raw_data = pd.read_csv(f"{self.path}/{self.data_type}.csv", lineterminator="\n")

        text = raw_data.phrase.tolist()

        intent = [[n.strip() for n in ast.literal_eval(i)] for i in raw_data.intents]
        # self.build_instance_map(intent)
        self.get_instance_map()

        # self.pos_tag_ids = torch.LongTensor(procssing_pos_tag(text, MAX_LENGTH, tokenizer))
        self.one_hot = self.one_hot_mapping(intent)
        self.input_data = self.encode_mi_file(text, MAX_LENGTH)

        print(f"{self.input_data.input_ids.size(0)} {self.data_type} data has been loaded!")

    # def build_instance_map(self, labels):
    #     if not os.path.exists(f"{self.path}/vocab.txt") and self.data_type == "train":
    #         self.index2instance = list(set([j for i in labels for j in i]))
    #         print(self.index2instance)
    #         self.instance2index = {j: i for i, j in enumerate(self.index2instance)}
    #         f = open(f"{self.path}/vocab.txt", "w")
    #         f.writelines([line + "\n" for line in self.index2instance])
    #         f.close()

    def get_instance_map(self):
        try:
            f = open(f"{self.path}/vocab.txt", "r")
            self.index2instance = f.read().splitlines()
            self.num_labels = len(self.index2instance)
            self.instance2index = {j: i for i, j in enumerate(self.index2instance)}
            f.close()
        except FileNotFoundError:
            print("Vocab file not exist!")

    def __len__(self):
        return self.input_data.input_ids.size(0)

    def __getitem__(self, item):

        return {"input_ids": self.input_data["input_ids"][item],
         "attention_mask": self.input_data["attention_mask"][item],
         "intent": self.one_hot[item],
         }

    def encode_mi_file(self, text, MAX_LENGTH):
        tokenized_data = self.tokenizer(
            text,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        # print(tokenized_data.input_ids.size())

        assert tokenized_data.input_ids.size(1) == MAX_LENGTH
        return tokenized_data

    def one_hot_mapping(self, labels):
        one_hot = []
        for label in labels:
            tmp = [0.] * len(self.instance2index)
            for i in label:
                tmp[self.instance2index[i]] = 1
            one_hot.append(tmp)

        return torch.FloatTensor(one_hot)

    def get_num_labels(self):
        return self.num_labels

def get_labels_vocab(config, path):
    try:
        f = open(f"{path}/vocab.txt", "r")
        index2instance = f.read().splitlines()
        config.num_classes = len(index2instance)
        f.close()
        return index2instance
    except FileNotFoundError:
        print("Vocab file not exist!")

def get_Aktify_labels():
    return 67
