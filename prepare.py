import torch
import torch.nn
from datasets import load_dataset
from nltk.parse import CoreNLPParser
import nltk
from transformers import AutoTokenizer
from utils.config import get_pos_tag_mapping



def get_clinc_datasets(tokenizer):
    data = load_dataset('clinc_oos', 'small')

    MAX_LENGTH = 64

    # data["train"] = data["train"].map(lambda e: {'pos_tag_ids': nltk_pos_tag(e, MAX_LENGTH)}, batched=True)
    # data["validation"] = data["validation"].map(lambda e: {'pos_tag_ids': nltk_pos_tag(e,MAX_LENGTH)}, batched=True)
    # data["test"] = data["test"].map(lambda e: {'pos_tag_ids': nltk_pos_tag(e,MAX_LENGTH)}, batched=True)

    data["train"] = data["train"].map(lambda e: {'pos_tag_ids': procssing_pos_tag(e['text'], MAX_LENGTH, tokenizer)},
                                      batched=True)
    data["validation"] = data["validation"].map(lambda e: {'pos_tag_ids': procssing_pos_tag(e['text'], MAX_LENGTH, tokenizer)},
                                                batched=True)
    data["test"] = data["test"].map(lambda e: {'pos_tag_ids': procssing_pos_tag(e['text'], MAX_LENGTH, tokenizer)},
                                    batched=True)

    train_data = data["train"].map(
        lambda e: tokenizer(e["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH), batched=True)
    valid_data = data["validation"].map(
        lambda e: tokenizer(e["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH), batched=True)
    test_data = data["test"].map(
        lambda e: tokenizer(e["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH), batched=True)

    train_data.set_format(type="torch", columns=["attention_mask", "input_ids", "intent", "pos_tag_ids"])
    valid_data.set_format(type="torch", columns=["attention_mask", "input_ids", "intent", "pos_tag_ids"])
    test_data.set_format(type="torch", columns=["attention_mask", "input_ids", "intent", "pos_tag_ids"])

    print(f"The min label of datasets: {torch.min(train_data['intent'])}")
    print(f"The max label of datasets: {torch.max(train_data['intent'])}")

    return train_data, valid_data, test_data


def get_num_labels():
    return 151

####################################################################################################
####################################################################################################



####################################################################################################
############################################## Pos Tag #############################################
####################################################################################################

# pos_tagger = CoreNLPParser(url='http://localhost:9000', tagtype='pos')
# print(list(pos_tagger.tag('What is the airspeed of an unladen swallow ?'.split())))

def nltk_pos_tag(examples, max_length):
    # nltk.download('averaged_perceptron_tagger')
    pos_tagging = []
    for text in examples:
        pos_tagged = nltk.pos_tag(text.split())
        pos_ids = [0] + [get_pos_tag_mapping()[word_class] for _, word_class in pos_tagged]
        pos_ids += [2]
        if len(pos_ids) < max_length:
            pos_ids += [1] * (max_length - len(pos_ids))
        else:
            pos_ids = pos_ids[:max_length]
        pos_tagging.append(pos_ids)
    return pos_tagging


def procssing_pos_tag(examples, max_length, tokenizer):
    # nltk.download('averaged_perceptron_tagger')
    pos_tagging = []
    for text in examples:
        ids = tokenizer(text)
        pos_ids = nltk.pos_tag(text.split())
        process_pos_ids = [pos_ids[0]]
        idx = 1
        for k in ids.input_ids:
            if k == 0 or k == 101 or k == 102:
                continue
            # if k == 0 or k == 1 or k == 2:
            #     continue
            token = tokenizer.decode(k)
            if token.startswith(" ") and len(token) > 1:
                process_pos_ids.append(pos_ids[idx])
                idx += 1
            else:
                process_pos_ids.append(process_pos_ids[-1])

        # process_pos_ids = [0] + [get_pos_tag_mapping()[word_class] for _, word_class in process_pos_ids[1:]]
        # process_pos_ids += [2]
        process_pos_ids = [1] + [get_pos_tag_mapping()[word_class] for _, word_class in process_pos_ids[1:]]
        process_pos_ids += [2]

        if len(ids.input_ids) != len(process_pos_ids):
            print(ids.input_ids)
            print(process_pos_ids)

        assert len(ids.input_ids) == len(process_pos_ids)

        if len(process_pos_ids) < max_length:
            # process_pos_ids += [1] * (max_length - len(process_pos_ids))
            process_pos_ids += [0] * (max_length - len(process_pos_ids))

        else:
            process_pos_ids = process_pos_ids[:max_length]

        pos_tagging.append(process_pos_ids)
    return pos_tagging


