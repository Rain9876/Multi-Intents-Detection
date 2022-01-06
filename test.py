from torch import sigmoid
from transformers import RobertaConfig, AutoTokenizer, RobertaModel, RobertaForSequenceClassification, BertTokenizer
# from roberta import MyRobertaForSequenceClassification, MyRobertaModel, RobertaForSequenceClassification, RobertaModel
from Dataset import multi_intent_dataset
from utils.config import MyRobertaConfig, MyRobertaClassificationConfig, get_pos_tag_mapping
from prepare import get_clinc_datasets
import torch



from utils.building_utils import load_model


# config = MyRobertaClassificationConfig()
# print(config.num_pos_tag)

# model = RobertaModel(config,  add_pooling_layer=False)
# model.from_pretrained("roberta-base")
# print(model)
#
# new_model = MyRobertaModel(config, add_pooling_layer=False)
# new_model.from_pretrained("roberta-base")
#
# del new_model.pooler
#
# print(new_model)

# tokenizer = AutoTokenizer.from_pretrained("roberta-base")
# print(len(tokenizer.get_vocab()))
# print(config)
# print(RobertaConfig())
#


# data, _, _ = get_clinc_datasets(tokenizer)
#
# from torch.utils.data import DataLoader
#
# model = MyRobertaForSequenceClassification(config).cuda(0)
#
# for idx,i in enumerate(DataLoader(data)):
#     print(i["input_ids"].size())
#     print(i["attention_mask"].size())
#     print(i["intent"].size())
#     print(i["pos_tag_ids"])
#     print(model.roberta.embeddings.word_embeddings)
#     output = model(input_ids=i["input_ids"].cuda(0), attention_mask=i["attention_mask"].cuda(0), pos_tag_ids= i["pos_tag"].cuda(0), labels=i["intent"].cuda(0))
    # if idx == 20:
    #     break
    # print(model)

#
# PATH = "/home/song/Desktop/multi_intents/roberta.base/model.pt"
# checkpoint = torch.load(PATH)
# print(checkpoint["model"].keys())
#


#
# tokenizer = AutoTokenizer.from_pretrained('roberta-base')

# config = RobertaConfig.from_pretrained('roberta-base')
# print(config)
# print(type(config))
#
# md = MyRobertaForSequenceClassification.from_pretrained('roberta-base')
# print(md.config)
# print(type(md.config))
#
# config = MyRobertaClassificationConfig()
# print(config)
# print(type(config))


#####################################################################################################

#####################################################################################################
#
#
# torch.manual_seed(42)
# torch.cuda.manual_seed(42)
#
# ### VERIFY THE DIFFERENCE OF TWO LOADED MODELS
# rb = RobertaConfig.from_pretrained('roberta-base')
# model = RobertaForSequenceClassification(rb)
# model = load_model(model, "./pytorch_model.bin")
#
# roberta = RobertaForSequenceClassification.from_pretrained("roberta-base")
#
# model.config.problem_type = "single_label_classification"
# model.num_labels = 151
# num_hidden = model.classifier.out_proj.in_features
# model.classifier.out_proj = torch.nn.Linear(num_hidden, 151)
#
#
# roberta.config.problem_type = "single_label_classification"
# roberta.num_labels = 151
# num_hidden = roberta.classifier.out_proj.in_features
# roberta.classifier.out_proj = torch.nn.Linear(num_hidden, 151)
#
#
# for i, k in zip(model.named_parameters(), roberta.named_parameters()):
#     if not torch.equal(i[1], k[1]) or not i[0] == k[0]:
#         print("No")
#         model.load_state_dict({i[0]: k[1]}, strict=False)
#
# for i, k in zip(model.named_parameters(), roberta.named_parameters()):
#     if not torch.equal(i[1], k[1]) or not i[0] == k[0]:
#         print("Not Same")
#
#
# tokenizer = AutoTokenizer.from_pretrained('roberta-base')
#
# data, _, _ = get_clinc_datasets(tokenizer)
# from torch.utils.data import DataLoader
#
# model.eval()
# roberta.eval()
#
# for idx,i in enumerate(DataLoader(data)):
#
#     output = model(input_ids=i["input_ids"], attention_mask=i["attention_mask"], labels=i["intent"])
#     output1 = roberta(input_ids=i["input_ids"], attention_mask=i["attention_mask"], labels=i["intent"])
#     print(output.loss)
#     print(output1.loss)
#     print(output.logits)
#
#     print(output1.logits)
#     if torch.equal(output.loss, output1.loss):
#         print("OK")
#     if idx == 2:
#         break



#####################################################################################################

#####################################################################################################
# import nltk
# from datasets import load_dataset
#
# tokenizer = AutoTokenizer.from_pretrained('roberta-base')
#
# example = "my visa card  what's the apr on that"
#
# ids = tokenizer(example)
# pos_ids = nltk.pos_tag(example.split())
# process_pos_ids = [pos_ids[0]]
# print(ids)
# print(pos_ids)
# idx = 1
# for k in ids.input_ids:
#     if k == 0 or k == 1 or k == 2:
#         continue
#     token = tokenizer.decode(k)
#     if token.startswith(" ") and len(token) > 1:
#         process_pos_ids.append(pos_ids[idx])
#         idx+=1
#     else:
#         process_pos_ids.append(process_pos_ids[-1])
# print(process_pos_ids)
# process_pos_ids = [0] + [get_pos_tag_mapping()[word_class] for _, word_class in process_pos_ids[1:]]
# process_pos_ids += [2]
# print(process_pos_ids)
#
# assert  len(ids.input_ids) == len(process_pos_ids)

#
# from torch.utils.data import DataLoader
#
# tokenizer = AutoTokenizer.from_pretrained('roberta-base')
# data,_,_ = get_clinc_datasets(tokenizer)
#
# for idx, i in enumerate(DataLoader(data)):
#     print(i["input_ids"])
#     print(i["pos_tag_ids"])
#     if idx == 5:
#         break
#
# #
# import sys
#
# from utils.metrics import cal_score
#
# tokenizer = AutoTokenizer.from_pretrained('roberta-base')
# data = multi_intent_dataset(f"{sys.path[0]}/data/MixATIS_clean", "test", tokenizer)
# print(data[1]["pos_tag_ids"])
# print(data[1]["input_ids"])

import torch
from utils.metrics import cal_score

# 6x7
logits = torch.Tensor([[-0.0220, -0.0074,  0.0335,  0.0274, -0.0090,  0.0449, -0.0071],
        [-0.0901, -0.2241, -0.3263,  0.2891, -0.3841,  0.0106,  0.0317],
        [-0.0899, -0.1971, -0.2502,  0.2330, -0.3311,  0.0287,  0.0765],
        [-0.0652, -0.2073, -0.2047,  0.2083, -0.3741,  0.0812,  0.0978],
        [-0.1646, -0.1641, -0.2572,  0.1834, -0.3647, -0.0514,  0.1140],
        [-0.0483, -0.2115, -0.2946,  0.2871, -0.3048,  0.0276,  0.1015]])


labels = torch.Tensor([[0., 0., 0., 1., 0., 1., 0.],
        [0., 0., 0., 1., 0., 1., 1.],
        [0., 0., 0., 0., 0., 1., 1.],
        [0., 0., 0., 1., 0., 1., 1.],
        [0., 0., 0., 1., 0., 0., 1.],
        [0., 0., 0., 0., 0., 1., 1.]])

# from utils.metrics import accuracy
#
# # ac1, ac2 = accuracy(logits, labels, topk=(1,2))
#
# # print(ac1)
# # print(ac2)
#
#
# pred = torch.round(torch.sigmoid(logits))
# correct = (pred == labels).all(dim=1).sum()
#
# print(correct)
#
# res = cal_score(logits, labels)
# print(res)
#

tokenize = BertTokenizer.from_pretrained('bert-base-uncased')
print(tokenize.decode(103))
print(tokenize.decode(104))
print(tokenize.pad_token_id)
