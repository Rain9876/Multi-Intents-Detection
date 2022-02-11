from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.model_config import Config
from label_aware import MulCon
from Dataset import get_labels_vocab
from Dataset import multi_intent_dataset
from transformers import BertTokenizer
import torch

path = "/Users/yurunsong/Desktop/Multi-Intent/Multi-Intents-Detection/data/MixSNIPS_clean"

config = Config()

labels  = get_labels_vocab(config, path)
print(labels)

model = MulCon(config, labels)
lr = 1e-5
opt = Adam(model.parameters(), lr = lr)
epochs = 3

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_dataset = multi_intent_dataset(path, "train", tokenizer)

train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=False)

i = 0
while i < epochs:
	model.train()
	train_loss = []
	for data in tqdm(train_data_loader):
		opt.zero_grad()
		x_utter = data["input_ids"]
		x_mask = data["attention_mask"]
		y_labels = data["intent"]
		output = model(x_utter,x_mask,y_labels)
		# print(loss1)
		# print(loss2)
		# loss = loss1 * 0.5 + loss2 * 0.5
		output[0].backward()
		opt.step()
		train_loss.append(loss1.item())
	i+= 1
