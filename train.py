from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.model_config import Config
from label_aware import MulCon
from Dataset import get_labels_vocab
from Dataset import multi_intent_dataset, aktify_multi_intent_dataset
from transformers import BertTokenizer
import torch
from utils.metrics import calc_score,f1_score_intents

# path = "/Users/yurunsong/Desktop/Multi-Intent/Multi-Intents-Detection/data/MixSNIPS_clean"
path = "/home/song/Desktop/Multi-Intents-Detection/data/Aktify"

config = Config()

labels = get_labels_vocab(config, path)
print(labels)

model = MulCon(config, labels)

lr = 1e-5
opt = torch.optim.AdamW(model.parameters(), lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-2)
epochs = 50

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_dataset = aktify_multi_intent_dataset(path, "train", tokenizer)
test_dataset = aktify_multi_intent_dataset(path, "test", tokenizer)

train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=False)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=False)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = model.to(device)

i = 0
while i < epochs:
	model.train()
	train_loss = []
	test_loss = []

	for i, data in enumerate(train_data_loader):
		opt.zero_grad()
		x_utter = data["input_ids"].to(device)
		x_mask = data["attention_mask"].to(device)
		y_labels = data["intent"].to(device)
		output = model(x_utter,x_mask,y_labels)
		# print(loss1)
		# print(loss2)
		# loss = loss1 * 0.5 + loss2 * 0.5
		output[0].backward()
		opt.step()
		train_loss.append(output[0].item())

	model.eval()

	with torch.no_grad():
		for i, batch in enumerate(test_data_loader):
			input_ids = batch["input_ids"].to(device)
			attention_mask = batch["attention_mask"].to(device)
			labels = batch["intent"].to(device)
			output = model(input_ids, attention_mask, labels)
			logits = output[1]
			loss = output[0]
			_, _, _, acc = calc_score(logits, labels)
			P, R, f1, _ = f1_score_intents(logits, labels)
			acc /= input_ids.size(0)
			test_loss.append(output[0].item())

	print(f"loss {loss}")
	print(f"acc: {acc}")

	i+= 1
