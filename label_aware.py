import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
from transformers import BertTokenizer, BertModel, AutoModel, AutoTokenizer
from utils.module import MultiAttentionBlock
from utils.building_utils import InfoNCE
from contrastive_learning import PrototypicalContrastiveLearning, Sup_ContrastiveLearning, ContrastiveLearning
import os

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class MulCon(nn.Module):
    def __init__(self, config, labels):
        super(MulCon, self).__init__()
        self.config = config
        self.labels = labels

        self.utter_encoder = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True, output_attentions=True)
        # self.utter_encoder = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-stsb-mean-tokens', output_hidden_states=True, output_attentions=True)

        self.label_emb = None
        self.bertlabelencoder = None

        if self.config.mode == "self-attentive":
            print(self.config.mode)
            self.linear1 = nn.Linear(config.hidden, 256)
            self.linear2 = nn.Linear(4 * 256, config.hidden)
            self.tanh = nn.Tanh()
            self.context_vector = nn.Parameter(torch.randn(256, 4), requires_grad=True)

        if config.label_pretrained:
            if os.path.exists(config.emb_path) and config.load_label_emb:
                self.emb_weight = self.getLabelEmbeddings(config.emb_path)
            else:
                # Whether utter_encoder can be used since requires_grad = False
                # self.bertlabelencoder = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True, output_attentions=True)
                tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

                # self.bertlabelencoder = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-stsb-mean-tokens', output_hidden_states=True, output_attentions=True)
                # tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-stsb-mean-tokens')

                label_inputs = tokenizer.batch_encode_plus(labels, truncation=True, padding=True, return_tensors="pt")

                self.emb_weight = self.LabelRepresentation(self.utter_encoder, label_inputs, label_inputs.input_ids.device)

            self.label_emb = nn.Embedding.from_pretrained(self.emb_weight)
        else:
            self.label_emb = nn.Embedding(config.num_classes, config.hidden)

        self.MAB = MultiAttentionBlock(config)

        self.dropout = nn.Dropout(0.1)

        self.proj = nn.Linear(config.hidden, 1)

        self.pcl = PrototypicalContrastiveLearning(config)

        # self.cl = Sup_ContrastiveLearning(config)

        self.init_weights()

    def init_weights(self):
        """Initialize the weights"""
        if not self.config.label_pretrained:
            self.label_emb.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if self.label_emb.padding_idx is not None:
                self.label_emb.weight.data[self.label_emb.padding_idx].zero_()

        nn.init.xavier_normal_(self.proj.weight)


    def forward(self, x_utter, x_mask, y_labels):

        label_level_emb = self.label_level_network(x_utter, x_mask)

        label_logits = self.dropout(label_level_emb)

        label_logits = self.proj(label_logits).squeeze(2)  # (B x L)

        single_intent_utter_emb, label_indexs = self.get_all_utter_level_embedding(label_level_emb, y_labels)  # (N, H), (N, )

        all_label_key_emb = self.get_all_label_keys_embedding(label_indexs[1])  # (N x C-1 x H)

        pcl_loss = self.pcl(single_intent_utter_emb, all_label_key_emb)

        #cl_loss = self.cl(single_intent_utter_emb, label_indexs[1])
        #pcl_loss = 0

        cl_loss = 0

        # if self.train():
            # label_level_emb = self.label_level_network(x_utter, x_mask)
            # single_intent_utter_emb_2, _ = self.get_all_utter_level_embedding(label_level_emb, y_labels)  # (N, H), (N, )
            # cl_loss = self.cl(single_intent_utter_emb, single_intent_utter_emb_2, label_indexs[1])

        return label_logits, pcl_loss, cl_loss


    def label_level_network(self, x_utter, x_mask):

        last_hidden_states, pooled_output, hidden_states, attentions = self.utter_encoder(x_utter,
                                                                                          attention_mask=x_mask, return_dict=False)  # last hidden states (B, L, H)
        batch_label_embed = self.label_emb.weight.repeat(last_hidden_states.size(0), 1, 1).to(x_utter.device)

        # batch_label_embed = self.emb_weight.repeat(last_hidden_states.size(0),1,1)

        label_level_emb, _ = self.MAB(batch_label_embed, last_hidden_states, last_hidden_states, slf_attn_mask=x_mask)  # (B x L x H)

        return label_level_emb


    def LabelRepresentation(self, encoder, label_inputs, device):

        output = encoder(label_inputs.input_ids.to(device), attention_mask= label_inputs.attention_mask.to(device), return_dict=False)

        label_repre = self.label_pooling(output, attention_mask= label_inputs.attention_mask.to(device))

        # batch_size = 16
        # temp = []
        # for i in range(0, len(label_collections), batch_size):
        # 	output = self.bertlabelencoder(input_ids=label_inputs.input_ids[i:i+batch_size], attention_mask=label_inputs.attention_mask[i:i+batch_size])
        # 	pooling_states = self.label_pooling(output)
        # 	temp.append(pooling_states)
        # label_repre = torch.stack(temp)

        return label_repre

    def label_pooling(self, output, attention_mask):

        last_hidden, pooled_states, hidden_states, _ = output

        if self.config.mode == "cls":
            prototypes = last_hidden[:, 0, :]  # index of CLS

        elif self.config.mode == "mean-pooling":
            # Mean of layers
            # prototypes = torch.mean(torch.sum(torch.stack(hidden_states[-4:-3]), dim=0), dim=1)
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            prototypes = torch.sum(last_hidden *input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        elif self.config.mode == "self-attentive":
            b, _, _ = last_hidden.size()
            vectors = self.context_vector.unsqueeze(0).repeat(b, 1, 1)
            h = self.linear1(last_hidden) # (b, t, h)
            scores = torch.bmm(h, vectors) # (b, t, 4)
            scores = nn.Softmax(dim=1)(scores) # (b, t, 4)
            outputs = torch.bmm(scores.permute(0, 2, 1), h).view(b, -1) # (b, 4h)
            prototypes = self.linear2(outputs)

        else:  # pooling layer
            prototypes = pooled_states
        return prototypes


    def get_all_utter_level_embedding(self, label_level_emb, one_hot_emb):
        """
        :param label_level_emb: (B, L, H)
        :param one_hot_emb: (B, T)
        :return: extract_utter_emb (N, H)
        :return: label_tuple ((N, ) (N, ))

        """
        label_tuple = one_hot_emb.nonzero(as_tuple=True)  # (B, ) (N, )
        extract_utter_emb = label_level_emb[label_tuple[0], label_tuple[1], :]

        assert extract_utter_emb.size(0) == label_tuple[1].size(0)

        return extract_utter_emb, label_tuple

    def get_all_label_keys_embedding(self, labels_idx):
        """
        :param labels_idx: (N, )
        :return all_label_key_emb: label keys embedding (the first vector is a positive sample by default, and the rest are negative samples)
        """
        label_embs = self.label_emb.weight
        # label_embs = self.emb_weight  # (L x H)
        batch_size = labels_idx.size(0)
        neg_labels = torch.arange(self.config.num_classes).repeat(batch_size, 1).to(labels_idx.device)
        mask = torch.ones_like(neg_labels).scatter_(1, labels_idx.unsqueeze(1), 0.)
        neg_labels = neg_labels[mask.bool()].contiguous().view(batch_size, self.config.num_classes - 1)  # (N x C-1)
        label_keys = torch.cat([labels_idx.unsqueeze(1), neg_labels],dim=1)
        # neg_label_embs = label_emb.repeat(labels_idx.size(0),1,1) # (N x L x H)
        # all_neg_label_emb = neg_label_embs[labels_idx]    # (N x L-1 * H)
        all_label_key_emb = label_embs[label_keys, :]  # (N x C-1 x H)

        return all_label_key_emb


## Save and get label Embeddings from files
def saveLabelEmbedding(emb, file_path):
    emb = emb.detach()
    torch.save(emb, file_path)


def getLabelEmbedding(file_path):
    label_emb = None
    if not os.path.exists(file_path):
        label_emb = torch.load(file_path, map_location={'cuda:1': 'cuda:0'})
    return label_emb
