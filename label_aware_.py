import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torch.nn import BCEWithLogitsLoss
from transformers import BertTokenizer, BertModel, AutoModel, AlbertModel
from utils.module import MultiAttentionBlock
import pickle as pk
import os

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class MulCon(nn.Module):
    def __init__(self, config, labels):
        super(MulCon, self).__init__()
        self.config = config
        self.labels = labels
        self.utter_encoder = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True,
                                                       output_attentions=True)
        self.label_emb = None
        self.bertlabelencoder = None
        self.emb_weight =None
        
        self.bertlabelencoder = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True,
                                                                  output_attentions=True)
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        label_inputs = tokenizer.batch_encode_plus(label_lists, truncation=True, padding=True, return_tensors="pt")

        self.emb_weight = self.LabelRepresentation(label_inputs, device)

        self.label_emb = nn.Embedding.from_pretrained(emb_weight)

        self.MAB = MultiAttentionBlock(config)

        self.proj = nn.Linear(config.hidden, 1)

        self.pcl = PrototypicalContrastiveLearning(config)

        self.loss_fct = BCEWithLogitsLoss()

        self.init_weights()

    def init_weights(self):
        """Initialize the weights"""
        if not self.config.label_pretrained:
            self.label_emb.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if self.label_emb.padding_idx is not None:
                self.label_emb.weight.data[self.label_emb.padding_idx].zero_()

        self.proj.weight.data.normal_(mean=0.0, std=self.config.initializer_range)


    def forward(self, x_utter, x_mask, y_labels):

        last_hidden_states, pooled_output, hidden_states, attentions = self.utter_encoder(x_utter,
                                                                                          attention_mask=x_mask, return_dict=False)  # last hidden states (B, L, H)
        batch_label_embed = self.label_emb.weight.repeat(last_hidden_states.size(0),1,1)

        label_level_emb, _ = self.MAB(batch_label_embed, last_hidden_states, last_hidden_states)  # (B x L x H)

        label_logits = self.proj(label_level_emb).squeeze(2)  # (B x L)

        bce_loss = self.loss_fct(label_logits, y_labels)

        single_intent_utter_emb, label_indexs = self.get_all_utter_level_embedding(label_level_emb, y_labels)  # (N, H), (N, )

        all_label_key_emb = self.get_all_label_keys_embedding(label_indexs[1])  # (N x C-1 x H)

        pcl_loss = self.pcl(single_intent_utter_emb, all_label_key_emb)

        return bce_loss, label_logits, pcl_loss

    def LabelRepresentation(self, device):
        
        # Fuse labels into Model once together if labels are limited.
        output = self.bertlabelencoder(self.label_inputs.input_ids.to(device), attention_mask= self.label_inputs.attention_mask.to(device), return_dict=False)
        label_repre = self.label_pooling(output)
        return label_repre

    def label_pooling(self, output):
        # Convet hidden states to sentence representation or just use pooled states
        # Todo
        last_hidden, pooled_states, _, _ = output

        prototypes = None

        if self.config.mode == "cls":
            prototypes = last_hidden[:, 0, :]  # index of CLS
        elif self.config.mode == "mean":
            # Mean of Last layer
            prototypes = torch.mean(last_hidden, dim=1)
        elif self.config.mode == "self-attentive":
            pass
        elif self.config.mode == "self-attention":
            pass
        elif self.config.mode == "SAB":
            pass
        else:  # pooling layer
            prototypes = pooled_states
        return prototypes

    def get_all_utter_level_embedding(self, label_level_emb, one_hot_emb):
        """
            label_level_emb (B, L, H)
            one_hot_emb (B, T)
            extract_utter_emb(N, H)
        """
        label_tuple = one_hot_emb.nonzero(as_tuple=True)  # (B, ) (N, )
        extract_utter_emb = label_level_emb[label_tuple[0], label_tuple[1], :]

        assert extract_utter_emb.size(0) == label_tuple[1].size(0)

        return extract_utter_emb, label_tuple

    def get_all_label_keys_embedding(self, labels_idx):
        """
            labels_idx (N, )
        """
        label_emb = self.label_emb.weight  # (L x H)
        batch_size = labels_idx.size(0)
        neg_labels = torch.arange(self.config.num_classes).repeat(batch_size, 1).to(labels_idx.device)
        mask = torch.ones_like(neg_labels).scatter_(1, labels_idx.unsqueeze(1), 0.)
        neg_labels = neg_labels[mask.bool()].view(batch_size, self.config.num_classes - 1)  # (N x C-1)
        label_keys = torch.cat([labels_idx.unsqueeze(1), neg_labels],dim=1)
        all_label_key_emb = label_emb[label_keys, :]  # (N x C-1 x H)

        return all_label_key_emb


## Save and get label Embeddings from files
def saveLabelEmbedding(emb, file_path):
    emb = emb.detach()
    torch.save(emb, file_path)


def getLabelEmbedding(file_path):
    label_emb = None
    if not os._exists(file_path):
        label_emb = torch.load(file_path, map_location={'cuda:1': 'cuda:0'})
    return label_emb

class PrototypicalContrastiveLearning(nn.Module):

    def __init__(self, config) -> None:
        super(PrototypicalContrastiveLearning, self).__init__()

        self.all_intent_projection = AdapterLayer(config)
        self.pre_intent_projection = AdapterLayer(config)

    def init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            nn.init.eye_(module.weight)
            # module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, label_level_query, label_level_keys):
        label_level_query = self.pre_intent_projection(label_level_query)
        label_level_keys = self.all_intent_projection(label_level_keys)
        pcl_loss = contrastive_loss(label_level_query, label_level_keys)
        return pcl_loss


class AdapterLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.adapter_input = config.hidden
        self.adapter_latent = config.adapter_latent
        self.non_linearity = torch.tanh

        # down projection
        self.down_proj = nn.Linear(self.adapter_input, self.adapter_latent)
        # up projection
        self.up_proj = nn.Linear(self.adapter_latent, self.adapter_input)

        self.init_weights()

    def init_weights(self):
        """ Initialize the weights -> so that initially we the whole Adapter layer is a near-identity function """
        self.down_proj.weight.data.normal_(mean=0.0, std=0.02)
        self.down_proj.bias.data.zero_()
        self.up_proj.weight.data.normal_(mean=0.0, std=0.02)
        self.up_proj.bias.data.zero_()

    def forward(self, x):
        output = self.up_proj(self.non_linearity(self.down_proj(x)))
        output = x + output
        return output


def contrastive_loss(query: torch.Tensor, keys: torch.Tensor, temperature: float = 0.1):
    """
    Given the query vector and the keys matrix (the first vector is a positive sample by default, and the rest are negative samples), calculate the contrast learning loss, follow SimCLR
    :param query: shape=(N,d,)
    :param keys: shape=(N,C, d)
    :return: scalar
    """
    query = torch.nn.functional.normalize(query, dim=1)
    keys = torch.nn.functional.normalize(keys, dim=2)
    output = torch.nn.functional.cosine_similarity(query.unsqueeze(1), keys, dim=2)  # (N,C)
    numerator = torch.exp(torch.mean(output[:,0]) / temperature)
    denominator = torch.sum(torch.exp(torch.mean(output, dim=0) / temperature))
    return -torch.log(numerator / denominator)
