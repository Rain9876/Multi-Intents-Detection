import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.building_utils import InfoNCE


class PrototypicalContrastiveLearning(nn.Module):

    def __init__(self, config) -> None:
        super(PrototypicalContrastiveLearning, self).__init__()

        # self.all_intent_projection = AdapterLayer(config)
        # self.pre_intent_projection = AdapterLayer(config)

        # self.init_weights(self.all_intent_projection)
        # self.init_weights(self.pre_intent_projection)

        self.criterion = InfoNCE(negative_mode="paired")

    def init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # nn.init.eye_(module.weight)
            nn.init.xavier_normal_(module.weight)
            # module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, label_level_query, label_level_keys):
        """
        :param label_level_query:
        :param label_level_keys:
        :return:
        """
        # label_level_query = self.pre_intent_projection(label_level_query)
        # label_level_keys = self.all_intent_projection(label_level_keys)
        # print(label_level_query.size())
        # print(label_level_keys[:,0,:].size())
        # print(label_level_keys[:,1:,:].size())

        pcl_loss = self.criterion(label_level_query, label_level_keys[:,0,:], label_level_keys[:,1:,:])
        # pcl_loss = contrastive_loss_batch(label_level_query, label_level_keys)
        return pcl_loss


class ContrastiveLearning(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, utter_emb1, utter_emb2, labels):
        """
        :param utter_emb1: (N, H)
        :param utter_emb2: (N, H) The same embedding with X1 but with different drop mask
        :param labels: (N, )
        :return cl_loss: supervised contrastive loss in mini-batch
        """
        mapping = {}
        for i in range(self.config.num_classes):
            mapping[i] = (labels == i).nonzero(as_tuple=True)[0]
        positive_keys_index = []
        for i in labels:
            positive_keys_index.append(mapping[i.item()][torch.randint(len(mapping[i.item()]),(1,))])
        positive_index = torch.stack(positive_keys_index)  # (N, 1)

        all_key_embs = self.get_all_keys_embedding(utter_emb2, positive_index)
        cl_loss = contrastive_loss_batch(utter_emb1, all_key_embs)

        return cl_loss

    def get_all_keys_embedding(self, utter_emb, positive_index):
        """
        :param self:
        :param utter_emb: (N, H)
        :param positive_index: (N, 1) The index of utter_emb first dim as positive keys
        :return all_key_emb : keys embedding (the first vector is a positive sample by default, and the rest are negative samples)
        """
        batch_size = positive_index.size(0)
        neg_labels = torch.arange(batch_size).repeat(batch_size, 1).to(positive_index.device) #(N, N)
        mask = torch.ones_like(neg_labels).scatter_(1, positive_index, 0.)
        neg_labels = neg_labels[mask.bool()].view(batch_size, batch_size - 1)  # (N x N-1)
        label_keys = torch.cat([positive_index, neg_labels], dim=1) # (N, N)
        all_key_emb = utter_emb[label_keys, :]  # (N x N x H)

        return all_key_emb


class Sup_ContrastiveLearning(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, utter_emb, labels):
        """
        :param utter_emb: (N, H)
        :param labels: (N, )
        :return cl_loss: supervised contrastive loss in mini-batch
        """
        mapping = {}
        for i in range(self.config.num_classes):
            mapping[i] = (labels == i).nonzero(as_tuple=True)[0].tolist()

        loss = 0
        count = 0

        for i in range(labels.size(0)):  # For each sample
            neg_keys = list(range(labels.size(0)))
            pos_keys = mapping[labels[i].item()]
            pos_keys.remove(i)  # all postive except itself
            neg_keys.remove(i)  # all keys except itself

            if len(pos_keys) > 0:  # if no postive pair matched
                pos = utter_emb[torch.tensor(pos_keys), :]
                neg = utter_emb[torch.tensor(neg_keys), :]
                query = utter_emb[i, :]
                loss += contrastive_loss(query, pos, neg)
                count += 1

        cl_loss = loss / count

        return cl_loss



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


def contrastive_loss_batch(query: torch.Tensor, keys: torch.Tensor, temperature: float = 0.1):
    """
    Given the query vector and the keys matrix (the first vector is a positive sample by default, and the rest are negative samples), calculate the contrast learning loss, follow SimCLR
    :param query: shape=(N, d,)
    :param keys: shape=(N, C, d)
    :return: scalar
    """
    query = torch.nn.functional.normalize(query, dim=1)
    keys = torch.nn.functional.normalize(keys, dim=2)
    output = torch.nn.functional.cosine_similarity(query.unsqueeze(1), keys, dim=2)  # (N,C)
    numerator = torch.exp(torch.mean(output[:,0]) / temperature)
    denominator = torch.sum(torch.exp(torch.mean(output, dim=0) / temperature))
    return -torch.log(numerator / denominator)



def contrastive_loss(query: torch.Tensor, pos_keys: torch.Tensor, neg_keys: torch.Tensor, temperature: float = 0.1):
    """
    Given the query vector and the keys matrix (the first vector is a positive sample by default, and the rest are negative samples), calculate the contrast learning loss, follow SimCLR
    :param query: shape=(M, d,)
    :param keys: shape=(39, d)
    :return: scalar
    """
    query = torch.nn.functional.normalize(query, dim=0)
    pos_keys = torch.nn.functional.normalize(pos_keys, dim=1)
    neg_keys = torch.nn.functional.normalize(neg_keys, dim=1)

    pos_output = torch.nn.functional.cosine_similarity(query.unsqueeze(0), pos_keys)  # (M,)
    neg_output = torch.nn.functional.cosine_similarity(query.unsqueeze(0), neg_keys)  # (39,)

    numerator = torch.mean(torch.exp(pos_output / temperature))
    denominator = torch.sum(torch.exp(neg_output / temperature))
    return -torch.log(numerator / denominator)
