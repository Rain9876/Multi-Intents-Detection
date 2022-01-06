import os
from torch.distributed import get_rank
import torch
import matplotlib.pyplot as plt


def fix_state_dict_namespace(model_state_dict, local_rank=-1):
    old_keys = []
    new_keys = []
    for t in list(model_state_dict.keys()).copy():
        new_key = t

        if new_key.startswith('module.'):
            new_key = new_key.replace('module.', '')
        elif new_key.startswith('model.'):
            new_key = new_key.replace('model.', '')

        if new_key.endswith('.beta'):
            new_key = new_key.replace('.beta', '.bias')
        elif new_key.endswith('.gamma'):
            new_key = new_key.replace('.gamma', '.weight')

        old_keys.append(t)
        new_keys.append(new_key)

        # for requirement, mapping in zip(added_keys_requirements, added_keys_mappings):
        #    if all(r in new_key for r in requirement):
        #        for ori, new in mapping:
        #            #logger.info(new_key + '->' + new_key.replace(ori, new))
        #            added_key = new_key.replace(ori, new)
        #            model_state_dict[added_key] = model_state_dict[t].clone()

    for old_key, new_key in zip(old_keys, new_keys):
        model_state_dict[new_key] = model_state_dict.pop(old_key)

    if 'shared.weight' in new_keys and 'encoder.embed_tokens.weight' not in new_keys:
        model_state_dict['encoder.embed_tokens.weight'] = model_state_dict['shared.weight'].clone()
        if local_rank == -1 or get_rank() == 0:
            print(' cloning [encoder.embed_tokens.weight] from [shared.weight]...')

    if 'shared.weight' in new_keys and 'decoder.embed_tokens.weight' not in new_keys:
        model_state_dict['decoder.embed_tokens.weight'] = model_state_dict['shared.weight'].clone()
        if local_rank == -1 or get_rank() == 0:
            print(' cloning [decoder.embed_tokens.weight] from [shared.weight]...')

    return model_state_dict


def load_model(model, checkpoint, local_rank=-1):

    if checkpoint is not None and checkpoint.lower() != "none":
        if not os.path.exists(checkpoint):
            raise ValueError('checkpoint %s not exist' % checkpoint)
        model_state_dict = torch.load(checkpoint)
        model_state_dict = fix_state_dict_namespace(model_state_dict, local_rank)
        if local_rank == -1 or get_rank() == 0:
            print('loading finetuned model from %s' % checkpoint)

        strict = False
        if hasattr(model, 'transformer') and all(not e.startswith('transformer.') for e in model_state_dict.keys()):
            model = model.transformer
        if hasattr(model, 'tower') and model.tower:
            strict = True

        needed_keys = set(dict(model.named_parameters()).keys())
        loaded_keys = []
        for k, v in model_state_dict.items():
            if k not in needed_keys:
                continue
            try:
                model.load_state_dict({k: v}, strict=False)
                # if local_rank == -1 or get_rank() == 0:
                #    logger.info(' parameter [%s] loaded!' % k)
                loaded_keys.append(k)
            except RuntimeError as e:
                if local_rank == -1 or get_rank() == 0:
                    print(' ??? unmatched parameter [%s]' % k)
                if strict:
                    raise e

        loaded_keys = set(loaded_keys)
        missed_keys = needed_keys - loaded_keys

        if local_rank == -1 or get_rank() == 0:
            if len(missed_keys) > 0:
                for k in sorted(missed_keys):
                    print(' !!! parameter [%s] missed' % k)

    return model

########################################################################################################

########################################################################################################

"""Learning rate scheduler"""

import math
import torch
from torch.optim.optimizer import Optimizer


class LRSchedular(object):
    """Base class for learning rate schedular
    Arguments:
        optimizer (Optimizer): an instance of a subclass of Optimizer
        last_step (int): The index of last step. (Default: -1)
    """

    def __init__(self, optimizer, last_step=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError("{} is not an Optimizer".format(type(optimizer).__name__))
        self.optimizer = optimizer
        self.last_step = last_step
        self.base_lrs = [group["lr"] for group in self.optimizer.param_groups]
        self.lrs = self.base_lrs
        self.step()

    def state_dict(self):
        """Returns the state of the learning rate scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }

    def load_state_dict(self, state_dict):
        """Loads the learning rate scheduler's state.
        Arguments:
            state_dict (dict): warmup scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lr(self):
        return self.base_lrs

    def step(self, step=None):
        """Update the learning rates.
        Arguments:
            step (int): The index of current step. (Default: None)
        """
        if step is None:
            step = self.last_step + 1
        self.last_step = step

        lr_values = self.get_lr()
        self.lrs = lr_values

        for group, lr in zip(self.optimizer.param_groups, lr_values):
            group["lr"] = lr


class LinearWarmupRsqrtDecayLR(LRSchedular):
    """Learning rate warmup at the beginning then decay.

    References:
        https://arxiv.org/pdf/1804.00247.pdf Section 4.6
        https://github.com/google-research/pegasus/blob/13fcf2b1191e0df950436c82c33d672c1447f5ff/pegasus/params/estimator_utils.py#L112

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_step (int): Number of step for warmup.
        last_step (int): Last step. Default: -1.

    """

    def __init__(self, optimizer, warmup_step, last_step=-1):
        self.warmup_step = warmup_step
        super(LinearWarmupRsqrtDecayLR, self).__init__(optimizer, last_step)

    def get_lr(self):
        lr_values = list()
        for base_lr in self.base_lrs:
            lr = (
                    base_lr
                    * math.sqrt(self.warmup_step)
                    / math.sqrt(max(self.last_step, self.warmup_step))
            )
            lr = min((self.last_step + 1) / self.warmup_step * base_lr, lr)
            lr_values.append(lr)
        return lr_values


########################################################################################################

########################################################################################################
import re

def map_pos_tag_to_tokens(tokens, ids):
    if "[]" in ids:
        return []

    ids = re.findall("\d+", ids)
    num = 0
    idx = 0
    target_token = []
    for i in range(len(tokens)):
        token = tokens[i]
        num += len(token)
        if "<s>" in token:
            num -= 3
        if "</s>" in token:
            num -= 4
        if token.startswith("_"):
            num -= 1
        if num > int(ids[idx]):
            target_token.append(i)
            idx += 1
        if idx == len(ids):
            break
    return target_token