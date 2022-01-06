from transformers import RobertaConfig
import collections
config = RobertaConfig()
from nltk.data import load
import nltk


Roberta_Config = collections.namedtuple('Roberta_Config', [
    "attention_probs_dropout_prob", "bos_token_id","eos_token_id",
    "gradient_checkpointing", "hidden_act", "hidden_dropout_prob",
    "hidden_size", "initializer_range", "intermediate_size", "layer_norm_eps",
    "max_position_embeddings", "model_type", "num_attention_heads", "num_hidden_layers",
    "pad_token_id", "position_embedding_type", "transformers_version",
    "type_vocab_size", "use_cache", "vocab_size"])

config = Roberta_Config (
    attention_probs_dropout_prob = 0.1,
    bos_token_id = 0,
    eos_token_id = 2,
    gradient_checkpointing = False,
    hidden_act = "gelu",
    hidden_dropout_prob = 0.1,
    hidden_size = 768,
    initializer_range = 0.02,
    intermediate_size = 3072,
    layer_norm_eps = 1e-12,
    max_position_embeddings = 512,
    model_type = "roberta",
    num_attention_heads = 12,
    num_hidden_layers = 12,
    pad_token_id = 1,
    position_embedding_type = "absolute",
    transformers_version = "4.6.0",
    type_vocab_size = 2,
    use_cache = True,
    vocab_size = 30522
)


class MyRobertaConfig(RobertaConfig):
    model_type = "roberta"

    def __init__(self, **kwargs):
        super().__init__(pad_token_id=1, bos_token_id=0, eos_token_id=2, **kwargs)
        self.classifier_dropout = 0.1
        self.num_pos_tag = 48
        self._name_or_path = "roberta-base"
        self.architectures = [
            "RobertaForMaskedLM"
        ]
        self.layer_norm_eps = 1e-05
        self.max_position_embeddings = 514
        self.type_vocab_size = 1
        self.vocab_size= 50265

class MyRobertaClassificationConfig(RobertaConfig):
    model_type = "roberta"

    def __init__(self, **kwargs):
        super().__init__(pad_token_id=1, bos_token_id=0, eos_token_id=2, **kwargs)
        self.num_labels= 151
        self.classifier_dropout = 0.1
        self.num_pos_tag = 48
        self._name_or_path = "roberta-base"
        self.architectures = [
            "RobertaForMaskedLM"
        ]
        self.layer_norm_eps = 1e-05
        self.max_position_embeddings = 514
        self.type_vocab_size= 1
        self.vocab_size = 50265



# CC 	coordinating conjunction
# CD 	cardinal digit
# DT 	determiner
# EX 	existential there
# FW 	foreign word
# IN 	preposition/subordinating conjunction
# JJ 	This NLTK POS Tag is an adjective (large)
# JJR 	adjective, comparative (larger)
# JJS 	adjective, superlative (largest)
# LS 	list market
# MD 	modal (could, will)
# NN 	noun, singular (cat, tree)
# NNS 	noun plural (desks)
# NNP 	proper noun, singular (sarah)
# NNPS 	proper noun, plural (indians or americans)
# PDT 	predeterminer (all, both, half)
# POS 	possessive ending (parent\ ‘s)
# PRP 	personal pronoun (hers, herself, him, himself)
# PRP$ 	possessive pronoun (her, his, mine, my, our )
# RB 	adverb (occasionally, swiftly)
# RBR 	adverb, comparative (greater)
# RBS 	adverb, superlative (biggest)
# RP 	particle (about)
# TO 	infinite marker (to)
# UH 	interjection (goodbye)
# VB 	verb (ask)
# VBG 	verb gerund (judging)
# VBD 	verb past tense (pleaded)
# VBN 	verb past participle (reunified)
# VBP 	verb, present tense not 3rd person singular(wrap)
# VBZ 	verb, present tense with 3rd person singular (bases)
# WDT 	wh-determiner (that, what)
# WP 	wh- pronoun (who)
# WRB 	wh- adverb (how)

def get_pos_tag_mapping():
    tagdict = load('help/tagsets/upenn_tagset.pickle')
    tag = sorted(tagdict.keys())
    tag_map = dict(map(lambda x: (x[1], x[0]+2), enumerate(tag))) # 0 for masking
    return tag_map


