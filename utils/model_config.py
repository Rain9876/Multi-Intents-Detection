
class Config ():

  # "attention_probs_dropout_prob": 0.1,
  # "gradient_checkpointing": false,
  # "hidden_act": "gelu",
  # "hidden_dropout_prob": 0.1,
  # "hidden_size": 768,
  # "initializer_range": 0.02,
  # "intermediate_size": 3072,
  # "layer_norm_eps": 1e-12,
  # "max_position_embeddings": 512,
  # "model_type": "bert",
  # "num_attention_heads": 12,
  # "num_hidden_layers": 12,
  # "pad_token_id": 0,
  # "position_embedding_type": "absolute",
  # "transformers_version": "4.6.0",
  # "type_vocab_size": 2,
  # "use_cache": true,
  # "vocab_size": "bert-base-uncased"
  emb_path = "./util/emb.pt"
  num_classes = 18
  n_head = 12
  hidden = 768
  d_inner = 3072
  pad_token_id = 0
  dropout = 0.1
  d_v = 64
  d_k = 64
  initializer_range = 0.02
  label_pretrained = True
  file_path = "label_emb.pt"
  load_label_emb = False
  mode = "pooling"
  adapter_latent = 400






