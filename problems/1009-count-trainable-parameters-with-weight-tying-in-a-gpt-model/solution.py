def count_gpt_parameters(cfg: dict) -> list:
    tok_emb = cfg["vocab_size"] * cfg["emb_dim"]
    pos_emb = cfg["context_length"] * cfg["emb_dim"]
    qkv = 3 * cfg["emb_dim"] * cfg["emb_dim"] 
    if cfg["qkv_bias"]:
        qkv += 3*cfg["emb_dim"]
    attn = cfg["emb_dim"] * cfg["emb_dim"] + cfg["emb_dim"]
    layers = qkv * cfg["n_layers"] 
    norm = 2 * 2 * cfg["emb_dim"]
    ffn = 2 * 4 * cfg["emb_dim"] * cfg["emb_dim"] + 4*cfg["emb_dim"] + cfg["emb_dim"]
    block = qkv + attn  + norm + ffn
    transformer = cfg["n_layers"] * block
    final_norm = 2*cfg["emb_dim"]
    lm_head = cfg["emb_dim"] * cfg["vocab_size"]
    total_params =  tok_emb + pos_emb + transformer + final_norm + lm_head
    tied_params = total_params - lm_head
    return [total_params,tied_params]
