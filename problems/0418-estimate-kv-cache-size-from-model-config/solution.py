def estimate_kv_cache_size(model_config: dict, batch_size: int, seq_len: int) -> dict:
    if model_config.get("num_kv_heads") is not None:
        num_of_heads = model_config["num_kv_heads"]
    else:
        num_of_heads = model_config["num_attention_heads"]
    
    head_dim = model_config["hidden_size"] / model_config["num_attention_heads"]
    elements = batch_size * model_config["num_layers"] * 2  * num_of_heads * head_dim * seq_len
    bytes =  elements * model_config["dtype_bytes"]
    mb = bytes / (1024**2)
    layer = mb / model_config["num_layers"]
    token_size = model_config["num_layers"] * 2 * num_of_heads * head_dim * model_config["dtype_bytes"]

    return {
        "kv_cache_elements": elements,
        "kv_cache_size_bytes": bytes,
        "kv_cache_size_mb": round(mb,4),
        "per_layer_size_mb": round(layer,4),
        "per_token_size_kb": token_size/1024
    }