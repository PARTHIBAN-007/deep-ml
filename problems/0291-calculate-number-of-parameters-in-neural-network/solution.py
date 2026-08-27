def count_parameters(layers: list) -> int:
    total_params = 0
    
    for layer in layers:
        layer_type = layer.get("type")
        
        if layer_type == "dense":
            input_size = layer["input_size"]
            output_size = layer["output_size"]
            use_bias = layer.get("use_bias", True)
            
            weights = input_size * output_size
            biases = output_size if use_bias else 0
            total_params += weights + biases
            
        elif layer_type == "conv2d":
            in_channels = layer["in_channels"]
            out_channels = layer["out_channels"]
            kernel_size = layer["kernel_size"]
            use_bias = layer.get("use_bias", True)
            
            if isinstance(kernel_size, tuple):
                kernel_area = kernel_size[0] * kernel_size[1]
            else:
                kernel_area = kernel_size * kernel_size
                
            weights = in_channels * kernel_area * out_channels
            biases = out_channels if use_bias else 0
            total_params += weights + biases
            
        elif layer_type == "embedding":
            num_embeddings = layer["num_embeddings"]
            embedding_dim = layer["embedding_dim"]
            
            weights = num_embeddings * embedding_dim
            total_params += weights
            
    return total_params