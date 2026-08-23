def vocab_size(tokens, special_tokens=None):
    tokens_set = set(tokens)
    if special_tokens is not None:
        vocab = tokens_set.union(set(special_tokens))
        return len(vocab)
    return len(tokens_set)
