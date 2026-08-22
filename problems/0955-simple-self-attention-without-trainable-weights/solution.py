import numpy as np

def softmax(nums):
    summ = sum(np.exp(num) for num in nums)
    res = []
    for num in nums:
        res.append(np.exp(num)/summ)
    return res

def simple_self_attention(X: list[list[float]]) -> list[list[float]]:
    x = np.array(X)
    attn = x @ x.T

    context = []
    for nums in attn:
        context.append(softmax(nums))
    res = context @ x
    return  res 