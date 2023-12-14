import matplotlib.pylab as plt
import torch
import numpy as np


def plot(arr, x_str: str, y_str: str, title: str):
    plt.figure(figsize=(10, 6))
    plt.imshow(arr, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.xlabel(x_str)
    plt.ylabel(y_str)
    plt.title(title)
    plt.show()


# 计算attention
def get_attn_pad_mask(seq_q, seq_k):
    b_s, len_q = seq_q.size()
    b_s, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    pad_attn_mask = pad_attn_mask.expand(b_s, len_q, len_k)

    return pad_attn_mask


# 计算doceder阶段的遮蔽
def get_attn_subsequence_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)

    subsequence_mask = torch.from_numpy(subsequence_mask).byte()

    return subsequence_mask

