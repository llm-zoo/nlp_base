import torch
import math
import torch.nn as nn
from torch.nn.init import xavier_uniform_
import numpy as np
from utils import get_attn_pad_mask


# encode和decode阶段可共用
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        :param x: [batch_size, seq_len, d_model]
        '''
        x += self.pe[:x.size(0), :]
        x = self.dropout(x)

        return x


# 定义了q,k,v的计算encode和decode可共用
class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask, d_k):
        '''
        :param Q: [b_z, nums_head, seq, d_k]
        :param K: [b_z, nums_head, seq, d_k]
        :param V: [b_z, nums_head, seq, d_k]
        :param attn_mask: [b_z, nums_head, seq, seq]
        :param d_k: embed_dim // nums_head
        :return: context: [b_z, nums_head, seq, d_k]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)

        scores.masked_fill_(attn_mask, -1e9)
        attn = torch.softmax(scores, dim=-1)

        context = torch.matmul(attn, V)

        return context


# 多头机制，共用
class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.) -> None:
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == self.embed_dim, 'embed_dim must be divisible by num_heads'

        self.q_w = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_w = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_w = nn.Linear(embed_dim, embed_dim, bias=False)

        self.fc = nn.Linear(embed_dim, embed_dim, bias=False)
        # 非动态量化线性
        # self.fc = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.q_w.weight)
        xavier_uniform_(self.k_w.weight)
        xavier_uniform_(self.v_w.weight)

    def forward(self, query, key, value, enc_attn_mask):
        '''
        :param query: [b_z, seq, 768]
        :param key: [b_z, seq, 768]
        :param value: [b_z, seq, 768]
        :param enc_attn_mask: [b_z, seq, seq]
        '''
        residual, b_s = query, query.size(0)
        d_k = self.embed_dim // self.num_heads
        # [b_z, seq, 768] -> [b_z, seq, 8, 64] -> [b_z, 8, seq, 64]
        Q = self.q_w(query).view(b_s, -1, self.num_heads, d_k).transpose(1, 2)
        K = self.k_w(key).view(b_s, -1, self.num_heads, d_k).transpose(1, 2)
        V = self.v_w(value).view(b_s, -1, self.num_heads, d_k).transpose(1, 2)

        attn_mask = enc_attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        scale = ScaleDotProductAttention()
        # context: [b_z, nums_head, seq, d_k]
        context = scale(Q, K, V, attn_mask, d_k)

        context = context.transpose(1, 2).reshape(b_s, -1, self.embed_dim)

        output = self.fc(context)
        output += residual
        output = nn.LayerNorm(self.embed_dim)(output)

        return output


# 前馈网络共用
class FeedForWard(nn.Module):
    def __init__(self, d_model, feed_forward_dim, dropout):
        super(FeedForWard, self).__init__()
        self.d_model = d_model
        self.fc = nn.Sequential(
            nn.Linear(d_model, feed_forward_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(feed_forward_dim, d_model)
        )

    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)
        output += residual
        output = nn.LayerNorm(self.d_model)(output)

        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float=0.1, feedforward_dim: int=3072):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.feedforward = FeedForWard(d_model, feedforward_dim, dropout=dropout)

    def forward(self, inputs, enc_attn_mask):
        output = self.self_attn(inputs, inputs, inputs, enc_attn_mask)
        output = self.feedforward(output)

        return output


class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super(TransformerEncoder, self).__init__()
        self.src_emb = nn.Embedding(args.vocab_size, args.d_model)
        self.pos_emb = PositionalEncoding(args.d_model, args.dropout)
        self.layers = nn.ModuleList([TransformerEncoderLayer(args.d_model, args.nums_head, args.dropout, args.feedforward_dim)
                                     for _ in range(args.n_layers)])

    def forward(self, inputs):
        outputs = self.src_emb(inputs)
        outputs = self.pos_emb(outputs)

        enc_attn_mask = get_attn_pad_mask(inputs, inputs)
        for layer in self.layers:
            outputs = layer(outputs, enc_attn_mask)

        return outputs
