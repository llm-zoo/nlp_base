import torch
from argparse import ArgumentParser


parser = ArgumentParser()

parser.add_argument('--vocab_size', type=int, default=20000, help='')
parser.add_argument('--vocab_dec_size', type=int, default=20000, help='')
parser.add_argument('--d_model', type=int, default=512, help='')
parser.add_argument('--n_layers', type=int, default=12, help='')
parser.add_argument('--nums_head', type=int, default=8, help='')
parser.add_argument('--feedforward_dim', type=int, default=3072, help='')
parser.add_argument('--dropout', type=float, default=0.1, help='')

args = parser.parse_args()


