import tensorflow as tf
import numpy as np
import argparse
import pandas 

parser = argparse.ArgumentParser(description='Train skipgram embeddings')
parser.add_argument('--in', dest='in_m', type=str, default='in_m.txt',
                    help='')
parser.add_argument('--out', dest='out_M', type=str, default='out_m.txt',
                    help='')
parser.add_argument('--voc', dest='voc', type=str, default='voc_fnames.tsv',
                    help='Path to vocabulary')
parser.add_argument('--allowed', dest='allowed', type=str, default='reused_call_nodes.txt',
                    help='Path to vocabulary')

args = parser.parse_args()

in_matr = np.loadtxt(args.in_m)
out_matr = np.loadtxt(args.out_M)
