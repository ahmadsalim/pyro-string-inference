import string
import sys
from typing import List

import torch
import numpy as np

from sparse_markov_chain_transition import SparseMarkovChainTransition
from string_util import encode_string, normalize_string, decode_string


def main(args):
    alphabet = string.ascii_lowercase + string.digits + ' '
    vocab = ['hest'] * 30 + ['spand'] * 50 + ['tæppe'] * 70 + ['banan'] * 30 + ['fest'] * 40
    encoded = [encode_string(normalize_string(w), alphabet) for w in vocab]
    encoded_lens = torch.tensor([len(w) for w in encoded])
    encoded = torch.tensor([w + [0] * (max(map(len, encoded)) - int(l)) for (w, l) in zip(encoded, encoded_lens)])
    encoded_alphabet_size = len(alphabet) + 2
    smct = SparseMarkovChainTransition((encoded_lens, encoded), alphabet_size=encoded_alphabet_size,
                                       max_chain_length=10, order=3, smoothing=1.0)
    prev: List[int] = []
    for i in range(5):
        als = smct.get_pseudocounts(prev)
        ps = np.random.dirichlet(als)
        c = np.random.multinomial(1, ps).argmax()
        prev.append(c)
    print(decode_string(prev, alphabet))

    x = 1

if __name__ == '__main__':
    main(sys.argv)