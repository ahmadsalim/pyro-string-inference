import string
import sys

import torch

from sparse_markov_chain_transition import SparseMarkovChainTransition
from string_markov_model import StringMarkovModel
from string_util import encode_string, normalize_string, decode_string


def main(args):
    alphabet = string.ascii_lowercase + string.digits + ' '
    vocab = ['spand'] * 50 + ['hest'] * 30 + ['tæppe'] * 70 + ['banan'] * 30 + ['fest'] * 40
    encoded = [encode_string(normalize_string(w, eos_marker=False), alphabet) for w in vocab]
    encoded_lens = torch.tensor([len(w) for w in encoded])
    encoded = torch.tensor([w + [0] * (max(map(len, encoded)) - int(l)) for (w, l) in zip(encoded, encoded_lens)])
    encoded_alphabet_size = len(alphabet) + 2
    smct = SparseMarkovChainTransition((encoded_lens, encoded), alphabet_size=encoded_alphabet_size,
                                       max_chain_length=10, order=1, smoothing=0.1)
    smm = StringMarkovModel(smct)
    x = list(smm.forward().numpy())
    print(decode_string(x, alphabet))


if __name__ == '__main__':
    main(sys.argv)
