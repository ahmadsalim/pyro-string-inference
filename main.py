import string
import sys

import numpy as np
import torch

from sparse_markov_chain_transition import SparseMarkovChainTransition
from string_markov_model import StringMarkovModel
from string_util import encode_string, normalize_string, decode_string, encode_and_pack_list


def simulate_mistypings(c, mistypings):
    if c in mistypings:
        (p, repl) = mistypings[c]
        if np.random.binomial(1, p) > 0:
            return repl
        else:
            return c
    else:
        return c


def main(args):
    max_chain_length = 20
    alphabet = string.ascii_lowercase + string.digits + ' '
    alphabet_size = len(alphabet) + 2
    with open('google-10000-english-no-swears.txt', 'r') as f:
        vocab = [w.rstrip() for w in f.readlines() if len(w) > 2]
    encoded_vocab_lens, encoded_vocab = encode_and_pack_list(vocab, alphabet, max_chain_length)
    smct = SparseMarkovChainTransition((encoded_vocab_lens, encoded_vocab),
                                       alphabet_size=alphabet_size,
                                       max_chain_length=max_chain_length, order=2, smoothing=0.1)
    smm = StringMarkovModel(smct)
    keywords = ['total amount', 'invoice date', 'voucher number']
    data = [np.random.choice(keywords) for i in range(30)]
    mistypings = {'o': (0.1, '0'), 'l': (0.1, '1')}
    mistyped_data = [''.join(simulate_mistypings(c, mistypings) for c in w) for w in data]
    encoded_data_length, encoded_data = encode_and_pack_list(mistyped_data, alphabet, max_chain_length)
    smm.infer(encoded_data_length, encoded_data, num_iterations=10000)


if __name__ == '__main__':
    main(sys.argv)
