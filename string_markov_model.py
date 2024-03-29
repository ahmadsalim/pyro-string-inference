import torch
import torch.nn
import tqdm
from pyro.contrib.autoguide import AutoDelta
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO
from torch.nn import functional as nnf

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.util import ignore_jit_warnings
from pyro.optim import Adam
from torch.distributions import constraints

from neural_emitter import NeuralEmitter
from sparse_markov_chain_transition import SparseMarkovChainTransition


class StringMarkovModel(torch.nn.Module):
    def __init__(self, smct: SparseMarkovChainTransition, gru_hidden: int = 50, gru_depth: int = 1):
        super(StringMarkovModel, self).__init__()
        self.smct = smct
        self.gru = torch.nn.GRU(input_size=self.smct.alphabet_size, batch_first=True,
                                num_layers=gru_depth, hidden_size=gru_hidden)
        self.neural_emitter = NeuralEmitter(gru_depth * gru_hidden, self.smct.alphabet_size,
                                            smoothing=self.smct.smoothing)

    def model(self, lengths=None, sequences=None, expected_string_length: int = 5):
        with ignore_jit_warnings():
            assert sequences is None or lengths is not None
            assert lengths is None or lengths.max() <= self.smct.max_chain_length
            assert sequences is None or (0 <= sequences.min() and sequences.max() < self.smct.alphabet_size)
        binom_prob = pyro.sample('binom_prob', dist.Beta(min(1, expected_string_length),
                                                         min(1, self.smct.max_chain_length - expected_string_length)))
        lengths_size = 1 if sequences is None else sequences.size(0)
        with pyro.plate('lengths_plate', size=lengths_size, dim=-1):
            lengths = pyro.sample('lengths', dist.Binomial(self.smct.max_chain_length, binom_prob),
                                  obs=(lengths.float() if lengths is not None else lengths))
        if lengths.dim() == 0:
            lengths = lengths.unsqueeze(-1)
        sequence_size = 1 if sequences is None else sequences.size(0)
        with pyro.plate('sequences_plate', size=sequence_size, dim=-2) as batch:
            lengths = lengths[batch]
            prev = ()
            for t in pyro.markov(range(self.smct.max_chain_length), history=self.smct.order):
                if len(prev) > self.smct.order:
                    prev = prev[1:]
                probs_t = pyro.sample(f'probs_{t}', dist.Dirichlet(self.smct.get_pseudocounts(prev).unsqueeze(-2)))
                x_t = None if sequences is None else sequences[batch, t]
                with poutine.mask(mask=(t < lengths).unsqueeze(-1).unsqueeze(-1)):
                    x_t = pyro.sample(f'x_{t}', dist.Categorical(probs=probs_t), obs=x_t)
                prev = (*prev, x_t)

    def guide(self, lengths=None, sequences=None, expected_string_length: int = 5):
        pyro.module('gru', self.gru)
        pyro.module('neural_emitter', self.neural_emitter)
        binom_prob_v = pyro.param('binom_prob_v', torch.tensor(expected_string_length / self.smct.max_chain_length),
                                  constraint=constraints.unit_interval)
        binom_prob = pyro.sample('binom_prob', dist.Delta(binom_prob_v))
        if lengths is None:
            lengths = pyro.sample('lengths', dist.Binomial(self.smct.max_chain_length, binom_prob)).unsqueeze(-1)
        sequence_size = 1 if sequences is None else sequences.size(0)
        initial_pseudocounts = pyro.param('initial_pseudocounts',
                                          torch.ones(self.smct.alphabet_size, dtype=torch.float,
                                                     device=lengths.device),
                                          constraint=constraints.interval(1, 100))
        with pyro.plate('sequences', size=sequence_size, dim=-2) as batch:
            for t in pyro.markov(range(self.smct.max_chain_length), history=self.smct.order):
                if t == 0:
                    probs_t = pyro.sample(f'probs_{t}', dist.Dirichlet(initial_pseudocounts.unsqueeze(-2)).to_event())
                    h_t = torch.randn(self.gru.num_layers, sequence_size, self.gru.hidden_size, dtype=torch.float,
                                      device=lengths.device)
                else:
                    if sequences is not None:
                        x_t = nnf.one_hot(sequences[batch, t - 1:t], num_classes=self.smct.alphabet_size).float()
                    else:
                        x_t = dist.OneHotCategorical(probs_t).sample()
                    gru_out_t, h_t = self.gru.forward(x_t, h_t)
                    pseudo_counts_t = self.neural_emitter.forward(gru_out_t)
                    probs_t = pyro.sample(f'probs_{t}', dist.Dirichlet(pseudo_counts_t))

    def infer(self, lengths, sequences, num_iterations=100):
        optim = Adam(dict(lr=1e-1))
        elbo = Trace_ELBO()
        svi = SVI(self.model, self.guide, optim, elbo)
        pbar = tqdm.tqdm(range(num_iterations))
        for i in pbar:
            loss = svi.step(lengths, sequences)
            pbar.set_description(f'Epoch {i}: {loss}')

    def forward(self):
        tr = poutine.trace(self.model).get_trace()
        l = tr.nodes['lengths']['value']
        x = torch.cat([tr.nodes[f'x_{t}']['value'] for t in range(int(l))], dim=-1)
        return x.squeeze(0)
