import torch
import pyro
from pyro.distributions import TorchDistribution

class SparseMarkovChain(TorchDistribution):
    arg_constraints = {}

    def __init__(self, *, alphabet_size: int, prior_chains: torch.Tensor,
                 max_chain_length: int, order: int = 1,
                 batch_shape=torch.Size([]), validate_args=None):
        super(SparseMarkovChain, self).__init__(batch_shape=batch_shape,
                                                # Length + distribution over alphabet characters
                                                event_shape=1 + alphabet_size,
                                                validate_args=validate_args)
        assert 0 < alphabet_size
        assert 0 < max_chain_length
        assert 0 < order
        assert prior_chains.dim() == 2
        assert 1 <= prior_chains.size(-1) < 1 + max_chain_length
        assert 0 <= prior_chains[:, 0].max() < max_chain_length
        assert 0 <= prior_chains[:, 1:].min()
        assert prior_chains[:, 1:].max() < alphabet_size
        self.alphabet_size = alphabet_size
        self.max_chain_length = max_chain_length
        self.order = order
        self._initialize_tables(prior_chains)
        self._add_prior_chains(prior_chains)

    def _initialize_tables(self, prior_chains):
        self._transition_tables = []
        for i in range(self.order + 1):
            # This should really be sparse
            self._transition_tables.append(torch.ones((self.alphabet_size,) * (i + 1),
                                                      dtype=prior_chains.dtype,
                                                      device=prior_chains.device))

    def _add_prior_chains(self, prior_chains):
        for i in range(prior_chains.size(0)):
            l = int(prior_chains[i, 0])
            prev = ()
            for j in range(l):
                elem = int(prior_chains[i, j + 1])
                if len(prev) > self.order:
                    prev = prev[1:]
                if j == 0:
                    self._transition_tables[j][elem] += 1
                else:
                    self._transition_tables[j][prev][elem] += 1
                prev = prev + (elem,)
