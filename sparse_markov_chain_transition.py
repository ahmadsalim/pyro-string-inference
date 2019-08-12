from typing import List

import torch


class SparseMarkovChainTransition:
    def __init__(self, prior_chains: (torch.Tensor, torch.Tensor), *,
                 alphabet_size: int, max_chain_length: int, order: int = 1,
                 smoothing: float = 1):
        assert 0 < alphabet_size
        assert 0 < max_chain_length
        assert 0 < order
        assert prior_chains[1].dim() == 2
        assert 0 <= prior_chains[1].size(-1) < max_chain_length
        assert 0 <= prior_chains[0].max() < max_chain_length
        assert 0 <= prior_chains[1].min()
        assert prior_chains[1].max() < alphabet_size
        assert 0 < smoothing
        self.alphabet_size = alphabet_size
        self.max_chain_length = max_chain_length
        self.order = order
        self.smoothing = smoothing
        self._initialize_tables(prior_chains)
        self._add_prior_chains(prior_chains)

    def _initialize_tables(self, prior_chains):
        self._transition_tables = []
        for i in range(self.order + 1):
            # This should really be sparse
            self._transition_tables.append(torch.ones((self.alphabet_size,) * (i + 1),
                                                      dtype=torch.float,
                                                      device=prior_chains[1].device) * self.smoothing)

    def _add_prior_chains(self, prior_chains):
        for i in range(prior_chains[1].size(0)):
            l = prior_chains[0][i]
            prev = ()
            for j in range(l):
                elem = int(prior_chains[1][i, j])
                if len(prev) > self.order:
                    prev = prev[1:]
                if j == 0:
                    self._transition_tables[j][elem] += 1
                else:
                    self._transition_tables[min(j, self.order)][prev][elem] += 1
                prev = prev + (elem,)

    def get_pseudocounts(self, prev: List[torch.Tensor]):
        expected_order = min(len(prev), self.order)
        return self._transition_tables[expected_order][tuple(prev)[:expected_order]]
