
from typing import Optional, Tuple, Union, List, Dict
import numpy as np
from copy import deepcopy
from abc import ABC, abstractmethod

from .generator_2 import Generator

def combine_equations(*equations):
    # Base case: if there's only one equation, just return it
    if len(equations) == 1:
        return equations[0]

    # Step case: combine the first two equations and recursively call the function with the result
    combined = [f + e for f in equations[1] for e in equations[0]]

    # If there are more equations left, combine further
    if len(equations) > 2:
        return combine_equations(combined, *equations[2:])
    else:
        return np.asarray(combined)

def combine_sigmas(*sigmas):
    # Base case: if there's only one equation, just return it
    if len(sigmas) == 1:
        return sigmas[0]

    if (np.isfinite(sigmas[0])).any():
        if sigmas[1][0] == np.inf:
            sigmas[1][0] = 0
    if (np.isfinite(sigmas[1])).any():
        if sigmas[0][0] == np.inf:
            sigmas[0][0] = 0

    # Step case: combine the first two equations and recursively call the function with the result
    combined = [(f**2 + e**2) ** 0.5 for f in sigmas[1] for e in sigmas[0]]

    # If there are more equations left, combine further
    if len(sigmas) > 2:
        return combine_sigmas(combined, *sigmas[2:])
    else:
        return np.asarray(combined)
    
def combine_mus(*mus):
    return combine_equations(*mus)

def combine_distibutions(*distributions):
    # separate the mus and sigmas into separate arrays
    c_mus = combine_mus(*[distribution[0] for distribution in distributions])
    c_sigmas =combine_sigmas(*[distribution[1] for distribution in distributions])

    # zip them back into a distributions tuple and return
    return [(c_mus[i], c_sigmas[i]) for i in range(len(c_mus))]

def combine_matrices(*matrices):
    # Base case: if there's only one equation, just return it
    if len(matrices) == 1:
        return matrices[0]
    # Step case: combine the first two equations and recursively call the function with the result
    combined = [matrices[0] * f[:, None] for f in matrices[1].T]

    # If there are more equations left, combine further
    if len(matrices) > 2:
        return np.hstack(combine_matrices(combined, *matrices[2:]))
    else:
        return np.hstack(combined)


class StackedIndependentGenerator(Generator):
    def __init__(self, *args):
        # Check that every arg is a generator
        if not np.all([isinstance(a, Generator) for a in args]):
            raise ValueError("Can only combine `Generator` objects.")

        # Check that every generator has the same data shape
        if (
            not len(np.unique([a.data_shape for a in args if a.data_shape is not None]))
            <= 1
        ):
            raise ValueError("Can not have different `data_shape`.")

        self.generators = [a.copy() for a in args]
        self.data_shape = self.generators[0].data_shape
        self.lengths = [g.width for g in self.generators]
        self.fit_distributions = [(None, None) for g in self.generators for v in g.fit_distributions]
        # self.fit_distributions = [v for g in self.generators for v in g.fit_distributions]

    def __repr__(self):
        fit = "fit" if self.fit_mu is not None else ""
        return f"{type(self).__name__}({', '.join(list(self.arg_names))})[n, {self.width}] {fit}"
    
    def set_prior(self, index, value):
        # need to set the value both in self.priors and in self.generators.priors
        super().set_prior(index, value)

        # add in another line to set the prior for the corresponding sub generator
        # see self.fit() for code to work from on this
        raise NotImplementedError

    def prior_distributions(self):
        return [p for g in self.generators for p in g.prior_distributions]

    # should this be a property?
    def design_matrix(self, *args, **kwargs):
        return np.hstack([g.design_matrix(*args, **kwargs) for g in self.generators])

    @property
    def width(self):
        return np.sum([g.width for g in self.generators])

    @property
    def nvectors(self):
        return np.sum([g.nvectors for g in self.generators])

    def fit(self, *args, **kwargs):
        self.fit_mu, self.fit_sigma = self._fit(*args, **kwargs)
        # lengths = [g.width for g in self.generators]
        mu, sigma = (
            np.array_split(self.fit_mu, np.cumsum(self.lengths))[:-1],
            np.array_split(self.fit_sigma, np.cumsum(self.lengths))[:-1],
        )
        for idx, mu0, sigma0 in zip(np.arange(len(mu)), mu, sigma):
            self[idx].fit_mu = mu0
            self[idx].fit_sigma = sigma0

        indices = np.cumsum([0, *[g.width for g in self.generators]])
        for idx, a, b in zip(range(len(indices) - 1), indices[:-1], indices[1:]):
            self[idx].cov = self.cov[a:b, a:b]


class CrosstermGenerator(StackedIndependentGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prior_distributions = combine_distibutions(*[g.prior_distributions for g in self.generators])

    def design_matrix(self, *args, **kwargs):
        return combine_matrices(
            *[g.design_matrix(*args, **kwargs) for g in self.generators]
        )
    
    @property
    def _equation(self):
        return combine_equations(*[g._equation for g in self.generators])

    @property
    def arg_names(self):
        return np.unique(np.hstack([list(g.arg_names) for g in self.generators]))

    @property
    def nvectors(self):
        return len(self.arg_names)

    @property
    def width(self):
        return np.prod([g.width for g in self.generators])
    
    def fit(self, *args, **kwargs):
        self.fit_mu, self.fit_sigma = self._fit(*args, **kwargs)


