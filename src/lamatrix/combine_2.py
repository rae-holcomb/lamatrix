
from typing import Optional, Tuple, Union, List, Dict
import numpy as np
from copy import deepcopy
from abc import ABC, abstractmethod

from .generator_2 import Generator

# to do 7/22
# make PolynomialGenerator and OffsetGenerator
# Implement option 1a for multiplication, aka CrosstermGenerator
# Implement the inclusion of OffsetGenerator into StackedIndependentGenerator


# Questions
# Why are __add__ and __mul__ only defined for the Independent generator?

g1 = Polynomial()
g2 = Spline()

g1.prior_distribution --> [(value, value), (value, value), (value, value)]

SIG = g1 + g2

SIG.update_the_priors_somehow_for_g1_to_zeros()

SIG[0].prior_distribution --> [(0, 0), (0, 0), (0, 0)]
g1.prior_distribution --> [(value, value), (value, value), (value, value)]


SIG.breakdown(0) --> offset, generators with no offset
SIG = SIG[0] + SIG[1]

SIG[0] --> missing offset term
SIG[1] --> missing offset term
SIG[2] --> missing offset term
...
SIG = offset + SIG[0] + SIG[1]


g1 + g2
g2 + g1


SDG = g1 * g2
cannot do this: SDG[0]

######

SIG is made up of an offset, an augmented deepcopy of g1, and an augmented deepcopy of g2

all design matrices have a column of ones

NO design matrices have columns of ones, and to fit a model you explicitly must add a constant offset term

-> SplineGenerator() -> DM is splines EXCEPT ones
-> PolynomialGenerator() -> Is polynomial EXCEPT ones
-> OffsetGenerator() -> DM is just ones

SIG = (PolynomialGenerator('v', ...) + Spline('r', ...) + OffsetGenerator())

SIG[0]

class OffsetGenerator(PolynomialGenerator):
    """A generator which has no variable, and whos design matrix is entirely ones.""""
    def __init__(self):
        super().__init__(polyorder=0)

    def __repr__(self):
        ....

SIG.set_prior(index=15, value=1)
SIG[1].set_prior(index=15, value=1)


SIG = (OffsetGenerator() + SinusoidGenerator('x', ...))
SIG[0]
SIG[1]

(SIG + GaussianGenerator('y'...))


# We should protect against multiple OffsetGenerators being added to an independent generator
# also pay attention to whether stacking stacked independent gens appends or leads to a nested structure (it should append)
# fit parameters should be zero'd out after stacking, if people want to use the fit parameters they should update their priors.
#####

p1 = PolynomialGenerator('v', ..., polyorder=4)
p2 = PolynomialGenerator('z', ..., polyorder=4)

p1 * p2

(1 + v + v**2 + v**3 + v**4) 
+ z(1 + v + v**2 + v**3 + v**4) 
+ z**2(1 + v + v**2 + v**3 + v**4) 

1. Should multiply add in an offset term to ensure that the non-cross-terms are captured?
1a. p1 * p2 --> p1 + p2 + p1 x p2
2. Should we force users to do that explicly

CosineGenerator(phi) * SineGenerator(phi)

-> Cosine*Sine
-> Cosine*Sine, Sine, Cosine


p1 * p2 = p1 + p2 + p1 x p2

StackedIndependentGenerator([p1, p2, CrossTermGenerator(p1xp2)])




# offsets are special cases
# What happens when people make degenerate models? 


# g1 == SIG[0]? 
# g2 == SIG[1]?


class StackedIndependentGenerator(Generator):
    def __init__(self, *args):
        # we need to check that every arg is a generator?
        self.generators = [a.copy() for a in args]
        self.fit_distributions = [(None, None) for g in self.generators for v in g.fit_distributions]
        # self.fit_distributions = [v for g in self.generators for v in g.fit_distributions]


    def __init__(self, *args, **kwargs):
        if (
            not len(np.unique([a.data_shape for a in args if a.data_shape is not None]))
            <= 1
        ):
            raise ValueError("Can not have different `data_shape`.")
        self.generators = [a.copy() for a in args]
        self.data_shape = self.generators[0].data_shape
        # self.fit_mu = None
        # self.fit_sigma = None

    def __repr__(self):
        fit = "fit" if self.fit_mu is not None else ""
        return f"{type(self).__name__}({', '.join(list(self.arg_names))})[n, {self.width}] {fit}"
    
    def set_prior(self, index, value):
        # need to set the value both in self.priors and in self.generators.priors
        super().set_prior(index, value)

        # add in another line to set the prior for the corresponding sub generator
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
