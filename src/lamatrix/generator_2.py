
from typing import Optional, Tuple, Union, List, Dict
import numpy as np
from copy import deepcopy
from abc import ABC, abstractmethod
from .math_mixins import MathMixins



class Generator(ABC):
    def __init__(
        self,
        prior_distributions : List[Tuple]=None,
        *args, **kwargs
    ):
        # prior_distrbutions (always normal, and always specfied by mean and standard deviation)
        # fit_distributions (always normal, and always specfied by mean and standard deviation)

        if prior_distributions is None:
            self.prior_distributions = [(0, np.inf)] * self.width # A list of tuples
        else:
            self._validate_distributions(prior_distributions)
            self.prior_distributions = prior_distributions

        self.fit_distributions = [(None, None)] * self.width


    def _validate_arg_names(self):
        for arg in self.arg_names:
            if not isinstance(arg, str):
                raise ValueError("Argument names must be strings.")

    @property
    @abstractmethod
    def width(self):
        """Returns the width of the design matrix once built."""
        pass

    @property
    @abstractmethod
    def nvectors(self):
        """Returns the number of vectors required to build the object."""
        pass

    def __repr__(self):
        return "Generator"

    def copy(self):
        return deepcopy(self)
    
    def _validate_distribution(self,distribution:Tuple):
        """Checks that a distribution is a valid input with format (mean, std)."""
        # Is it a tuple of mean and std

        # Must be a tuple
        if not isinstance(distribution, tuple):
            raise ValueError("distribution must be a tuple of format (mean, std)")            
        # Must be float or int
        for value in distribution:
            if not isinstance(distribution, (float, int, np.int, np.float)):
                raise ValueError("Values in distribution must be numeric.")
        # Standard deviation must be positive
        if np.sign(distribution[1]) == -1:
            raise ValueError("Standard deviation must be positive.")
        return

    def _validate_distributions(self, distributions:List[Tuple]):
        """Checks that a list of distributions is a valid"""
        if not len(distributions) == self.width:
            raise ValueError("distributions must have the number of elements as the design matrix.")
        [self._validate_distribution(distribution) for distribution in distributions]
        return

    def set_prior(self, index:int, value:Tuple) -> None:
        """Sets a single prior."""
        # Checking this is a tuple with bounds (-np.inf:np.inf, 0:np.inf)
        # validate the prior
        self._validate_distribution(distribution=(index, value))
        self.prior_distributions[index] = value
        return

    def set_priors(self, distributions:List[Tuple]) -> None:
        """sets the full list of priors"""
        # Checking this is a tuple with bounds (-np.inf:np.inf, 0:np.inf)
        # validate the prior
        self._validate_distributions(distributions=distributions)
        self.prior_distributions = distributions
        return

    def freeze_element(self, index:int):
        """Freezes an element of the design matrix by setting prior_sigma to zero."""
        self.set_prior(index, (self.prior_distributions[index][0], 0))

    @property
    def prior_mean(self):
        return [distribution[0] for distribution in self.prior_distributions]

    @property
    def prior_std(self):
        return [distribution[1] for distribution in self.prior_distributions]

    @property
    def fit_mean(self):
        return [distribution[0] for distribution in self.fit_distributions]

    @property
    def fit_std(self):
        return [distribution[1] for distribution in self.fit_distributions]


    # helper function headers copied over from the old Generator object... 
    # but I won't copy them all here right now, can fill in later
    def _create_save_data(self):
        raise NotImplementedError
    
    @staticmethod
    def format_significant_figures(mean, error):
        raise NotImplementedError
    
    def _get_table_matter(self):
        raise NotImplementedError
    
    def _to_latex_table(self):
        raise NotImplementedError
    
    def to_latex(self):
        raise NotImplementedError
    
    def evaluate(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def design_matrix(self):
        """Returns a design matrix, given inputs listed in self.arg_names."""
        pass

    def fit(self, data, errors=None, *args, **kwargs):
        """Some comment"""
        X = self.design_matrix(*args, **kwargs)
        if np.prod(data.shape) != X.shape[0]:
            raise ValueError(f"Data must have shape {X.shape[0]}")
        if errors is None:
            errors = np.ones_like(data)
        self.data_shape = data.shape
        sigma_w_inv = X.T.dot(
            X / errors.ravel() ** 2
        ) + np.diag(1 / self.prior_std**2)
        self.cov = np.linalg.inv(sigma_w_inv)
        B = X.T.dot(
            data.ravel() / errors.ravel() ** 2
        ) + np.nan_to_num(self.prior_mean / self.prior_std**2)
        fit_mean = np.linalg.solve(sigma_w_inv, B)
        fit_std = self.cov.diagonal() ** 0.5
        return fit_mean, fit_std


class Polynomial1DGenerator(Generator):
    def __init__(
        self,
        x_name: str = "x",
        polyorder: int = 3,
        prior_distributions=None,
        data_shape=None,
    ):        
        if polyorder < 1:
            raise ValueError("Must have polyorder >= 1.")
        self.x_name = x_name
        # self._validate_arg_names()
        self.polyorder = polyorder
        self.data_shape = data_shape
        
        # calling this will validate the priors fill out the default if None is given
        super().__init__(prior_distributions=prior_distributions)

    @property
    def width(self):
        return self.polyorder

    @property
    def nvectors(self):
        return 1

    @property
    def arg_names(self):
        return {self.x_name}

    # @property
    # def _INIT_ATTRS(self):
    #     return [
    #         "x_name",
    #         "prior_mu",
    #         "prior_sigma",
    #         "data_shape",
    #         "polyorder",
    #     ]

    def design_matrix(self, *args, **kwargs):
        """Build a 1D polynomial in x

        Parameters
        ----------
        {} : np.ndarray
            Vector to create polynomial of

        Returns
        -------
        X : np.ndarray
            Design matrix with shape (len(x), self.nvectors)
        """
        if not self.arg_names.issubset(set(kwargs.keys())):
            raise ValueError(f"Expected {self.arg_names} to be passed.")
        x = kwargs.get(self.x_name).ravel()
        return np.vstack([x**idx for idx in range(1, self.polyorder + 1)]).T

    def fit(self, *args, **kwargs):
        self.fit_mu, self.fit_sigma = self._fit(*args, **kwargs)

    @property
    def _equation(self):
        eqn = [
            f"\mathbf{{{self.x_name}}}^{{{idx}}}" for idx in range(1, self.polyorder + 1)
        ]
        #        eqn[0] = ""
        return eqn
    
    def get_term(self, index):
        """Returns the term in the equation corresponding to a certain index. This will make it easier for the user to make sure they are working with the right element of the design matrix."""
        raise NotImplementedError

    # @property
    # def gradient(self):
    #     return dPolynomial1DGenerator(
    #         weights=self._mu,
    #         x_name=self.x_name,
    #         polyorder=self.polyorder,
    #         data_shape=self.data_shape,
    #         offset_prior=(self._mu[1], self._sigma[1]),
    #     )


class OffsetGenerator(Generator):
    """
    A generator which has no variable, and whos design matrix is entirely ones.
    """
    def __init__(self, prior_distributions=None):
        super().__init__(prior_distributions=prior_distributions)

        # need to define self.data_shape somehow... 

    def __repr__(self):
        ...

    @property
    def width(self):
        return 1

    @property
    def nvectors(self):
        return 0

    @property
    def arg_names(self):
        return {}
    
    @property
    def _equation(self):
        return []
    
    def design_matrix(self, *args, **kwargs):
        """Build a 1D polynomial in x

        Parameters
        ----------
        {} : np.ndarray
            Vector to create polynomial of

        Returns
        -------
        X : np.ndarray
            Design matrix with shape (len(x), self.nvectors)
        """
        print((kwargs))
        if len(kwargs) < 1:
            raise ValueError("Cannot create design matrix without data.")

        x = np.ones_like(next(iter(kwargs.values())))
        return np.atleast_2d(x).T
