import numpy as np

from sklearn.utils import check_random_state
from skllearn.model_selection import ParameterGrid

def _phi_d(d):
    """Generate the seed of the d-dimensional low-discrepance R-sequence.
    See http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
    """
    x = 1 + math.log(2) / d
    while True:
        u = x ** (-d)
        x1 = x + (u * (1 + x) - x) / (d + 1 - u)
        if x1 >= x:
            break
        x = x1
    return x

class QuasiRandomParameterSampler:
    """Emulates sklearn.model_selection.ParameterSampler, but uses
    quasi-random (low discrepancy) sampling to explore more systematically.
    Parameters
    ----------
    param_distributions : dict
        Dictionary with parameters names (`str`) as keys and distributions
        or lists of parameters to try. Distributions must provide a ``rvs``
        method for sampling (such as those from scipy.stats.distributions).
        If a list is given, it is sampled uniformly.
        If a list of dicts is given, first a dict is sampled uniformly, and
        then a parameter is sampled using that dict as above.
    n_iter : int
        Number of parameter settings that are produced.
    random_state : int, RandomState instance or None, default=None
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.
        Pass an int for reproducible output across multiple
        function calls.
        See :term:`Glossary <random_state>`.
    Returns
    -------
    params : dict of str to any
        **Yields** dictionaries mapping each estimator parameter to
        as sampled value.
    """

    def __init__(self, param_distributions, n_iter, *, random_state=None):
        if not isinstance(param_distributions, (Mapping, Iterable)):
            raise TypeError(
                "Parameter distribution is not a dict or a list,"
                f" got: {param_distributions!r} of type "
                f"{type(param_distributions).__name__}"
            )

        if isinstance(param_distributions, Mapping):
            # wrap dictionary in a singleton list to support either dict
            # or list of dicts
            param_distributions = [param_distributions]

        for dist in param_distributions:
            if not isinstance(dist, dict):
                raise TypeError(
                    "Parameter distribution is not a dict ({!r})".format(dist)
                )
            for key in dist:
                if not isinstance(dist[key], Iterable) and not hasattr(
                    dist[key], "rvs"
                ):
                    raise TypeError(
                        f"Parameter grid for parameter {key!r} is not iterable "
                        f"or a distribution (value={dist[key]})"
                    )

        self.n_iter = n_iter
        self.random_state = random_state
        self.param_distributions = param_distributions
        self._dim = len(self.param_distributions) + max(len(d) for d in self.param_distributions)
        phi = _phi_d(self.dim)
        self._alpha = phi ** (-np.arange(1, len(dists) + 1))

    def _is_all_lists(self):
        return all(
            all(not hasattr(v, "rvs") for v in dist.values())
            for dist in self.param_distributions
        )

    def __iter__(self):
        rng = check_random_state(self.random_state)
        if self.random_state == 0:
            #The median is a reasonable starting point.
            quantiles = np.full(self._dim, 0.5)
        else:
            quantiles = rng.random(self._dim)
        for _ in range(self.n_iter):
            dist = rng.choice(self.param_distributions)
            # Always sort the keys of a dictionary, for reproducibility
            items = sorted(dist.items())
            params = dict()
            for i, (k, v) in enumerate(items):
                if hasattr(v, "ppf"):
                    params[k] = v.ppf(quantiles[i + 1])
                elif hasattr(v, "rvs"):
                    params[k] = v.rvs(random_state=rng)
                else:
                    # Guard against potential rounding errors (probably unneccessary).
                    params[k] = v[min(len(v) - 1, int(quantiles[i + 1] * len(v)))]
            yield params
            quantiles += self._alpha
            quantiles %= 1
    def __len__(self):
        """Number of points that will be sampled."""
        if self._is_all_lists():
            grid_size = len(ParameterGrid(self.param_distributions))
            return min(self.n_iter, grid_size)
        else:
            return self.n_iter
