import math
import unittest
import numpy as np
from scipy.stats import bernoulli, expon, uniform

from qr_sampler import QuasiRandomParameterSampler, _phi_d

class TestPhi(unittest.TestCase):
    def test_phi(self):
        phis = [1.6180339887498948482, 1.32471795724474602596, 1.22074408460575947536]
        for d in range(1, len(phis) + 1):
            self.assertTrue(abs(_phi_d(d) - phis[d - 1]) < 1e-15)

class TestParamSampler(unittest.TestCase):
    def test_param_sampler(self):
        param_distributions = {"kernel": ["rbf", "linear"], "C": uniform(0, 1)}
        sampler = QuasiRandomParameterSampler(
            param_distributions=param_distributions, n_iter=10, random_state=0
        )
        samples = [x for x in sampler]
        self.assertTrue(len(samples) == 10)
        for sample in samples:
            self.assertTrue(sample["kernel"] in ["rbf", "linear"])
            self.assertTrue(0 <= sample["C"] <= 1)

        # test that repeated calls yield identical parameters
        param_distributions = {"C": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
        sampler = QuasiRandomParameterSampler(
            param_distributions=param_distributions, n_iter=3, random_state=0
        )
        self.assertTrue([x for x in sampler] == [x for x in sampler])

        param_distributions = {"C": uniform(0, 1)}
        sampler = QuasiRandomParameterSampler(
            param_distributions=param_distributions, n_iter=10, random_state=0
        )
        self.assertTrue([x for x in sampler] == [x for x in sampler])
