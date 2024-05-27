#!/usr/bin/env python
# coding: utf-8
# Use only for non-subsampled Gaussian mechanisms.

import numpy as np
from scipy import optimize
from scipy.special import erfc

EPS_TARGET = 0.1
DELTA_TARGET = 1e-5
N_CLASSES = 10
DELTA_ESTIMATE = 1e-5
delta_term = np.sqrt(2 * np.log(1.25 / DELTA_ESTIMATE))
SIGMA = np.sqrt(N_CLASSES) * delta_term / EPS_TARGET


# cf. Theorem 5,
# David M Sommer, Sebastian Meiser, and Esfandiar Mohammadi.
# Privacy loss classes: The central limit theorem in differential privacy.
# Proceedings on privacy enhancing technologies, 2019(2):245-269, 2019.
n = np.array(N_CLASSES, dtype=np.float64)
sigma = np.array(1 / SIGMA, dtype=np.float64)
mu = np.array(sigma**2 / 2, dtype=np.float64)

f = lambda eps: 0.5 * (
    erfc((eps - n * mu) / (np.sqrt(2 * n) * sigma))
    - np.exp(eps) * erfc((eps + n * mu) / (np.sqrt(2 * n) * sigma))
)

# for finding an upper_bound, eps is assumed to be bounded between 0 and 400.
upper_bound = optimize.bisect(lambda eps: f(eps) - DELTA_TARGET, 0, 400)
print(f"Upper-bound {upper_bound} for delta={f(upper_bound)}")
