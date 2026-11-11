# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

"""Independent oracle: SciPy BFGS with observed Hessian; NumPy dense diagnostics.

Run with /usr/bin/python3 from any directory. Inputs are rounded before fitting.
"""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr
from scipy.stats import norm, chi2

random = np.random.default_rng(20261111)
predictors = random.normal(size=(96, 2)).astype(np.float32).astype(float)
response = (random.normal(size=96) < 0.3 + predictors @ [0.7, -0.5]).astype(float)
mask = np.ones(96, dtype=int)
mask[[4, 19, 71]] = 0
offsets = [0]
neighbors = []
weights = []
matrix = np.zeros((96, 96))
for row in range(96):
    for neighbor, weight in sorted([((row + 1) % 96, 0.25), ((row + 7) % 96, 0.75)]):
        neighbors.append(neighbor)
        weights.append(weight)
        matrix[row, neighbor] = weight
    offsets.append(len(neighbors))
selected = mask != 0
means = predictors[selected].mean(axis=0)
design = np.column_stack([np.ones(sum(selected)), predictors[selected] - means])
observed = response[selected]
sides = 2 * observed - 1


def objective(estimate):
    signed = sides * (design @ estimate)
    ratio = np.exp(norm.logpdf(signed) - log_ndtr(signed))
    return -sum(log_ndtr(signed)), -(design.T @ (sides * ratio))


fit = minimize(objective, np.zeros(3), jac=True, method="BFGS", options={"gtol": 1e-10})
assert np.linalg.norm(objective(fit.x)[1]) < 1e-6
linear = design @ fit.x
probabilities = norm.cdf(linear).clip(1e-7, 1 - 1e-7)
densities = norm.pdf(linear)
information = design.T @ ((densities**2 / (norm.cdf(linear) * norm.cdf(-linear)))[:, None] * design)
transform = np.eye(3)
transform[0, 1:] = -means
coefficients = transform @ fit.x
errors = np.sqrt(np.diag(transform @ np.linalg.inv(information) @ transform.T))
matrix = matrix[np.ix_(selected, selected)]
variance = probabilities * (1 - probabilities)
naive = observed - probabilities
generalized = densities * naive / variance
standardized = naive / np.sqrt(variance)
trace = np.trace(matrix.T @ matrix + matrix @ matrix)
variance_matrix = np.diag(variance)
variance_trace = np.trace(matrix.T @ variance_matrix @ matrix @ variance_matrix + matrix @ variance_matrix @ matrix @ variance_matrix)
mean_information = np.mean(densities**2 / variance)
pinkse = (generalized @ matrix @ generalized / mean_information)**2 / trace
kelejian = naive @ matrix @ naive / np.sqrt(variance_trace)
slade = (standardized @ matrix @ standardized)**2 / trace
fitted = [None] * 96
for row, value in zip(np.flatnonzero(selected), probabilities):
    fitted[row] = value
fixture = {
    "oracle": "SciPy BFGS on centered design; dense NumPy expected information and diagnostics",
    "predictors": predictors.ravel().tolist(), "response": response.tolist(), "mask": mask.tolist(),
    "offsets": offsets, "neighbors": neighbors, "weights": weights,
    "expected": {
        "coefficients": coefficients.tolist(), "standardErrors": errors.tolist(),
        "fitted": fitted, "marginalEffects": (np.mean(densities) * coefficients[1:]).tolist(),
        "logLikelihood": -fit.fun,
        "tests": [pinkse, chi2.sf(pinkse, 1), kelejian, chi2.sf(kelejian**2, 1), slade, chi2.sf(slade, 1)],
    },
}
Path(__file__).with_suffix(".json").write_text(json.dumps(fixture, indent=2, allow_nan=False) + "\n")
