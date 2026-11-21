# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/segregation/segregation-definition.oracle.py

Reardon/Firebaugh multigroup H and D; Duncan/Duncan group-vs-rest D;
Lieberson isolation/exposure; Atkinson b=0.3. NumPy dense linear algebra
constructs environments, independently of the contributor's CSR walks and
shader reductions. H is explicitly the sum of defined local t_i(E-E_i)/(TE)
terms, not 1 minus an entropy average when populated islands are missing.
The environment form retains original t_i, X_g, T; the smoothed-population
form replaces the population table (without PySAL's integer rounding).
Empty environments have NaN local outputs and contribute zero to global sums;
undefined denominators give NaN. All entropy logarithms are natural.
"""
import json
import numpy as np
from scipy.special import entr

COUNTS = np.array([[8, 2, 0], [2, 8, 0], [5, 5, 0], [0, 0, 0],
                   [1, 3, 6], [4, 1, 5], [0, 4, 6], [3, 3, 4]], dtype=float)
WEIGHTS = np.zeros((8, 8))
WEIGHTS[0, 1] = WEIGHTS[1, 0] = 1
WEIGHTS[2, 4] = WEIGHTS[2, 5] = 0.5
WEIGHTS[3, 2] = 1
WEIGHTS[4, 5] = 2
WEIGHTS[5, 4], WEIGHTS[5, 6] = 0.25, 0.75
WEIGHTS[6, 5] = 1
# Row 7 is a populated island. selfWeight is zero; self slots are excluded.


def compute(counts, environment):
    totals = counts.sum(axis=1)
    population = totals.sum()
    group_totals = counts.sum(axis=0)
    proportions = group_totals / population
    diversity = entr(proportions).sum()
    environment_totals = environment.sum(axis=1)
    valid = environment_totals > 0
    composition = np.full(environment.shape, np.nan)
    composition[valid] = environment[valid] / environment_totals[valid, None]
    local_entropy = entr(composition).sum(axis=1)
    local_theil = totals * (diversity - local_entropy) / (diversity * population)
    local_dissimilarity = totals[:, None] * abs(composition - proportions) / (2 * population * proportions * (1 - proportions))
    interaction = (counts[valid].T @ composition[valid]) / group_totals[:, None]
    dissimilarity = np.nansum(local_dissimilarity, axis=0)
    multi_dissimilarity = np.nansum(totals[:, None] * abs(composition - proportions)) / (2 * population * sum(proportions * (1 - proportions)))
    atkinson = 1 - proportions / (1 - proportions) * (
        np.nansum(totals[:, None] * (1 - composition) ** 0.7 * composition ** 0.3, axis=0) / group_totals
    ) ** (1 / 0.7)
    indices = np.concatenate(([np.nansum(local_theil), multi_dissimilarity, diversity],
                              dissimilarity, np.diag(interaction), atkinson, interaction.ravel()))
    return dict(indices=indices, localEnvironment=composition.ravel(), localEntropy=local_entropy,
                localDissimilarity=local_dissimilarity.ravel(), localTheil=local_theil)


def serialize(value):
    if isinstance(value, dict):
        return {key: serialize(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [serialize(item) for item in value.tolist()]
    if isinstance(value, list):
        return [serialize(item) for item in value]
    return None if not np.isfinite(value) else value


if __name__ == "__main__" and "--review" in __import__("sys").argv:
    # /usr/bin/python3 -B modules/analysis/test/gpu-spatial-analysis/segregation/segregation-definition.oracle.py --review
    counts = np.array([[10., 0.], [0., 10.]])
    for diagonal in [False, True]:
        weights = np.ones((2, 2))
        # The explicit diagonal is discarded; selfWeight supplies it exactly once.
        np.fill_diagonal(weights, 0)
        environment = (np.eye(2) + weights) @ counts
        result = compute(counts, environment)
        print("R2/R8", diagonal, "H,D", result["indices"][:2].tolist())
    for scale in [1, 1e-32]:
        counts = np.full((2, 2), scale)
        print("R7", scale, "Atkinson", compute(counts, counts)["indices"][7:9].tolist())
elif __name__ == "__main__":
    original = compute(COUNTS, COUNTS)
    environment = WEIGHTS @ COUNTS
    results = {}
    for form in ["environment", "smoothed-population"]:
        spatial = compute(COUNTS if form == "environment" else environment, environment)
        results[form] = {key: np.concatenate((original[key], spatial[key])) for key in original}
    print(json.dumps(dict(license="MIT", copyright="Copyright (c) vis.gl contributors",
                          fixtures=serialize(results)), indent=2, allow_nan=False))
