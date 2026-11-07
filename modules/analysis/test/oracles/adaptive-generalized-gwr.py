# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent NumPy-distance + statsmodels weighted GLM composition, not mgwr parity.
Run with numpy and statsmodels installed; writes its sibling fixture deterministically.
Inputs are float32-rounded before float64 reference fitting. No GPU solver is mirrored.
"""
import json
from pathlib import Path
from importlib.metadata import version
import numpy as np
import statsmodels.api as sm

positions = np.array([[column, row] for row in range(4) for column in range(6)], dtype=np.float32)
predictors = np.array([((index * 7) % 13 - 6) / 6 for index in range(24)], dtype=np.float32)
mask = np.ones(24, dtype=bool)
mask[[3, 19]] = False
models = []
for family in ['poisson', 'logistic']:
    response = np.array([index % 4 if family == 'poisson' else int(index % 5 in [0, 2]) for index in range(24)], dtype=np.float32)
    offset = np.array([0.1 * (index % 3) if family == 'poisson' else 0 for index in range(24)], dtype=np.float32)
    for kernel in ['gaussian', 'bisquare']:
        for mode in ['fixed', 'adaptive']:
            coefficients = []
            fitted = []
            bandwidths = []
            for focus in range(24):
                distance = np.linalg.norm(positions.astype(float) - positions[focus], axis=1)
                bandwidth = 3.0 if mode == 'fixed' else np.sort(distance[mask])[15] * 1.00001
                bandwidths.append(float(bandwidth))
                squared = (distance / bandwidth) ** 2
                weights = np.exp(-0.5 * squared) if kernel == 'gaussian' else np.where(squared < 1, (1 - squared) ** 2, 0)
                included = mask & (weights > 0)
                result = sm.GLM(response[included], sm.add_constant(predictors[included]), family=sm.families.Poisson() if family == 'poisson' else sm.families.Binomial(), freq_weights=weights[included], offset=offset[included]).fit(tol=1e-11, maxiter=200)
                coefficients.extend(result.params.tolist() if mask[focus] else [None, None])
                fitted.append(float(result.predict([1, predictors[focus]], offset=offset[focus])[0]) if mask[focus] else None)
            models.append(dict(family=family, kernel=kernel, mode=mode, response=response.tolist(), offset=offset.tolist(), coefficients=coefficients, fitted=fitted, bandwidths=bandwidths))
fixture = dict(provenance=dict(reference='NumPy distance order statistic and statsmodels freq_weights GLM composition', versions={name: version(name) for name in ['numpy', 'statsmodels']}, rounding='float32 input, float64 reference', tolerance=0.003), positions=positions.flatten().tolist(), predictors=predictors.tolist(), mask=mask.astype(int).tolist(), models=models)
Path(__file__).with_name('fixtures').joinpath('adaptive-generalized-gwr.json').write_text(json.dumps(fixture, indent=2) + '\n')
