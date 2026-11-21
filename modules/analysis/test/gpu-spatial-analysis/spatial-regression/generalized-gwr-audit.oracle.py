# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-regression/generalized-gwr-audit.oracle.py

Independent saturated two-category likelihood solutions, not a port of IRLS/Cholesky.
Each category's weighted sample mean is its fitted Poisson mean/Bernoulli probability.
The two category links determine intercept and slope exactly, including constant offsets.
Covariance and influence use the GLM score derivative and independent row variances.
Print literals for generalized-gwr-audit-fixtures.ts; tests never invoke Python.
"""
import json
from decimal import Decimal, localcontext
import numpy as np
from scipy.special import expit, logit

positions = np.array([0] * 6 + [2] * 6 + [1, 1], dtype=float)
predictors = np.array([-1] * 3 + [1] * 3 + [-1] * 3 + [1] * 3 + [-1, 1], dtype=float)
mask = np.array([True] * 12 + [False] * 2)
design = np.column_stack([np.ones(14), predictors])
tolerance = float(np.float32(1e-5))

# R1: independent saturated-category LOO means, then 50-digit Poisson likelihood ratios.
def large_count_scores():
    observed = np.array([10000, 10001, 10002, 10001, 10003, 10005, 10005, 10006, 10007, 10003, 10005, 10007])
    scores = []
    for bandwidth in [10, .5]:
        terms = []
        for focus in range(12):
            eligible = (np.arange(12) != focus) & (predictors[:12] == predictors[focus])
            weights = np.exp(-.5 * ((positions[:12] - positions[focus]) / bandwidth) ** 2)
            mean = np.average(observed[eligible], weights=weights[eligible])
            with localcontext() as context:
                context.prec = 50
                count = Decimal(int(observed[focus]))
                prediction = Decimal(str(mean))
                terms.append(2 * (count * (count / prediction).ln() - count + prediction))
        scores.append(float(sum(terms) / 12))
    return scores

print('R1 large-count LOO:', large_count_scores())
print('R3 exposure shift: beta0 -> beta0+14, beta1 and means unchanged (eta=X*beta+offset).')
print('R4 extrapolated query mean:', float(np.exp(.5547189562170504 - .013428224342895145 * 1000 + .125)))
print('R5 rank-one constant-predictor design: singular before any accepted update, iterationCount=0.')
print('R9 logistic query eta=100.25: float32 probability', np.float32(expit(100.25)), 'influence=0.')


def fit(family, kernel, adaptive=False, bandwidth=3, leave_one_out=False):
    observed = np.array(([0, 1, 2, 1, 3, 5, 2, 3, 4, 0, 2, 4] if family == 'poisson'
                         else [0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1]) + [0, 0], dtype=float)
    offsets = .25 + .125 * predictors if family == 'poisson' else .25 * predictors
    coefficients, fitted, errors, influence, iterations = [], [], [], [], []
    for focus in range(14):
        training = mask.copy()
        if leave_one_out:
            training[focus] = False
        distance = abs(positions - positions[focus])
        radius = np.sort(distance[training])[7] * 1.00001 if adaptive else bandwidth
        weights = np.exp(-.5 * (distance / radius) ** 2) if kernel == 'gaussian' else np.maximum(0, 1 - (distance / radius) ** 2) ** 2
        weights[~training] = 0
        category_means = np.array([np.average(observed[predictors == level], weights=weights[predictors == level]) for level in [-1, 1]])
        category_offsets = np.array([offsets[0], offsets[3]])
        category_links = np.log(category_means) if family == 'poisson' else logit(category_means)
        category_coefficients = category_links - category_offsets
        beta = np.array([np.mean(category_coefficients), np.diff(category_coefficients)[0] / 2])
        linear = design @ beta + offsets
        mean = np.exp(linear) if family == 'poisson' else expit(linear)
        variance = mean if family == 'poisson' else mean * (1 - mean)
        information = design.T @ np.diag(weights * variance) @ design
        inverse = np.linalg.inv(information)
        covariance = inverse @ design.T @ np.diag(weights ** 2 * variance) @ design @ inverse
        coefficients.append(beta.tolist())
        fitted.append(float(mean[focus]))
        errors.append(np.sqrt(np.diag(covariance)).tolist())
        influence.append(float(weights[focus] * variance[focus] * (design[focus] @ inverse @ design[focus])))
        # Independent scalar Newton recurrence for two saturated category links. Count the
        # documented relative coefficient criterion in the focal centered/scaled basis.
        links = category_offsets.copy()
        for step in range(1, 65):
            current_mean = np.exp(links) if family == 'poisson' else expit(links)
            current_variance = current_mean if family == 'poisson' else current_mean * (1 - current_mean)
            next_links = links + (category_means - current_mean) / current_variance
            def centered_coefficients(values):
                adjusted = values - category_offsets
                return np.array([adjusted[0 if predictors[focus] == -1 else 1], np.diff(adjusted)[0]])
            previous = centered_coefficients(links)
            change = np.max(abs(centered_coefficients(next_links) - previous) / np.maximum(1, abs(previous)))
            links = next_links
            if change <= tolerance:
                break
        iterations.append(step)
    prediction = np.array(fitted)[:12]
    if family == 'poisson':
        positive = observed[:12] > 0
        terms = prediction - observed[:12]
        terms[positive] += observed[:12][positive] * np.log(observed[:12][positive] / prediction[positive])
        deviance = 2 * sum(terms)
    else:
        deviance = -2 * sum(np.where(observed[:12] == 1, np.log(prediction), np.log1p(-prediction)))
    trace = sum(influence[:12])
    score = deviance / 12 if leave_one_out else deviance + 2 * trace + 2 * trace * (trace + 1) / (11 - trace)
    return dict(coefficients=coefficients, fitted=fitted, standardErrors=errors, influence=influence,
                iterationCount=iterations, score=float(score))


for family in ['poisson', 'logistic']:
    for kernel in ['gaussian', 'bisquare']:
        for adaptive in [False, True]:
            result = fit(family, kernel, adaptive)
            # Repeated rows share a fit at each of the three sites; all 14 outputs are
            # reconstructed from these six distinct category/site rows in the TS fixture.
            samples = [0, 3, 6, 9, 12, 13]
            compact = {key: [value[index] for index in samples] for key, value in result.items() if key != 'score'}
            compact['aicc'] = [fit(family, kernel, bandwidth=value)['score'] for value in [2.5, 3]]
            compact['loo'] = [fit(family, kernel, bandwidth=value, leave_one_out=True)['score'] for value in [2.5, 3]]
            print(f'{family}/{kernel}/{"adaptive" if adaptive else "fixed"}:')
            print(json.dumps(compact, indent=2))
