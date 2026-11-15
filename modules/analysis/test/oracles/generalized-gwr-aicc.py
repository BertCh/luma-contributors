# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent NumPy working-response IRLS and mgwr GLM diagnostics/search conventions.

Run: /usr/bin/python3 modules/analysis/test/oracles/generalized-gwr-aicc.py
No repository solver is used. Fits use numpy.linalg.lstsq on square-root weighted designs;
influence uses numpy.linalg.pinv. Inputs are rounded once to float32, then fitted in float64.
AICc: mgwr/diagnostics.py get_AICc and get_AIC (Poisson/Binomial).
Search: mgwr/search.py golden_section, delta .38197, ties to lower interior;
Fixed-distance history follows golden section. Adaptive expectations exhaust every integer in the
supplied bounds independently of GPU search/stopping logic. No final 2-decimal rounding.
"""
import json
from pathlib import Path
import numpy as np
from scipy.special import expit, xlogy

positions = np.array([[column, row] for row in range(6) for column in range(8)], dtype=np.float32).astype(float)
predictors = np.array([((index * 7) % 17 - 8) / 8 for index in range(48)], dtype=np.float32).astype(float)
mask = np.ones(48, dtype=bool)
mask[[0, 3, 19, 47]] = False
models = []

def fit(family, kernel, mode, response, offsets, bandwidth):
    design = np.column_stack([np.ones(48), predictors])
    coefficients = np.full((48, 2), np.nan)
    fitted = np.full(48, np.nan)
    influence = np.full(48, np.nan)
    for focus in range(48):
        distances = np.linalg.norm(positions - positions[focus], axis=1)
        radius = bandwidth if mode == 'fixed' else np.sort(distances[mask])[int(bandwidth) - 1] * 1.00001
        weights = np.exp(-.5 * (distances / radius) ** 2) if kernel == 'gaussian' else np.maximum(0, 1 - (distances / radius) ** 2) ** 2
        selected = mask & (weights > 0)
        local_design = design[selected]
        local_response = response[selected]
        local_offsets = offsets[selected]
        local_weights = weights[selected]
        # A working-response weighted least-squares IRLS, independently of the GPU score solver.
        linear = np.log(local_response + .5) if family == 'poisson' else np.where(local_response == 1, 1., -1.)
        previous = np.zeros(2)
        for iteration in range(300):
            mean = np.exp(linear) if family == 'poisson' else expit(linear)
            variance = mean if family == 'poisson' else mean * (1 - mean)
            working = linear + (local_response - mean) / variance - local_offsets
            roots = np.sqrt(local_weights * variance)
            estimate = np.linalg.lstsq(local_design * roots[:, None], working * roots, rcond=None)[0]
            linear = local_design @ estimate + local_offsets
            if np.max(np.abs(estimate - previous)) < 1e-11:
                break
            previous = estimate
        else:
            raise RuntimeError('oracle did not converge')
        coefficients[focus] = estimate
        eta = design[focus] @ estimate + offsets[focus]
        fitted[focus] = np.exp(eta) if family == 'poisson' else expit(eta)
        variance = np.exp(linear) if family == 'poisson' else expit(linear) * expit(-linear)
        information = local_design.T @ ((weights[selected] * variance)[:, None] * local_design)
        focus_variance = fitted[focus] if family == 'poisson' else fitted[focus] * (1 - fitted[focus])
        influence[focus] = design[focus] @ np.linalg.pinv(information) @ design[focus] * focus_variance * weights[focus] if mask[focus] else 0
    observed = response[mask]
    mean = fitted[mask]
    deviance = np.sum(2 * (xlogy(observed, observed / mean) - observed + mean)) if family == 'poisson' else -2 * np.sum(xlogy(observed, mean) + xlogy(1 - observed, 1 - mean))
    trace = np.sum(influence[mask])
    count = int(np.sum(mask))
    score = deviance + 2 * trace + 2 * trace * (trace + 1) / (count - trace - 1)
    return dict(score=float(score), trace=float(trace), deviance=float(deviance), coefficients=coefficients.flatten().tolist(), fitted=fitted.tolist())

for family in ['poisson', 'logistic']:
    response = np.array([int((index * 11) % 7 < 3) if family == 'logistic' else (index * 7 + index // 8) % 5 for index in range(48)], dtype=float)
    offsets = np.array([.1 * (index % 3) if family == 'poisson' else 0 for index in range(48)], dtype=np.float32).astype(float)
    for kernel in ['gaussian', 'bisquare']:
        for mode in ['fixed', 'adaptive']:
            lower, upper = (2., 8.) if mode == 'fixed' else (28., 44.)
            bounds = [lower, upper, .0001]
            if mode == 'adaptive':
                # An exhaustive objective oracle, deliberately independent of golden-section control.
                candidates = []
                best = None
                for bandwidth in range(int(lower), int(upper) + 1):
                    result = fit(family, kernel, mode, response, offsets, bandwidth)
                    candidates.append(dict(bandwidth=bandwidth, **result))
                    if best is None or result['score'] < best['score']:
                        best = dict(bandwidth=bandwidth, **result)
                models.append(dict(family=family, kernel=kernel, mode=mode, response=response.tolist(), offset=offsets.tolist(), bounds=bounds, history=[], candidates=candidates, searchReference='exhaustive-integer', iterationCount=None, converged=None, **best))
                continue
            first = lower + .38197 * (upper - lower)
            second = upper - .38197 * (upper - lower)
            history = []
            best = None
            converged = False
            for iteration in range(12):
                first_fit = fit(family, kernel, mode, response, offsets, first)
                second_fit = fit(family, kernel, mode, response, offsets, second)
                history.extend([dict(bandwidth=first, score=first_fit['score']), dict(bandwidth=second, score=second_fit['score'])])
                choose_first = first_fit['score'] <= second_fit['score']
                winner = dict(bandwidth=first if choose_first else second, **(first_fit if choose_first else second_fit))
                if best is None or winner['score'] < best['score']:
                    best = winner
                converged = abs(first_fit['score'] - second_fit['score']) <= bounds[2]
                if choose_first:
                    upper = second
                    second = first
                    first = lower + .38197 * (upper - lower)
                else:
                    lower = first
                    first = second
                    second = upper - .38197 * (upper - lower)
                converged = converged or first == second or first <= lower or second >= upper
                if converged:
                    break
            models.append(dict(family=family, kernel=kernel, mode=mode, response=response.tolist(), offset=offsets.tolist(), bounds=bounds, history=history, candidates=[], searchReference='golden-section', iterationCount=iteration + 1, converged=int(converged), **best))

fixture = dict(provenance=dict(reference='NumPy working-response IRLS, least-squares QR/SVD, mgwr GLM AICc; exhaustive adaptive integer oracle and fixed golden-section history', numpy=np.__version__, inputRounding='float32', fitPrecision='float64', maximumIterations=12), positions=positions.flatten().tolist(), predictors=predictors.tolist(), mask=mask.astype(int).tolist(), models=models)
Path(__file__).with_name('fixtures').joinpath('generalized-gwr-aicc.json').write_text(json.dumps(fixture, indent=2, allow_nan=False) + '\n')
print('Generated', len(models), 'models; iterations:', [model['iterationCount'] for model in models])
