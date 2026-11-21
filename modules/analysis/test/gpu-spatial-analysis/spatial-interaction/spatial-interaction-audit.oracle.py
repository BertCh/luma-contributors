# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
# Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-interaction/spatial-interaction-audit.oracle.py
"""Independent Poisson profile oracle, using direct NumPy mass scaling (not log balancing).

mu = exp(beta*x) times a fitted global, origin, destination, or origin AND destination
intercept. IPF alternates origin then destination scaling. Deviance is
2 sum[y log(y/mu) - y + mu], with the y=0 term equal to 2*mu.
The fixture has duplicate cells, a structural zero, a missing cell and an empty group.
"""
import json
import numpy as np
from scipy.special import xlogy

ORIGINS = np.array([0, 0, 1, 1, 0, 1])
DESTINATIONS = np.array([0, 1, 0, 1, 0, 1])
FLOWS = np.array([10, 20, 30, 40, 5, 15], dtype=float)
COVARIATE = np.array([0, 1, 2, 0, 0, 1], dtype=float)


def fit(constraint, coefficient, budget=256):
    means = np.exp(coefficient * COVARIATE)
    origin_totals = np.bincount(ORIGINS, weights=FLOWS, minlength=3)
    destination_totals = np.bincount(DESTINATIONS, weights=FLOWS, minlength=3)
    error = 0.0
    iterations = 0
    if constraint == 'unconstrained':
        means *= FLOWS.sum() / means.sum()
    else:
        for iterations in range(1, budget + 1):
            for enabled, ids, totals in [
                (constraint != 'destination', ORIGINS, origin_totals),
                (constraint != 'origin', DESTINATIONS, destination_totals)
            ]:
                if enabled:
                    sums = np.bincount(ids, weights=means, minlength=3)
                    factors = np.divide(totals, sums, out=np.zeros(3), where=sums > 0)
                    means *= factors[ids]
            errors = []
            if constraint != 'destination':
                errors.extend(abs(np.bincount(ORIGINS, weights=means, minlength=3) - origin_totals) / np.maximum(1, origin_totals))
            if constraint != 'origin':
                errors.extend(abs(np.bincount(DESTINATIONS, weights=means, minlength=3) - destination_totals) / np.maximum(1, destination_totals))
            error = max(errors)
            if error <= 1e-5:
                break
    deviance = 2 * np.sum(xlogy(FLOWS, FLOWS / means) - FLOWS + means)
    return dict(predictions=[*means, 0, float('nan')],
                diagnostics=[deviance, error, iterations, int(error > 1e-5)],
                originResiduals=list(origin_totals - np.bincount(ORIGINS, weights=means, minlength=3)),
                destinationResiduals=list(destination_totals - np.bincount(DESTINATIONS, weights=means, minlength=3)))


for constraint in ['unconstrained', 'origin', 'destination', 'doubly']:
    candidates = [fit(constraint, coefficient) for coefficient in [-np.log(2), 0, -np.log(2)]]
    # np.argmin is stable: a tied grid chooses its first candidate.
    chosen = int(np.argmin([candidate['diagnostics'][0] for candidate in candidates]))
    selected = candidates[chosen]
    print(constraint, json.dumps(dict(**selected, chosen=chosen, candidates=[candidate['diagnostics'] for candidate in candidates])))

print('one-sweep', json.dumps(fit('doubly', -np.log(2), budget=1)))

zero_flows = np.array([10, 20, 30, 40, 5, 0], dtype=float)
zero_means = np.full(6, zero_flows.sum()/6)
print('observed-zero unconstrained predictions', list(zero_means), 'deviance', 2*np.sum(xlogy(zero_flows, zero_flows/zero_means)-zero_flows+zero_means))

# One golden-section refinement for an unconstrained 2x2 fixture. Direct Poisson
# profile: mu = sum(y)*softmax(beta*x); ties shrink the upper endpoint.
flows = np.array([10, 20, 30, 40], dtype=float)
covariate = np.array([0, 1, 2, 0], dtype=float)
ratio = (np.sqrt(5) - 1) / 2
lower, upper = -1.0, 1.0


def profile(coefficient):
    means = np.exp(coefficient * covariate)
    means *= flows.sum() / means.sum()
    return means, 2 * np.sum(xlogy(flows, flows / means) - flows + means)


first, second = lower + (1 - ratio) * (upper - lower), lower + ratio * (upper - lower)
if profile(first)[1] <= profile(second)[1]:
    upper = second
else:
    lower = first
coefficients = [lower + (1 - ratio) * (upper - lower), lower + ratio * (upper - lower)]
fits = [profile(coefficient) for coefficient in coefficients]
chosen = int(np.argmin([fit[1] for fit in fits]))
print('bounded-one-refinement', json.dumps(dict(coefficients=coefficients, predictions=list(fits[chosen][0]), summary=[coefficients[chosen], fits[chosen][1], 0, 0], candidates=[[fit[1], 0, 0, 0] for fit in fits], calibration=[lower, upper, upper-lower, 0])))
