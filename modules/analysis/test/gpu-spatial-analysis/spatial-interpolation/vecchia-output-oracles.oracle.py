# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-interpolation/vecchia-output-oracles.oracle.py

Independent scalar Gaussian conditioning: C(h)=2 exp(-h), one predecessor per site.
No production code is imported. A scalar system has pivot ratio one.
"""
import math

values = [2, -1, 3]
errors = [0, 0.5, 0.25]
conditionals = []
terms = []
for row in range(3):
    cross = 2 * math.exp(-1)
    mean = 0 if row == 0 else cross * values[row - 1] / (2 + errors[row - 1])
    variance = 2 if row == 0 else 2 - cross * cross / (2 + errors[row - 1])
    conditionals.extend([mean, variance, 1, int(row > 0)])
    total_variance = variance + errors[row]
    terms.append(-0.5 * (math.log(2 * math.pi * total_variance) + (values[row] - mean)**2 / total_variance))
print('conditionals =', conditionals)
print('siteTerms =', terms)
print('logLikelihood =', sum(terms))

# Two zero-error predecessors: normalized covariance [[1,rho],[rho,1]] has
# elimination pivots 1 and 1-rho². Markov exponential covariance gives weights [0,rho].
rho = math.exp(-1)
print('two-predecessor conditionals =', [0, 2, 1, 0, 2*rho, 2*(1-rho*rho), 1, 1,
    -rho, 2*(1-rho*rho), 1-rho*rho, 2])
