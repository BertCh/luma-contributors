# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/geographical-detector/geographical-detector-audit.oracle.py

Wang factor detector, population variance and all nonempty strata (singleton
retained). Independent NumPy reductions and SciPy noncentral F survival.
"""
import numpy as np
from scipy.stats import ncf, f

values = np.array([1, 2, 2, 4, 8, 9, 12, 13, 17], dtype=float)
labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 3])
for shift, scale in [(0, 1), (4, 2)]:
    response = shift + scale * values
    groups = [response[labels == label] for label in np.unique(labels)]
    total = ((response - response.mean()) ** 2).sum()
    within = sum(((group - group.mean()) ** 2).sum() for group in groups)
    q = 1 - within / total
    numerator, denominator = len(groups) - 1, len(response) - len(groups)
    statistic = denominator / numerator * q / (1 - q)
    noncentrality = (sum(group.mean() ** 2 for group in groups)
                    - sum(np.sqrt(len(group)) * group.mean() for group in groups) ** 2 / len(response)) / (total / len(response))
    summary = [q, statistic, noncentrality, len(groups), len(response), numerator, denominator, total]
    packed = np.array(summary, dtype=np.float32)
    print('summary', [float(value) for value in summary])
    print('packed summary p-value', ncf.sf(float(packed[1]), numerator, denominator, float(packed[2])))

for value, numerator, denominator, noncentrality in [(0.1, 1.5, 40, 0), (3, 5, 40, 8), (1.25, 1, 7, 2), (25, 3, 50, 20), (5000, 2, 300, 10000)]:
    tail = ncf.sf(value, numerator, denominator, noncentrality) if noncentrality else f.sf(value, numerator, denominator)
    print('tail', [value, numerator, denominator, noncentrality, float(tail)])

# R5: compute F from mean squares, independently of a rounded q.
response = np.array([0, 0, 0, 1, 10000, 10000, 10000, 10001], dtype=float)
groups = [response[:4], response[4:]]
total = ((response - response.mean()) ** 2).sum()
within = sum(((group - group.mean()) ** 2).sum() for group in groups)
statistic = 6 * (total - within) / within
noncentrality = (sum(group.mean() ** 2 for group in groups) - sum(2 * group.mean() for group in groups) ** 2 / 8) / (total / 8)
print('R5', total, within, statistic, noncentrality, ncf.sf(statistic, 1, 6, noncentrality))
print('R5 packed tail', ncf.sf(float(np.float32(statistic)), 1, 6, float(np.float32(noncentrality))))
# A one-ulp SSD-reduction perturbation illustrates why the exact packed q=1 is
# not a valid device expectation after switching from 1-SSW/SST to direct SSB/SST.
packed_between = np.nextafter(np.float32(total - within), np.float32(-np.inf))
packed_q = packed_between / np.float32(total)
print('v3-1 q rounding', {'exact': (total - within) / total,
                        'packed between': float(packed_between), 'packed q': float(packed_q),
                        'absolute tolerance': 2. ** -22})

groups = [np.array([-1, 1, -1, 1], dtype=float), np.array([-1, 1, -1, 1], dtype=float) + 2 ** -13]
response = np.concatenate(groups)
total = ((response - response.mean()) ** 2).sum()
within = sum(((group - group.mean()) ** 2).sum() for group in groups)
between = sum(len(group) * (group.mean() - response.mean()) ** 2 for group in groups)
statistic = 6 * between / within
noncentrality = (sum(group.mean() ** 2 for group in groups) - sum(2 * group.mean() for group in groups) ** 2 / 8) / (total / 8)
print('review 2 R2', between / total, statistic, noncentrality, ncf.sf(statistic, 1, 6, noncentrality))

# One dyadic two-row stratum and six symmetric singleton strata. Derive the
# SSDs directly from groups rather than from q or the detector implementation.
groups = [np.array([0., 2. ** -50])] + [np.array([value]) for value in [8192., -8192.] * 3]
response = np.concatenate(groups)
within = sum(((group - group.mean()) ** 2).sum() for group in groups)
between = sum(len(group) * (group.mean() - response.mean()) ** 2 for group in groups)
statistic = (between / 6) / within
total = ((response - response.mean()) ** 2).sum()
noncentrality = (sum(group.mean() ** 2 for group in groups) - sum(np.sqrt(len(group)) * group.mean() for group in groups) ** 2 / 8) / (total / 8)
assert within == 2. ** -101 and between == 6 * 2. ** 26 and statistic == 2. ** 127
print('review 2 R3', {'within': within, 'between': between, 'F': statistic, 'lambda': noncentrality, 'p': ncf.sf(statistic, 6, 1, noncentrality)})
