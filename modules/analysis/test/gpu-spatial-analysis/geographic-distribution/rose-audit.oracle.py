# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
# Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/geographic-distribution/rose-audit.oracle.py
"""Rose moments: weighted descriptive mean/R/variance, unweighted n*R^2
Rayleigh statistic; half-open equal-width bins; axial vectors doubled.
Rows 5,6,7 are respectively masked, zero-length and out-of-range group.
"""
import json
import numpy as np

vectors = np.array([[1, 0], [1, 0], [0, 1], [-1, 0], [0, 1]], dtype=float)
weights = np.array([1, 2, 4, 8, 3])
groups = np.array([0, 0, 0, 0, 1])
for axial in [False, True]:
    for compass in [False, True]:
        summary, bins = [], []
        for group in range(3):
            unit = vectors[groups == group]
            mass = weights[groups == group]
            if len(unit) == 0:
                summary.extend([0, *([float('nan')] * 4)])
                bins.extend([0, 0, 0, 0])
                continue
            angles = np.mod(np.arctan2(unit[:, 0], unit[:, 1]) if compass else np.arctan2(unit[:, 1], unit[:, 0]), 2*np.pi)
            period = np.pi if axial else 2*np.pi
            # Exact cardinal unit vectors avoid transcendental pi residuals at a tie.
            moments = (unit[:, 1] + 1j*unit[:, 0]) if compass else (unit[:, 0] + 1j*unit[:, 1])
            if axial:
                moments = moments**2
            weighted = np.average(moments, weights=mass)
            mean = np.mod(np.angle(weighted), 2*np.pi) / (2 if axial else 1)
            summary.extend([len(unit), mean, abs(weighted), 1-abs(weighted), abs(moments.sum())**2/len(unit)])
            histogram, _ = np.histogram(np.mod(angles, period), bins=np.arange(5)*period/4, weights=mass)
            bins.extend(histogram.tolist())
        print(('axial' if axial else 'directional') + ('/compass' if compass else '/cartesian'), json.dumps(dict(summary=summary, bins=bins)))
