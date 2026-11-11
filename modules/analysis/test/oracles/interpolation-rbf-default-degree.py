# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""SciPy RBFInterpolator references for GPULocalRBF at its DEFAULT polynomial degree.

Math correctness review 2026-11-10, interpolation-density-12: every other SciPy fixture of
GPULocalRBF passes degree=2, so the default degree of each kernel had no outside check.

Run from the repository root with the pinned set of requirements.txt (NumPy and SciPy only):
  python modules/analysis/test/oracles/interpolation-rbf-default-degree.py
The first run (2026-11-10, NumPy 2.0.2 and SciPy 1.13.1) differs from the pinned run (NumPy 2.5.4,
SciPy 1.16.3) by at most 3.9e-14 relative, the rounding of the 32-site dense solve.

Inputs are the `radial` block of interpolation-advanced-reference.json (60 sites, 4 queries, the
32 nearest sites of each query, per-site smoothing), so nothing here is random. Each query is one
direct scipy.interpolate.RBFInterpolator fit on its 32 sites with epsilon 0.9. The degree passed
to SciPy is the default GPULocalRBF documents for the kernel: 1 for thin-plate-spline and cubic,
2 for quintic, 0 for linear and multiquadric (these four equal SciPy's own default), and -1 (no
polynomial) for gaussian, inverse-multiquadric and inverse-quadratic, where SciPy's default is 0.
"""
import json
from pathlib import Path

import numpy as np
import scipy
from scipy.interpolate import RBFInterpolator

HERE = Path(__file__).parent
SOURCE = HERE.parent / 'gpu-spatial-analysis/spatial-interpolation/interpolation-advanced-reference.json'
radial = json.loads(SOURCE.read_text())['radial']
positions = np.array(radial['positions'], dtype=float)
values = np.array(radial['values'], dtype=float)
queries = np.array(radial['queries'], dtype=float)
smoothing = np.array(radial['smoothing'], dtype=float)
neighborhoods = np.array(radial['neighborhoods'], dtype=int)

DEFAULT_DEGREES = {
    'thin-plate-spline': 1,
    'cubic': 1,
    'quintic': 2,
    'linear': 0,
    'multiquadric': 0,
    'gaussian': -1,
    'inverse-multiquadric': -1,
    'inverse-quadratic': -1,
}

expected = {}
for kernel, degree in DEFAULT_DEGREES.items():
    expected[kernel] = [
        float(
            RBFInterpolator(
                positions[indices],
                values[indices],
                kernel=kernel.replace('-', '_'),
                degree=degree,
                epsilon=0.9,
                smoothing=smoothing[indices],
            )(query[None, :])[0]
        )
        for query, indices in zip(queries, neighborhoods)
    ]

result = {
    'versions': {'numpy': np.__version__, 'scipy': scipy.__version__},
    'source': 'interpolation-advanced-reference.json radial block',
    'epsilon': 0.9,
    'degrees': DEFAULT_DEGREES,
    'expected': expected,
}
output = HERE / 'fixtures/interpolation-rbf-default-degree.json'
output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
print(json.dumps({'fixture': str(output), 'versions': result['versions']}))
