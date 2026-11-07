# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

"""Independent NumPy 2.2.6 / SciPy 1.15.3 fixtures. Run explicitly; production never calls Python."""
import json
from pathlib import Path
import numpy as np
import scipy
from scipy.signal import savgol_filter
from scipy.optimize import nnls, minimize

samples = np.array([1.3, 2.1, 1.8, 3.2, 2.7, 4.1, 5.3, 4.8, 6.4])
endmembers = np.array([[1.0, 0.3, 0.1], [0.2, 1.1, 0.4], [0.5, 0.2, 1.3], [0.8, 0.6, 0.3]])
spectra = np.array([[0.48, 0.79, 0.6, 0.65], [-0.1, 1.2, 0.4, 0.5], [1.4, 0.3, -0.2, 0.9]])
solutions = {}
for constraint in ['unconstrained', 'nonnegative', 'simplex']:
    coefficients = []
    residuals = []
    for spectrum in spectra:
        if constraint == 'unconstrained':
            abundance = np.linalg.lstsq(endmembers, spectrum, rcond=None)[0]
        elif constraint == 'nonnegative':
            abundance = nnls(endmembers, spectrum)[0]
        else:
            solution = minimize(lambda abundance: np.sum((endmembers @ abundance - spectrum) ** 2),
                np.ones(3) / 3, method='SLSQP', bounds=[(0, None)] * 3,
                constraints={'type': 'eq', 'fun': lambda abundance: abundance.sum() - 1},
                options={'ftol': 1e-13, 'maxiter': 1000})
            assert solution.success
            abundance = solution.x
        coefficients.extend(abundance.tolist())
        residuals.append(float(np.sum((endmembers @ abundance - spectrum) ** 2)))
    solutions[constraint] = {'abundances': coefficients, 'residual': residuals}
fixture = {'versions': {'numpy': np.__version__, 'scipy': scipy.__version__},
    'temporal': {'values': samples.tolist(), 'windowLength': 7, 'polynomialOrder': 2,
                 'expected': savgol_filter(samples, 7, 2, mode='interp').tolist()},
    'unmixing': {'values': spectra.flatten().tolist(), 'endmembers': endmembers.T.flatten().tolist(), 'solutions': solutions}}
Path(__file__).with_name('fixtures').joinpath('raster-remote-sensing.json').write_text(json.dumps(fixture, indent=2) + '\n')
