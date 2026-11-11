# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

"""Independent SciPy least-squares and Wald fixtures. Run with /usr/bin/python3."""

import itertools
import json
from pathlib import Path

import numpy
from scipy import linalg, stats


def create_fixture(common, offset):
    row = numpy.arange(60, dtype=float)
    regimes = (row % 3).astype(int)
    predictors = numpy.column_stack((2 + numpy.sin(1.1 * row) + row / 30 + offset,
                                  -1 + numpy.cos(0.9 * row) - offset / 3)).astype(numpy.float32).astype(float)
    response = (7 + 0.15 * regimes + (0.5 + 0.08 * regimes) * (predictors[:, 0] - offset)
                - 0.4 * (predictors[:, 1] + offset / 3) + 0.6 * numpy.sin(row * 2.3)
                + 0.2 * numpy.cos(row * 0.3)).astype(numpy.float32).astype(float)
    mask = (row % 11 != 5).astype(int)
    used = mask != 0
    values = numpy.column_stack((numpy.ones(len(row)), predictors))
    columns = []
    offsets = []
    for column, shared in enumerate(common):
        offsets.append(len(columns))
        columns.extend([values[:, column]] if shared else
                       [values[:, column] * (regimes == regime) for regime in range(3)])
    design = numpy.column_stack(columns)[used]
    full_design = numpy.column_stack([values[:, column] * (regimes == regime)
                                   for column in range(3) for regime in range(3)])[used]
    observed = response[used]
    estimate = linalg.lstsq(design, observed, lapack_driver='gelsd')[0]
    full_estimate = linalg.lstsq(full_design, observed, lapack_driver='gelsd')[0]
    residual = observed - design @ estimate
    full_residual = observed - full_design @ full_estimate
    residual_squares = residual @ residual
    full_squares = full_residual @ full_residual
    inverse_design = linalg.pinv(design)
    covariance = inverse_design @ inverse_design.T * residual_squares / (len(observed) - len(columns))
    restriction_count = 9 - len(columns)
    chow = [(residual_squares - full_squares) / restriction_count / (full_squares / (len(observed) - 9))
            if restriction_count else None, restriction_count, len(observed) - 9]
    stability = []
    for selected in [[0], [1], [2], [0, 1, 2]]:
        contrasts = []
        for column in selected:
            if not common[column]:
                for regime in range(1, 3):
                    contrast = numpy.zeros(len(columns))
                    contrast[offsets[column] + regime] = 1
                    contrast[offsets[column]] = -1
                    contrasts.append(contrast)
        if not contrasts:
            stability.extend([None, 0, None])
            continue
        contrast = numpy.array(contrasts)
        difference = contrast @ estimate
        statistic = float(difference @ linalg.solve(contrast @ covariance @ contrast.T, difference))
        stability.extend([statistic, len(contrasts), stats.chi2.sf(statistic, len(contrasts))])
    fitted = [None] * len(row)
    for index, value in zip(numpy.flatnonzero(used), design @ estimate):
        fitted[index] = float(value)
    return dict(name=f'{common}-offset-{offset}', common=common, predictorCount=2, regimeCount=3,
                predictors=predictors.ravel().tolist(), response=response.tolist(), regimes=regimes.tolist(),
                mask=mask.tolist(), expected=dict(coefficients=estimate.tolist(), covariance=covariance.ravel().tolist(),
                                                 fitted=fitted, chow=chow, stability=stability))


fixtures = [create_fixture(list(common), 0) for common in itertools.product([False, True], repeat=3)]
fixtures += [create_fixture([False, False, True], 4096), create_fixture([True, False, True], 4096)]
Path(__file__).with_name('restricted-regime-regression-operation-fixtures.json').write_text(
    json.dumps(fixtures, indent=2, allow_nan=False) + '\n')
