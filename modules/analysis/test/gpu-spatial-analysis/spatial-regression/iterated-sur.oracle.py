# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

"""Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-regression/iterated-sur.oracle.py

Prints literal fixtures. Independent dense NumPy block GLS from the definition;
divisor n, common complete cases, and no residual degrees-of-freedom correction.
Spatial first-stage lambda literals originate in the adjacent independent SciPy
spatial moment oracle, not in the TypeScript implementation.
"""

import json
from pathlib import Path
import numpy


def solve(predictors, responses, mask, budget, spatial=None):
    equations, rows, columns = predictors.shape
    width = columns + 1
    used = numpy.flatnonzero(mask)
    designs = numpy.concatenate((numpy.ones((equations, len(used), 1)), predictors[:, used]), axis=2)
    observed = responses[:, used]
    previous = numpy.array([numpy.linalg.lstsq(design, response, rcond=None)[0]
                         for design, response in zip(designs, observed)])
    residual = observed - numpy.einsum('eri,ei->er', designs, previous)
    structural_designs = designs.copy()
    if spatial is not None:
        dense, parameters = spatial
        # Parameters are estimated once. Filter intercept as well as the other columns.
        for equation in range(equations):
            filtering = numpy.eye(rows) - parameters[equation] * dense
            designs[equation] = filtering @ designs[equation]
            observed[equation] = filtering @ observed[equation]
            residual[equation] = filtering @ residual[equation]
        previous = numpy.array([numpy.linalg.lstsq(design, response, rcond=None)[0]
                             for design, response in zip(designs, observed)])
    block = numpy.zeros((equations * len(used), equations * width))
    for equation in range(equations):
        block[equation * len(used):(equation + 1) * len(used), equation * width:(equation + 1) * width] = designs[equation]
    for count in range(1, budget + 1):
        sigma = residual @ residual.T / len(used)
        precision = numpy.kron(numpy.linalg.inv(sigma), numpy.eye(len(used)))
        information = block.T @ precision @ block
        covariance = numpy.linalg.inv(information)
        coefficients = numpy.linalg.solve(information, block.T @ precision @ observed.ravel()).reshape(equations, width)
        # Spatial implementation solves with an intercept of one then restores the
        # structural intercept. Apply the same stated stopping convention in those units.
        comparison = coefficients.copy()
        old = previous.copy()
        if spatial is not None:
            comparison[:, 0] *= 1 - parameters
            old[:, 0] *= 1 - parameters
        change = numpy.max(numpy.abs(comparison - old) / numpy.maximum(1, numpy.abs(old)))
        residual = observed - numpy.einsum('eri,ei->er', designs, coefficients)
        if (spatial is None or count > 1) and change < 1e-5:
            status = 0
            break
        previous = coefficients
        status = 4
    final_residual = responses[:, used] - numpy.einsum('eri,ei->er', structural_designs, coefficients)
    packed_residual = [[None] * rows for _ in range(equations)]
    for equation in range(equations):
        for position, row in enumerate(used):
            packed_residual[equation][row] = float(final_residual[equation, position])
    return dict(coefficients=coefficients.ravel().tolist(), covariance=covariance.ravel().tolist(),
                residualCovariance=sigma.ravel().tolist(), residuals=sum(packed_residual, []),
                iterationCount=[count], status=[status])


row = numpy.arange(40)
predictors = numpy.array([numpy.column_stack((((row * (7 + equation) + equation * 3) % 17 - 8) / 4,
                                       ((row * (3 + equation) + equation) % 13 - 6) / 4))
                       for equation in range(3)], dtype=numpy.float32).astype(float)
# Repeated predictor values, different responses, and a common excluded row.
predictors[:, 1] = predictors[:, 0]
common = numpy.sin(row * 1.7) + 0.5 * numpy.cos(row * 0.31)
responses = numpy.array([2 + equation + predictors[equation] @ [0.7 + 0.2 * equation, -0.4]
                      + common + 0.3 * numpy.cos(row * (0.7 + 0.2 * equation))
                      for equation in range(3)], dtype=numpy.float32).astype(float)
mask = numpy.ones(40, dtype=int)
mask[7] = 0
fixtures = []
for scene in ['ordinary', 'shifted', 'spatial']:
    if scene == 'spatial':
        source = json.loads(Path(__file__).with_name('seemingly-unrelated-regression-spatial-operation-fixtures.json').read_text())
        values = numpy.array(source['predictors']).reshape(2, 40, 2)
        observed = numpy.array(source['responses']).reshape(2, 40)
        selected = numpy.ones(40, dtype=int)
        weights = source['weights']
        dense = numpy.zeros((40, 40))
        for position in range(40):
            for slot in range(weights['offsets'][position], weights['offsets'][position + 1]):
                dense[position, weights['neighbors'][slot]] = weights['weights'][slot]
        spatial = dense, numpy.array(source['expected']['parameters'])
    else:
        values = predictors.copy()
        observed = responses.copy()
        selected = mask
        spatial = None
        if scene == 'shifted':
            values = values * 2 + 8
            observed = (observed * 2 + 16).astype(numpy.float32).astype(float)
    expectations = [dict(maximumIterations=budget, expected=solve(values, observed.copy(), selected, budget, spatial))
                    for budget in [1, 2, 100]]
    fixtures.append(dict(name=scene, equationCount=len(values), predictorCount=2, rowCount=40,
                         predictors=values.ravel().tolist(), responses=observed.ravel().tolist(),
                         mask=selected.tolist(), weights=source['weights'] if spatial else None,
                         parameters=source['expected']['parameters'] if spatial else None, rounds=expectations))
print(json.dumps(fixtures, indent=2, allow_nan=False))
