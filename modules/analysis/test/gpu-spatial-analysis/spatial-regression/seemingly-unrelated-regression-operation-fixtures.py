# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

"""Independent SciPy SVD fits of whitened block systems; run with /usr/bin/python3."""

import json
from pathlib import Path

import numpy
from scipy import linalg, optimize


def create_fixture(equation_count, supplied=False):
    row = numpy.arange(40, dtype=float)
    predictors = numpy.array([numpy.column_stack((
        2 + numpy.sin(row * (0.7 + equation / 5)) + row / 30,
        -1 + numpy.cos(row * (1.1 + equation / 6)) + (row % 3) / 3
    )) for equation in range(equation_count)]).astype(numpy.float32).astype(float)
    errors = numpy.array([0.6 * numpy.sin(row * 1.7) + 0.4 * numpy.cos(row * (0.3 + equation / 7))
                       + 0.2 * numpy.sin(row * (2.3 + equation / 4)) for equation in range(equation_count)])
    responses = numpy.array([4 + equation + predictors[equation] @ numpy.array([0.6 + 0.2 * equation, -0.4])
                          + errors[equation] for equation in range(equation_count)]).astype(numpy.float32).astype(float)
    initial_residuals = (errors + numpy.arange(equation_count)[:, None] * 0.1).astype(numpy.float32).astype(float) if supplied else None
    mask = (row % 9 != 2).astype(int)
    used = mask != 0
    designs = [numpy.column_stack((numpy.ones(used.sum()), predictors[equation, used])) for equation in range(equation_count)]
    observed = responses[:, used]
    initial = numpy.array([observed[equation] - designs[equation] @ linalg.lstsq(designs[equation], observed[equation], lapack_driver='gelsd')[0]
                        for equation in range(equation_count)]) if initial_residuals is None else initial_residuals[:, used]
    residual_covariance = initial @ initial.T / used.sum()
    lower = linalg.cholesky(residual_covariance, lower=True)
    whitening = linalg.solve_triangular(lower, numpy.eye(equation_count), lower=True)
    block_design = linalg.block_diag(*designs)
    whitening = numpy.kron(whitening, numpy.eye(used.sum()))
    whitened_design = whitening @ block_design
    whitened_response = whitening @ observed.ravel()
    coefficients = linalg.lstsq(whitened_design, whitened_response, lapack_driver='gelsd')[0]
    inverse = linalg.pinv(whitened_design)
    covariance = inverse @ inverse.T
    residual = (observed.ravel() - block_design @ coefficients).reshape(equation_count, -1)
    residuals = [[None] * len(row) for _ in range(equation_count)]
    for equation in range(equation_count):
        for position, original in enumerate(numpy.flatnonzero(used)):
            residuals[equation][original] = float(residual[equation, position])
    return dict(name=f'{equation_count}-equations-supplied-{supplied}', equationCount=equation_count, predictorCount=2,
                rowCount=len(row), predictors=predictors.ravel().tolist(), responses=responses.ravel().tolist(),
                mask=mask.tolist(), initialResiduals=None if initial_residuals is None else initial_residuals.ravel().tolist(),
                expected=dict(coefficients=coefficients.tolist(), covariance=covariance.ravel().tolist(),
                              residualCovariance=residual_covariance.ravel().tolist(), residuals=sum(residuals, [])))


Path(__file__).with_name('seemingly-unrelated-regression-operation-fixtures.json').write_text(json.dumps(
    [create_fixture(2), create_fixture(3), create_fixture(2, supplied=True)], indent=2, allow_nan=False) + '\n')


def create_spatial_fixture():
    original = create_fixture(2)
    row_count = original['rowCount']
    predictors = numpy.array(original['predictors']).reshape(2, row_count, 2)
    responses = numpy.array(original['responses']).reshape(2, row_count)
    designs = [numpy.column_stack((numpy.ones(row_count), predictors[equation])) for equation in range(2)]
    initial = numpy.array([responses[equation] - designs[equation] @ linalg.lstsq(
        designs[equation], responses[equation], lapack_driver='gelsd')[0] for equation in range(2)])
    dense = numpy.zeros((row_count, row_count))
    offsets = [0]
    neighbors = []
    weights = []
    for row in range(row_count):
        for neighbor, weight in sorted([((row + 1) % row_count, 0.7), ((row + 3) % row_count, 0.3)]):
            dense[row, neighbor] = numpy.float32(weight)
            neighbors.append(neighbor)
            weights.append(float(numpy.float32(weight)))
        offsets.append(len(neighbors))
    trace = numpy.sum(dense ** 2)
    bound = 0.99 / numpy.max(numpy.sum(numpy.abs(dense), axis=1))
    parameters = []
    for residuals in initial:
        lagged = dense @ residuals
        twice_lagged = dense @ lagged
        matrix = numpy.array([
            [2 * residuals @ lagged, -lagged @ lagged, row_count],
            [2 * lagged @ twice_lagged, -twice_lagged @ twice_lagged, trace],
            [residuals @ twice_lagged + lagged @ lagged, -lagged @ twice_lagged, 0]
        ]) / row_count
        target = numpy.array([residuals @ residuals, lagged @ lagged, residuals @ lagged]) / row_count

        def objective(parameter):
            free = target - matrix[:, :2] @ [parameter, parameter ** 2]
            variance = max(0, free @ matrix[:, 2] / (matrix[:, 2] @ matrix[:, 2]))
            residual = free - variance * matrix[:, 2]
            return residual @ residual

        grid = numpy.linspace(-bound, bound, 2001)
        values = numpy.array([objective(parameter) for parameter in grid])
        candidates = [-bound, bound]
        for position in range(1, len(grid) - 1):
            if values[position] <= values[position - 1] and values[position] <= values[position + 1]:
                candidates.append(optimize.minimize_scalar(objective, bounds=(grid[position - 1], grid[position + 1]),
                                                         method='bounded', options={'xatol': 1e-14}).x)
        parameters.append(min(candidates, key=lambda parameter: (objective(parameter), parameter)))
    filtered_designs = [designs[equation] - parameters[equation] * dense @ designs[equation] for equation in range(2)]
    filtered_responses = numpy.array([responses[equation] - parameters[equation] * dense @ responses[equation] for equation in range(2)])
    filtered_initial = numpy.array([initial[equation] - parameters[equation] * dense @ initial[equation] for equation in range(2)])
    residual_covariance = filtered_initial @ filtered_initial.T / row_count
    lower = linalg.cholesky(residual_covariance, lower=True)
    whitening = numpy.kron(linalg.solve_triangular(lower, numpy.eye(2), lower=True), numpy.eye(row_count))
    filtered_block = linalg.block_diag(*filtered_designs)
    whitened = whitening @ filtered_block
    coefficients = linalg.lstsq(whitened, whitening @ filtered_responses.ravel(), lapack_driver='gelsd')[0]
    inverse = linalg.pinv(whitened)
    covariance = inverse @ inverse.T
    residuals = responses.ravel() - linalg.block_diag(*designs) @ coefficients
    return dict(rowCount=row_count, equationCount=2, predictorCount=2, predictors=original['predictors'],
                responses=original['responses'], weights=dict(offsets=offsets, neighbors=neighbors, weights=weights),
                expected=dict(parameters=parameters, coefficients=coefficients.tolist(), covariance=covariance.ravel().tolist(),
                              residualCovariance=residual_covariance.ravel().tolist(), residuals=residuals.tolist()))


Path(__file__).with_name('seemingly-unrelated-regression-spatial-operation-fixtures.json').write_text(
    json.dumps(create_spatial_fixture(), indent=2, allow_nan=False) + '\n')
