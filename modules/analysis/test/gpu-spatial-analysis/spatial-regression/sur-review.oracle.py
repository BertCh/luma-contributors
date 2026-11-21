# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-regression/sur-review.oracle.py
Independent dense block GLS; supplied residuals seed only round 1, never convergence.
"""
import json
import numpy
from pathlib import Path
from scipy import linalg, optimize


def fit(predictors, responses, seed, budget):
    equations, rows = predictors.shape
    designs = [numpy.column_stack((numpy.ones(rows), values)) for values in predictors]
    block = numpy.zeros((equations * rows, 2 * equations))
    previous = []
    for equation, design in enumerate(designs):
        block[equation * rows:(equation + 1) * rows, equation * 2:(equation + 1) * 2] = design
        previous.extend(numpy.linalg.lstsq(design, responses[equation], rcond=None)[0])
    previous = numpy.array(previous)
    residuals = seed.copy()
    for count in range(1, budget + 1):
        sigma = residuals @ residuals.T / rows
        precision = numpy.kron(numpy.linalg.inv(sigma), numpy.eye(rows))
        information = block.T @ precision @ block
        covariance = numpy.linalg.inv(information)
        coefficients = numpy.linalg.solve(information, block.T @ precision @ responses.ravel())
        residuals = (responses.ravel() - block @ coefficients).reshape(equations, rows)
        change = numpy.max(numpy.abs(coefficients - previous) / numpy.maximum(1, numpy.abs(previous)))
        status = 0 if count > 1 and change < 1e-5 else 4
        if status == 0:
            break
        previous = coefficients
    return dict(coefficients=coefficients.tolist(), covariance=covariance.ravel().tolist(),
                residualCovariance=sigma.ravel().tolist(), residuals=residuals.ravel().tolist(),
                status=[status], iterationCount=[count])


rows = numpy.arange(8)
first = 2 * (rows % 2) - 1
second = 2 * ((rows // 2) % 2) - 1
third = 2 * ((rows // 4) % 2) - 1
predictors = numpy.array([first, second], dtype=float)
responses = numpy.array([2 + first + second + first * second,
                         3 + 2 * second + 0.5 * first + first * second + third])
result = dict(predictors=predictors.ravel().tolist(), responses=responses.ravel().tolist(),
                      initialResiduals=predictors.ravel().tolist(),
                      rounds=[dict(maximumIterations=budget, expected=fit(predictors, responses, predictors, budget))
                              for budget in [1, 2, 100]])


def spatial_fit(predictors, responses, dense, budget=1, tolerance=1e-5, history=None):
    equations, rows, columns = predictors.shape
    designs = [numpy.column_stack((numpy.ones(rows), values)) for values in predictors]
    # Normalize the accepted near-unit rows, then use exactly the same W in every stage.
    dense = dense / dense.sum(axis=1)[:, None]
    initial = numpy.array([response - design @ linalg.lstsq(design, response, lapack_driver='gelsd')[0]
                           for design, response in zip(designs, responses)])
    parameters = []
    trace = numpy.sum(dense ** 2)
    for errors in initial:
        lagged = dense @ errors
        twice = dense @ lagged
        matrix = numpy.array([[2 * errors @ lagged, -lagged @ lagged, rows],
                              [2 * lagged @ twice, -twice @ twice, trace],
                              [errors @ twice + lagged @ lagged, -lagged @ twice, 0]]) / rows
        target = numpy.array([errors @ errors, lagged @ lagged, errors @ lagged]) / rows
        def objective(parameter):
            free = target - matrix[:, :2] @ [parameter, parameter ** 2]
            variance = max(0, free @ matrix[:, 2] / (matrix[:, 2] @ matrix[:, 2]))
            return numpy.sum((free - variance * matrix[:, 2]) ** 2)
        grid = numpy.linspace(-0.99, 0.99, 2001)
        candidates = [-0.99, 0.99]
        for position in range(1, len(grid) - 1):
            if objective(grid[position]) <= min(objective(grid[position - 1]), objective(grid[position + 1])):
                candidates.append(optimize.minimize_scalar(objective, bounds=(grid[position - 1], grid[position + 1]), method='bounded', options={'xatol': 1e-14}).x)
        parameters.append(min(candidates, key=lambda parameter: (objective(parameter), parameter)))
    filtering = [numpy.eye(rows) - parameter * dense for parameter in parameters]
    filtered_designs = [transform @ design for transform, design in zip(filtering, designs)]
    filtered_response = numpy.array([transform @ response for transform, response in zip(filtering, responses)])
    filtered_residuals = numpy.array([transform @ errors for transform, errors in zip(filtering, initial)])
    block = linalg.block_diag(*filtered_designs)
    previous = numpy.concatenate([linalg.lstsq(design, response, lapack_driver='gelsd')[0] for design, response in zip(filtered_designs, filtered_response)])
    for count in range(1, budget + 1):
        sigma = filtered_residuals @ filtered_residuals.T / rows
        precision = numpy.kron(numpy.linalg.inv(sigma), numpy.eye(rows))
        information = block.T @ precision @ block
        covariance = numpy.linalg.inv(information)
        coefficients = numpy.linalg.solve(information, block.T @ precision @ filtered_response.ravel())
        filtered_residuals = (filtered_response.ravel() - block @ coefficients).reshape(equations, rows)
        comparison, old = coefficients.copy(), previous.copy()
        comparison[::columns + 1] *= 1 - numpy.array(parameters)
        old[::columns + 1] *= 1 - numpy.array(parameters)
        change = numpy.max(numpy.abs(comparison - old) / numpy.maximum(1, numpy.abs(old)))
        status = 0 if history is None or (count > 1 and change < tolerance) else 4
        residuals = responses.ravel() - linalg.block_diag(*designs) @ coefficients
        output = dict(coefficients=coefficients.tolist(), covariance=covariance.ravel().tolist(), residualCovariance=sigma.ravel().tolist(), residuals=residuals.tolist(), status=[status], iterationCount=[count], parameters=parameters)
        if history is not None:
            history.append(output)
        if status == 0:
            break
        previous = coefficients
    return output


source = json.loads(Path(__file__).with_name('seemingly-unrelated-regression-spatial-operation-fixtures.json').read_text())
spatial_predictors = numpy.array(source['predictors']).reshape(2, 40, 2)
spatial_responses = numpy.round(numpy.array(source['responses']).reshape(2, 40) * 8) / 8
weight_values = numpy.array([value * (1 + 8e-6 if (slot // 2) % 2 else 1 - 8e-6) for slot, value in enumerate(source['weights']['weights'])], dtype=numpy.float32).astype(float)
dense = numpy.zeros((40, 40))
for row in range(40):
    for slot in range(source['weights']['offsets'][row], source['weights']['offsets'][row + 1]):
        dense[row, source['weights']['neighbors'][slot]] = weight_values[slot]
result['spatialShift'] = dict(predictors=source['predictors'], responses=spatial_responses.ravel().tolist(), weights=dict(offsets=source['weights']['offsets'], neighbors=source['weights']['neighbors'], weights=weight_values.tolist()), expected=spatial_fit(spatial_predictors, spatial_responses, dense))
strong = json.loads(Path(__file__).with_name('iterated-sur-fixtures.json').read_text())[0]
history = []
strong_expected = spatial_fit(numpy.array(strong['predictors']).reshape(3, 40, 2), numpy.array(strong['responses']).reshape(3, 40), dense, budget=100, tolerance=0.02, history=history)
result['spatialFinal'] = dict(predictors=strong['predictors'], responses=strong['responses'], weights=result['spatialShift']['weights'], tolerance=0.02, expected=strong_expected, previous=history[-2])
print(json.dumps(result, indent=2))
