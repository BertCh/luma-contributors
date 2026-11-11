# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

"""Independent Gaussian batch conditioning oracle. Run with /usr/bin/python3.

Each prefix is a single joint Gaussian over all states, conditioned on all measured positions.
There is no Kalman recurrence or RTS backward pass. The last block of each prefix gives the
filtered moments; blocks of the complete segment give the smoothed moments.
"""

import json
from pathlib import Path

import numpy


def solveBatch(positions, times, noise, measurement, velocity):
    count = len(times)
    width = 4 * count
    precision = numpy.zeros((width, width))
    information = numpy.zeros(width)
    initial = numpy.zeros((4, 4))
    initial[:2, :2] = measurement
    initial[2:, 2:] = velocity
    inverseInitial = numpy.linalg.inv(initial)
    precision[:4, :4] = inverseInitial
    information[:4] = inverseInitial @ numpy.r_[positions[0], [0, 0]]
    for row in range(1, count):
        delta = times[row] - times[row - 1]
        transition = numpy.eye(4)
        transition[:2, 2:] = numpy.eye(2) * delta
        process = numpy.block([
            [noise * delta**3 / 3, noise * delta**2 / 2],
            [noise * delta**2 / 2, noise * delta],
        ])
        constraint = numpy.zeros((4, width))
        constraint[:, (row - 1) * 4:row * 4] = -transition
        constraint[:, row * 4:(row + 1) * 4] = numpy.eye(4)
        precision += constraint.T @ numpy.linalg.solve(process, constraint)
        if numpy.isfinite(positions[row]).all():
            observation = numpy.zeros((2, width))
            observation[:, row * 4:row * 4 + 2] = numpy.eye(2)
            precision += observation.T @ numpy.linalg.solve(measurement, observation)
            information += observation.T @ numpy.linalg.solve(measurement, positions[row])
    covariance = numpy.linalg.inv(precision)
    state = numpy.linalg.solve(precision, information)
    return state.reshape(count, 4), [covariance[row * 4:(row + 1) * 4, row * 4:(row + 1) * 4] for row in range(count)]


def createFixture(full):
    positions = numpy.array([[2, -1], [3, 0], [numpy.nan, numpy.nan], [6, 2], [-2, 4], [-1, 2], [0, 1], [5, 0]], dtype=numpy.float32).astype(float)
    times = numpy.array([0, 0.5, 2, 3, 0, 1, 1.5, 5], dtype=numpy.float32).astype(float)
    parameters = numpy.array([0.4, 0.1, 0.6, 2, 0.3, 1, 3, -0.2, 2, 2] if full else [0.4, 2, 3, 2], dtype=numpy.float32).astype(float)
    def readCovariance(offset):
        return numpy.array([[parameters[offset], parameters[offset + 1]], [parameters[offset + 1], parameters[offset + 2]]])
    noise = readCovariance(0) if full else numpy.eye(2) * parameters[0]
    measurement = readCovariance(3) if full else numpy.eye(2) * parameters[1]
    velocity = readCovariance(6) if full else numpy.eye(2) * parameters[2]
    filteredStates = numpy.zeros((8, 4))
    smoothedStates = numpy.zeros((8, 4))
    filteredCovariances = [None] * 8
    smoothedCovariances = [None] * 8
    for start, end in [(0, 4), (4, 7), (7, 8)]:
        for finish in range(start + 1, end + 1):
            states, covariances = solveBatch(positions[start:finish], times[start:finish], noise, measurement, velocity)
            filteredStates[finish - 1] = states[-1]
            filteredCovariances[finish - 1] = covariances[-1]
        states, covariances = solveBatch(positions[start:end], times[start:end], noise, measurement, velocity)
        smoothedStates[start:end] = states
        smoothedCovariances[start:end] = covariances
    def packCovariances(covariances):
        return numpy.concatenate([matrix.flatten(order='F') if full else matrix[numpy.ix_([0, 2], [0, 2])].flatten(order='F') for matrix in covariances]).tolist()
    return {
        'covarianceMode': 'full' if full else 'isotropic',
        'positions': [float(value) if numpy.isfinite(value) else None for value in positions.flat],
        'timestamps': times.tolist(),
        'trackOffsets': [0, 4, 8],
        'parameters': parameters.tolist(),
        'filteredState': filteredStates.flatten().tolist(),
        'filteredCovariance': packCovariances(filteredCovariances),
        'smoothedState': smoothedStates.flatten().tolist(),
        'smoothedCovariance': packCovariances(smoothedCovariances),
        'status': [1, 0, 2, 0, 1, 0, 0, 1],
    }


fixture = {
    '_license': 'SPDX-License-Identifier: MIT',
    '_copyright': 'SPDX-FileCopyrightText: Copyright (c) vis.gl contributors',
    'oracle': 'NumPy joint Gaussian precision solve, independent of sequential filtering/smoothing',
    'cases': [createFixture(False), createFixture(True)],
}
Path(__file__).with_name('trajectory-kalman-batch.fixture.json').write_text(json.dumps(fixture, indent=2, allow_nan=False) + '\n')
