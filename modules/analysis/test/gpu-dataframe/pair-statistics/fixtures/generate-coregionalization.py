# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Regenerate with /usr/bin/python3. Independent NumPy Goulard–Voltz oracle.

Uses bin residuals directly (the implementation instead caches sufficient statistics),
and numpy.linalg.eigh rather than the implementation's Jacobi eigensolver.
Matérn shapes use 80-digit Decimal evaluation of the closed-form correlations;
the implementation instead uses a bounded small-argument exponential series.
"""
import json
from pathlib import Path
from decimal import Decimal, localcontext
import numpy as np



def matern_shape_decimal(model, lag, distance_range):
    """Independent high-precision subtraction, retaining the binary64 input values exactly."""
    factor = 4.749031386012701 if model == 3 else 5.924462618951797
    with localcontext() as context:
        context.prec = 80
        argument = Decimal.from_float(factor) * Decimal.from_float(float(lag)) / Decimal.from_float(float(distance_range))
        correlation_polynomial = 1 + argument
        if model == 4:
            correlation_polynomial += argument * argument / 3
        return float(1 - correlation_polynomial * (-argument).exp())


def shape(model, lag, nugget, sill, distance_range):
    scaled = lag / distance_range
    if model == 0:
        value = np.where(scaled >= 1, 1, 1.5 * scaled - 0.5 * scaled ** 3)
    elif model == 1:
        value = -np.expm1(-3 * scaled)
    elif model == 2:
        value = -np.expm1(-3 * scaled ** 2)
    else:
        value = np.fromiter((matern_shape_decimal(model, distance, distance_range) for distance in lag), dtype=float, count=len(lag))
    return nugget + sill * value


def fixture(name, variable_count, structures, bins, counts, empirical, budget=128, tolerance=1e-5):
    distances = np.asarray(bins, dtype=np.float32).astype(float)
    parameters = np.asarray([tolerance] + sum([list(item) for item in structures], []), dtype=np.float32).astype(float)
    empirical = np.asarray(empirical, dtype=np.float32).astype(float)
    weights = np.asarray(counts, dtype=np.uint32).astype(float)
    usable = (distances > 0) & np.isfinite(distances) & (weights > 0) & np.isfinite(empirical).all(axis=1)
    upper = np.triu_indices(variable_count)
    observations = np.zeros((len(bins), variable_count, variable_count))
    observations[:, upper[0], upper[1]] = empirical
    observations[:, upper[1], upper[0]] = empirical
    shapes = np.stack([shape(int(parameters[1+4*index]), distances, *parameters[2+4*index:5+4*index]) for index in range(len(structures))], axis=1)
    matrices = np.zeros((len(structures), variable_count, variable_count))
    total_weight = weights[usable].sum()
    converged = 0
    iteration_count = 0
    change = 0
    if total_weight:
        normalized_weights = weights[usable] / total_weight
        for iteration in range(budget):
            previous = matrices.copy()
            for structure in range(len(structures)):
                # Form the bin residual of all OTHER structures, directly.
                residual = observations[usable].copy()
                for other in range(len(structures)):
                    if other != structure:
                        residual -= shapes[usable, other, None, None] * matrices[other]
                current_shape = shapes[usable, structure]
                conditional = np.einsum('b,bij->ij', normalized_weights * current_shape, residual)
                conditional /= np.dot(normalized_weights, current_shape ** 2)
                eigenvalues, eigenvectors = np.linalg.eigh(conditional)
                matrices[structure] = (eigenvectors * np.maximum(eigenvalues, 0)) @ eigenvectors.T
            largest = max(np.abs(previous).max(), np.abs(matrices).max())
            absolute_change = np.abs(matrices - previous).max()
            change = absolute_change / largest if largest else 0
            iteration_count += 1
            if absolute_change <= parameters[0] * largest:
                converged = 1
                break
        residual = observations[usable] - np.einsum('bs,sij->bij', shapes[usable], matrices)
        objective = np.einsum('b,bij,bij->', normalized_weights, residual, residual)
    else:
        objective = None
    return dict(name=name, variableCount=variable_count, structureCount=len(structures), maximumIterations=budget,
                distances=distances.tolist(), semivariances=empirical.ravel().tolist(), pairCounts=counts,
                parameters=parameters.tolist(), matrices=matrices.ravel().tolist() if total_weight else None,
                residual=objective, relativeChange=change if total_weight else None, iterationCount=iteration_count,
                converged=converged, validity=int(total_weight > 0), invalidCount=int((~usable).sum()))


fixtures = [
    fixture('indefinite-cross', 2, [(0, 0, 1, 1)], [1, 2], [3, 11], [[1, 2, 1], [1, 2, 1]]),
    # Gaussian shapes are ~3e-16 / 3e-18: direct 1 - exp(-argument) loses digits or vanishes.
    fixture('gaussian-tiny-lag-1e8', 2, [(2, 0, 1, 1e8)], [1], [1], [[3e-16, 0, 3e-16]]),
    fixture('gaussian-tiny-lag-1e9', 2, [(2, 0, 1, 1e9)], [1], [1], [[3e-18, 0, 3e-18]]),
    fixture('exponential-tiny-lag', 2, [(1, 0, 1, 1e18)], [1], [1], [[3e-18, 0, 3e-18]]),
    fixture('negative-cross', 2, [(1, 0, 1, 2)], [0.2, 0.7, 1.2], [3, 10, 7], [[1, -0.6, 2], [1.8, -1, 3], [2, -1.1, 3.5]]),
    fixture('padding', 2, [(0, 0, 1, 1)], [1, 0, 2], [10, 20, 0], [[2, -1, 3], [99, 99, 99], [999, 999, 999]]),
    fixture('repeated-eigenvalues', 2, [(0, 0, 1, 1)], [1, 2], [4, 8], [[2, 0, 2], [2, 0, 2]]),
    fixture('zero-matrix', 2, [(0, 0, 1, 1)], [1], [4], [[0, 0, 0]]),
    fixture('duplicate-structures', 2, [(0, 0, 1, 1), (0, 0, 1, 1)], [1, 2], [4, 8], [[2, -1, 3], [2, -1, 3]]),
    fixture('matern-5-2', 2, [(4, 0, 1, 2)], [0.2, 0.7, 1.2], [3, 10, 7], [[1, -0.6, 2], [1.8, -1, 3], [2, -1.1, 3.5]]),
    fixture('all-zero-counts', 2, [(0, 0, 1, 1)], [1, 2], [0, 0], [[2, -1, 3], [999, 999, 999]]),
    fixture('empty', 2, [(0, 0, 1, 1)], [], [], np.empty((0, 3))),
    fixture('zero-budget', 2, [(0, 0, 1, 1)], [1], [10], [[2, -1, 3]], budget=0),
]
# Empirical identity-model bins are rounded to float32 from an independent Decimal shape.
for model, family in [(3, 'matern-3-2'), (4, 'matern-5-2')]:
    for distance_range in [1e8, 1e9]:
        semivariance = matern_shape_decimal(model, 1, distance_range)
        fixtures.append(fixture(f'{family}-tiny-lag-{distance_range:.0e}', 2,
                                [(model, 0, 1, distance_range)], [1], [1],
                                [[semivariance, 0, semivariance]]))

random = np.random.default_rng(20261114)
for variable_count, structure_count, budget in [(3, 2, 512), (4, 4, 3), (3, 3, 1024)]:
    structures = [(0, 1, 0, 1)] + [(index % 5, 0, 1, 0.7 + index * 2) for index in range(1, structure_count)]
    bins = np.linspace(0.1, 5, 20).astype(np.float32)
    truth = []
    for structure in range(structure_count):
        loadings = random.normal(size=(variable_count, variable_count)) / (structure + 1)
        truth.append(loadings @ loadings.T)
    observations = np.einsum('bs,sij->bij', np.stack([shape(item[0], bins.astype(float), *item[1:]) for item in structures], axis=1), truth)
    upper = np.triu_indices(variable_count)
    empirical = observations[:, upper[0], upper[1]] + random.normal(scale=0.015, size=(20, len(upper[0])))
    fixtures.append(fixture(f'{variable_count}-variables-{structure_count}-structures', variable_count, structures,
                            bins.tolist(), random.integers(1, 100, size=20).tolist(), empirical, budget=budget))
Path(__file__).with_name('coregionalization-numpy.json').write_text(json.dumps(fixtures, indent=2, allow_nan=False) + '\n')

# Scalar regressions test float64 accuracy from zero through the CPU series transition and tail.
shape_fixtures = []
for model, family, factor in [(3, 'matern-3/2', 4.749031386012701), (4, 'matern-5/2', 5.924462618951797)]:
    distances = [0, 1e-12, 1e-9, 1e-8, 1e-5, 0.01,
                 (0.5 / factor) * (1 - 2e-15), 0.5 / factor, (0.5 / factor) * (1 + 2e-15),
                 0.1, 1, 10, 1000]
    for distance in distances:
        shape_fixtures.append(dict(model=family, scaledDistance=distance,
                                   shape=matern_shape_decimal(model, distance, 1)))
Path(__file__).with_name('matern-shapes-decimal.json').write_text(json.dumps(shape_fixtures, indent=2, allow_nan=False) + '\n')
