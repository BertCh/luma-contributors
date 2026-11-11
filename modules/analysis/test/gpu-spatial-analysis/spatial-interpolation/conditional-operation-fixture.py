# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

"""Generate a SciPy Schur-complement/Cholesky oracle and integer Philox normals.

Run /usr/bin/python3. The CPU twin factors the full joint matrix; this script
conditions through scipy.linalg.solve then factors the smaller Schur complement.
Philox uses Python's exact 64-bit products, independent of the TS 16-bit decomposition.
"""
import json
from pathlib import Path

import numpy
from scipy.linalg import cholesky, solve

positions = numpy.asarray([[0, 0], [1, 0], [0, 1]], dtype=numpy.float32).astype(float)
queries = numpy.asarray([[0.3, 0.4], [0.9, -0.1], [1.3, 0.8], [0.5, 1.2]], dtype=numpy.float32).astype(float)
observations = numpy.asarray([[1, 0.1, 0, 1], [-0.2, 0.2, 0, 1], [0.4, 0.05, 0, 1]], dtype=numpy.float32).astype(float)
seed, replicate_offset, replicate_count, mean = 1234, 7, 3, 0.3
sill, nugget, effective_range = 1.2, 0.2, 2.5


def philox(counter, key):
    mask = 2**32 - 1
    counter = list(counter)
    key = list(key)
    for iteration in range(10):
        if iteration:
            key = [(key[0] + 0x9E3779B9) & mask, (key[1] + 0xBB67AE85) & mask]
        first = 0xD2511F53 * counter[0]
        second = 0xCD9E8D57 * counter[2]
        counter = [(second >> 32) ^ counter[1] ^ key[0], second & mask,
                   (first >> 32) ^ counter[3] ^ key[1], first & mask]
    return counter


normals = numpy.empty((replicate_count, len(queries)))
for replicate in range(replicate_count):
    for query in range(len(queries)):
        # Registered PHILOX_PURPOSES.conditionalGaussianSimulation, counter/key layout of GPU.
        words = philox([replicate + replicate_offset, query, 0, 0], [seed, 0x43475349])
        radius = ((words[0] >> 9) + 0.5) / 8388608
        angle = ((words[1] >> 8) + 0.5) / 16777216
        normals[replicate, query] = numpy.sqrt(-2 * numpy.log(radius)) * numpy.cos(2 * numpy.pi * angle)


def covariance(first, second, code):
    difference = first[:, None, :] - second[None, :, :]
    scaled = numpy.linalg.norm(difference, axis=2) / effective_range
    if code == 0:
        correlation = numpy.where(scaled >= 1, 0, 1 - 1.5 * scaled + 0.5 * scaled**3)
    elif code == 1:
        correlation = numpy.exp(-3 * scaled)
    elif code == 2:
        correlation = numpy.exp(-3 * scaled**2)
    else:
        argument = (4.749031386012701 if code == 3 else 5.924462618951797) * scaled
        correlation = (1 + argument + (argument**2 / 3 if code == 4 else 0)) * numpy.exp(-argument)
    return sill * correlation + nugget * numpy.all(difference == 0, axis=2)


fixtures = []
for code, model in enumerate(['spherical', 'exponential', 'gaussian', 'matern-3/2', 'matern-5/2']):
    observation_covariance = covariance(positions, positions, code) + numpy.diag(observations[:, 1])
    cross = covariance(queries, positions, code)
    conditional_mean = mean + cross @ solve(observation_covariance, observations[:, 0] - mean, assume_a='pos')
    conditional_covariance = covariance(queries, queries, code) - cross @ solve(observation_covariance, cross.T, assume_a='pos')
    conditional_lower = cholesky(conditional_covariance, lower=True)
    draws = conditional_mean[None, :] + normals @ conditional_lower.T
    moments = numpy.column_stack([conditional_mean, numpy.diag(conditional_covariance)])
    fixtures.append(dict(model=model, moments=moments.ravel().tolist(), values=draws.ravel().tolist()))

result = dict(positions=positions.ravel().tolist(), observations=observations.ravel().tolist(), queries=queries.ravel().tolist(), seed=seed, replicateOffset=replicate_offset, replicateCount=replicate_count, mean=mean, sill=sill, nugget=nugget, range=effective_range, normals=normals.ravel().tolist(), fixtures=fixtures)
Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
