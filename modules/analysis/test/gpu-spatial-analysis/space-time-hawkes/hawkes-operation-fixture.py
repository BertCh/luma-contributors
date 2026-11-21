# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

"""Independent score oracle: SciPy least_squares solves the four finite-window EM equations.

Run with /usr/bin/python3. This does not iterate EM: it solves all equations simultaneously,
using a dense parent-weight matrix and direct intensities instead of the twin's log-domain walk.
"""
import json
from pathlib import Path

import numpy
from scipy.optimize import least_squares, brentq

positions = numpy.asarray([[0, 0], [0.2, 0.1], [0.3, 0.2], [3, 3], [3.2, 3.1], [3.4, 3.2], [6, 0], [6.1, 0.2]], dtype=numpy.float32).astype(float)
times = numpy.asarray([0.1, 0.3, 0.7, 3, 3.3, 3.8, 6, 6.5], dtype=numpy.float32).astype(float)
parameters = numpy.asarray([100, 10, 10, 20, 1e-7, 0.01, 0.5, 2, 0.3], dtype=numpy.float32).astype(float)
area, duration, maximum_lag, maximum_distance = parameters[:4]
lags = times[:, None] - times[None, :]
distance_squared = ((positions[:, None, :] - positions[None, :, :])**2).sum(axis=2)
eligible = (lags > 0) & (lags <= maximum_lag) & (distance_squared <= maximum_distance**2)


def evaluate(theta):
    background, branching, decay, variance = numpy.exp(theta)
    weights = numpy.where(eligible, branching * decay / (2 * numpy.pi * variance) * numpy.exp(-decay * numpy.maximum(lags, 0) - distance_squared / (2 * variance)), 0)
    intensity = background + weights.sum(axis=1)
    responsibility = weights / intensity[:, None]
    triggered = responsibility.sum()
    mass = -numpy.expm1(-decay * (duration - times)).sum()
    lag_sum = (responsibility * lags).sum()
    def profile_score(rate):
        remaining = duration-times
        window_mass = -numpy.expm1(-rate*remaining).sum()
        derivative = (remaining*numpy.exp(-rate*remaining)).sum()
        return triggered/rate-lag_sum-triggered*derivative/window_mass
    lower = 1e-6/duration
    next_decay = lower if profile_score(lower)<=0 else brentq(profile_score, lower, triggered/lag_sum, xtol=1e-14)
    next_mass = -numpy.expm1(-next_decay*(duration-times)).sum()
    next_values = numpy.array([(background / intensity).sum() / (area * duration), triggered / next_mass, next_decay, (responsibility * distance_squared).sum() / (2 * triggered)])
    return next_values, background / intensity


initial = numpy.log([parameters[5], parameters[6], parameters[7], parameters[8]**2])
solution = least_squares(lambda theta: numpy.log(evaluate(theta)[0]) - theta, initial, xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=2000)
if not solution.success or numpy.linalg.norm(solution.fun) > 1e-10:
    raise RuntimeError(solution.message)
background, branching, decay, variance = numpy.exp(solution.x)
# Independently evaluate the point-process density at the fixed point. Keep the offspring
# compensator visible so a background-only Poisson likelihood cannot satisfy this oracle.
offspring_intensity = numpy.where(eligible, branching * decay / (2 * numpy.pi * variance) * numpy.exp(-decay * numpy.maximum(lags, 0) - distance_squared / (2 * variance)), 0)
offspring_compensator = branching * (-numpy.expm1(-decay * (duration - times))).sum()
log_likelihood = numpy.log(background + offspring_intensity.sum(axis=1)).sum() - background * area * duration - offspring_compensator
result = dict(positions=positions.ravel().tolist(), times=times.tolist(), parameters=parameters.tolist(), estimates=[background, branching, decay, numpy.sqrt(variance)], backgroundProbability=evaluate(solution.x)[1].tolist(), logLikelihood=float(log_likelihood), offspringCompensator=float(offspring_compensator), cyclingInput=dict(
    positions=[.8060699701309204, .8980562090873718, .2734651565551758, .80330491065979, .3847271800041199, .23606307804584503, .12136342376470566, .6866846084594727],
    times=[4.640753269195557, 1.3180384635925293, 4.200069904327393, 2.6269774436950684],
    parameters=numpy.asarray([10, 10, 10, 10, 1e-20, .01, .5, 1, .5], dtype=numpy.float32).astype(float).tolist()))
Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
