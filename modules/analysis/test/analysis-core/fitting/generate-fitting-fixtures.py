# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent float64 NumPy/SciPy LRN-02 oracle. Run with /usr/bin/python3.

Adam: Kingma & Ba Algorithm 1, https://arxiv.org/pdf/1412.6980.
Fourier: Tancik et al. page 2, https://arxiv.org/pdf/2006.10739.
Harmonics: scipy.special.sph_harm (Condon-Shortley phase), DLMF 14.30.1.
No trained models, third-party weights or datasets; all samples below are synthetic.
"""
import json
from pathlib import Path
import numpy
import scipy
from scipy.special import sph_harm


def fit(initial, evaluate, steps):
    weights = numpy.asarray(initial, dtype=numpy.float64)
    first = numpy.zeros_like(weights)
    second = numpy.zeros_like(weights)
    history = []
    for iteration in range(1, steps + 1):
        loss, gradient = evaluate(weights)
        history.append(float(loss))
        first = 0.9 * first + 0.1 * gradient
        second = 0.999 * second + 0.001 * gradient ** 2
        weights -= 0.05 * (first / (1 - 0.9 ** iteration)) / (
            numpy.sqrt(second / (1 - 0.999 ** iteration)) + 1e-8)
    history.append(float(evaluate(weights)[0]))
    return dict(weights=weights.tolist(), firstMoment=first.tolist(),
                secondMoment=second.tolist(), lossHistory=history)


quadratic_initial = [2, -1, 0.5]
quadratic_target = [-1, 2, 0]
quadratic_scale = numpy.array([1, 2, 3])
def quadratic(weights):
    residual = weights - quadratic_target
    return numpy.sum(quadratic_scale * residual ** 2) / 2, quadratic_scale * residual


features = numpy.array([[1, -1, 0], [1, 0, 1], [1, 1, 0], [1, 2, -1],
                     [1, -2, 1], [1, 0.5, 0.5], [1, 1, 2], [1, -1, -2], [1, 2, 1]])
labels = numpy.array([0, 1, 1, 1, 0, 0, 1, 0, 1])
def logistic(weights):
    score = features @ weights
    loss = numpy.mean(numpy.logaddexp(0, score) - labels * score)
    gradient = features.T @ (1 / (1 + numpy.exp(-score)) - labels) / len(labels)
    return loss, gradient


def harmonics(coordinates, maximum_degree):
    result = []
    for longitude, latitude in coordinates:
        azimuth = numpy.deg2rad(longitude)
        polar = numpy.deg2rad(90 - latitude)
        row = []
        for degree in range(maximum_degree + 1):
            for order in range(-degree, degree + 1):
                value = sph_harm(abs(order), degree, azimuth, polar)
                row.append(float(value.real if order == 0 else numpy.sqrt(2) * (
                    value.imag if order < 0 else value.real)))
        result.extend(row)
    return result


def philox_keys(row_count, seed=(123, 456), epoch=0):
    """Independent vectorized Random123 Philox4x32-10, SC2011 constants."""
    mask = numpy.uint64(0xffffffff)
    first = numpy.arange(row_count, dtype=numpy.uint64)
    second = numpy.full(row_count, epoch, dtype=numpy.uint64)
    third = numpy.zeros(row_count, dtype=numpy.uint64)
    fourth = numpy.full(row_count, 0x4649544d, dtype=numpy.uint64)
    key_first, key_second = map(numpy.uint64, seed)
    for iteration in range(10):
        product_first = first * numpy.uint64(0xd2511f53)
        product_second = third * numpy.uint64(0xcd9e8d57)
        first, second, third, fourth = (
            (product_second >> numpy.uint64(32)) ^ second ^ key_first,
            product_second & mask,
            (product_first >> numpy.uint64(32)) ^ fourth ^ key_second,
            product_first & mask)
        key_first = (key_first + numpy.uint64(0x9e3779b9)) & mask
        key_second = (key_second + numpy.uint64(0xbb67ae85)) & mask
    return first


keys = philox_keys(131072)
unique_keys, key_counts = numpy.unique(keys, return_counts=True)
collision_key = unique_keys[key_counts > 1][0]
collision_rows = numpy.flatnonzero(keys == collision_key).tolist()
minibatch = dict(seed=[123, 456], rowCount=257,
    order=numpy.argsort(philox_keys(257), kind='stable').tolist(),
    collision=dict(rowCount=131072, key=int(collision_key), rows=collision_rows))


# Isolate bias correction from optimizer trajectory error. Each case starts with zero moments
# and a unit gradient at a caller-supplied completed-iteration count. Evaluate 1-beta^t with
# NumPy log/expm1 in binary64, using exactly the float32 parameters that the GPU receives.
adam_bias_correction = []
nearest_beta = float(numpy.nextafter(numpy.float32(1), numpy.float32(0)))
for beta_first, beta_second, iterations in [
    (0.9, 0.999, [1, 2, 3, 7, 8, 8191, 8192, 10000]),
    (nearest_beta, nearest_beta, [1, 2, 3, 8191, 8192, 10000]),
    (0, 0, [1, 10000]),
    (0, nearest_beta, [1, 10000]),
    (nearest_beta, 0, [1, 10000])]:
    beta_first = float(numpy.float32(beta_first))
    beta_second = float(numpy.float32(beta_second))
    for iteration in iterations:
        first_correction = 1.0 if beta_first == 0 else -numpy.expm1(iteration * numpy.log(beta_first))
        second_correction = 1.0 if beta_second == 0 else -numpy.expm1(iteration * numpy.log(beta_second))
        first = (1 - beta_first) / first_correction
        second = (1 - beta_second) / second_correction
        weight = -float(numpy.float32(0.05)) * first / (numpy.sqrt(second) + float(numpy.float32(1e-8)))
        adam_bias_correction.append(dict(beta1=beta_first, beta2=beta_second, iteration=iteration,
            firstCorrection=float(first_correction), secondCorrection=float(second_correction), weight=float(weight)))


# Independent first-step range and finite-loss plateau cases. Binary64 has ample exponent
# range for these intermediates; parameters are first rounded to the GPU's binary32 values.
adam_range_updates = []
for name, gradient, learning_rate, epsilon in [
    ('overflow-positive', 2.0 ** 64, 2.0 ** 64, 1e-8),
    ('overflow-negative', -(2.0 ** 64), 2.0 ** 64, 1e-8),
    ('underflow-positive', 2.0 ** -50, 2.0 ** -126, 1e-30),
    ('underflow-negative', -(2.0 ** -50), 2.0 ** -126, 1e-30)]:
    gradient = float(numpy.float32(gradient))
    learning_rate = float(numpy.float32(learning_rate))
    epsilon = float(numpy.float32(epsilon))
    adam_range_updates.append(dict(name=name, gradient=gradient, learningRate=learning_rate,
        epsilon=epsilon, firstMoment=0.5 * gradient, secondMoment=0.5 * gradient ** 2,
        weight=-learning_rate * gradient / (abs(gradient) + epsilon)))

large_loss = 3 * 2.0 ** 126
next_large_loss = float(numpy.nextafter(numpy.float32(large_loss), numpy.finfo(numpy.float32).max))
adam_loss_plateaus = []
for previous_loss, current_loss, tolerance in [
    (large_loss, -large_loss, 1.5), (-large_loss, large_loss, 1.5),
    (large_loss, -large_loss, 2), (large_loss, large_loss / 2, 0.5),
    (large_loss, large_loss / 2, 0.25), (0, large_loss, 1.5),
    (0, large_loss, large_loss), (0.5, -0.5, 1), (0.5, -0.5, 0.5),
    (large_loss, 0, 0.5), (large_loss, 0, 1),
    (large_loss, next_large_loss, 1e-7), (large_loss, next_large_loss, 6e-8),
    (-large_loss, -next_large_loss, 1e-7), (-large_loss, -next_large_loss, 6e-8)]:
    previous_loss = float(numpy.float32(previous_loss))
    current_loss = float(numpy.float32(current_loss))
    tolerance = float(numpy.float32(tolerance))
    converged = abs(current_loss - previous_loss) <= tolerance * max(1, abs(previous_loss))
    adam_loss_plateaus.append(dict(previousLoss=previous_loss, currentLoss=current_loss,
        tolerance=tolerance, converged=int(converged)))


coordinates = [[0, 0], [90, 0], [-180, 45], [180, 45], [13, 90], [-70, -90],
               [45, 89.99], [-122.5, 37.75]]
fourier_coordinates = numpy.array([[0, 0, 0], [0.25, -0.5, 1], [-0.75, 0.5, -0.25]])
frequencies = numpy.array([[0, 0, 0], [1, 0, 0], [0.5, -1, 2], [2, 3, -0.25]])
phase = 2 * numpy.pi * fourier_coordinates @ frequencies.T
rows = numpy.arange(769)
gradients = numpy.stack([(rows % 17 - 8) / 8, (rows % 11 - 5) / 4, numpy.ones(769)], axis=1)
mask = rows % 13 != 0
fixture = dict(numpy=numpy.__version__, scipy=scipy.__version__, minibatch=minibatch,
    adamBiasCorrection=adam_bias_correction, adamRangeUpdates=adam_range_updates,
    adamLossPlateaus=adam_loss_plateaus,
    quadratic=dict(initial=quadratic_initial, target=quadratic_target, scale=quadratic_scale.tolist(),
                   steps=8, **fit(quadratic_initial, quadratic, 8)),
    logistic=dict(features=features.tolist(), labels=labels.tolist(), initial=[0, 0, 0],
                  steps=10, **fit([0, 0, 0], logistic, 10)),
    gradients=dict(rowCount=769, sums=gradients[mask].sum(axis=0).tolist()),
    fourier=dict(coordinates=fourier_coordinates.tolist(), frequencies=frequencies.tolist(),
                 features=numpy.stack([numpy.cos(phase), numpy.sin(phase)], axis=2).flatten().tolist()),
    harmonics=dict(coordinates=coordinates, maximumDegree=4, features=harmonics(coordinates, 4)),
    highHarmonics=dict(coordinates=[[21, 48]], maximumDegree=39, features=harmonics([[21, 48]], 39)))
Path(__file__).with_name('fixtures').joinpath('fitting-numpy-scipy.json').write_text(
    json.dumps(fixture, indent=2, allow_nan=False) + '\n')
