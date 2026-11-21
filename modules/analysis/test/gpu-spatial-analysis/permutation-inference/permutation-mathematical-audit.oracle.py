# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

"""Independent oracle: /usr/bin/python3 <this file>.

No TypeScript, WGSL, recorded buffers or production helpers are read. Philox uses
Python's full-width integer multiplication (Random123 Philox4x32-10 constants).
Conditional draws use an actual shrinking list; total permutations compose the
Hoang/Morris/Rogaway swap-or-not involutions. Statistics use NumPy matrix formulas.
Inclusive ties, observed +1, doubled minimum tail, population simulation deviation.
The masked eighth row and island seventh row leave the first six analysis rows.
Translation and positive scaling preserve centered statistics; G preserves scaling.
"""

import json
import math
import numpy as np
from scipy.special import ndtr

WORD_MASK = (1 << 32) - 1
SEED = 3
PERMUTATIONS = 21
NEIGHBORS = [[1, 2], [0, 3], [1, 4], [2, 5], [0, 5], [0, 4]]
VALUES = np.array([1, 2, 2, 4, 5, 7], dtype=float)
SECOND_VALUES = np.array([7, 3, 5, 2, 11, 8], dtype=float)
WEIGHTS = np.zeros((6, 6))
for row, neighbors in enumerate(NEIGHBORS):
    WEIGHTS[row, neighbors] = 1


def philox(counter, key):
    counter = list(counter)
    key = list(key)
    for round_index in range(10):
        if round_index:
            key = [(key[0] + 0x9E3779B9) & WORD_MASK,
                   (key[1] + 0xBB67AE85) & WORD_MASK]
        product_first = 0xD2511F53 * counter[0]
        product_second = 0xCD9E8D57 * counter[2]
        counter = [(product_second >> 32) ^ counter[1] ^ key[0],
                   product_second & WORD_MASK,
                   (product_first >> 32) ^ counter[3] ^ key[1],
                   product_first & WORD_MASK]
    return counter


def stream(seed, first, second, purpose):
    block = 0
    while True:
        yield from philox([block, first, second, purpose],
                          [seed & WORD_MASK, seed >> 32])
        block += 1


def below(words, bound):
    # Lemire rejection derived with arbitrary-precision products, not u32 partials.
    threshold = (1 << 32) % bound
    while True:
        product = next(words) * bound
        if (product & WORD_MASK) >= threshold:
            return product >> 32


def hash_word(value):
    value ^= value >> 16
    value = value * 0x7FEB352D & WORD_MASK
    value ^= value >> 15
    value = value * 0x846CA68B & WORD_MASK
    return value ^ (value >> 16)


def total_order(seed, permutation):
    order = list(range(6))
    for round_index in range(6 * math.ceil(math.log2(6)) + 6):
        words = stream(seed, permutation, round_index, 0x5A0B5A0B)
        pivot = below(words, 6)
        salt = next(words)
        for row, position in enumerate(order):
            partner = (pivot - position) % 6
            if hash_word(max(position, partner) ^ salt) >> 31:
                order[row] = partner
    return order


def conditional_order(seed, row, permutation):
    # Ordinary full Fisher-Yates swaps; the focal row never enters the list.
    order = [other for other in range(6) if other != row]
    words = stream(seed, row, permutation, 0x10CA1)
    for drawn in range(2):
        picked = drawn + below(words, len(order) - drawn)
        order[drawn], order[picked] = order[picked], order[drawn]
    return order[:2]


def tails(observed, simulated, alternative='two-sided', stop_after=None):
    greater = np.cumsum(np.array(simulated) >= observed)
    lesser = np.cumsum(np.array(simulated) <= observed)
    multiplier = 2 if alternative == 'two-sided' else 1
    counts = greater if alternative == 'greater' else lesser if alternative == 'lesser' else np.minimum(greater, lesser)
    if alternative == 'folded':
        mean = np.mean(simulated)
        counts = np.cumsum(np.abs(np.array(simulated) - mean) >= abs(observed - mean))
    used = len(simulated)
    stopped = False
    if stop_after is not None:
        reached = np.flatnonzero(counts >= stop_after)
        if len(reached):
            used, stopped = int(reached[0]) + 1, True
    count = int(counts[used - 1])
    probability = min(1, multiplier * count / used) if stopped else min(1, multiplier * (count + 1) / (used + 1))
    return count, probability, used


def global_statistic(name, order, values=VALUES, second_values=SECOND_VALUES):
    centered = values - values.mean()
    centered_second = second_values - second_values.mean()
    squared = centered @ centered
    cross_scale = math.sqrt(squared * (centered_second @ centered_second))
    permuted = centered[order]
    if name == 'moran':
        return 6 * (permuted @ WEIGHTS @ permuted) / (WEIGHTS.sum() * squared)
    if name == 'geary':
        differences = permuted[:, None] - permuted[None, :]
        return 5 * np.sum(WEIGHTS * differences ** 2) / (2 * WEIGHTS.sum() * squared)
    if name == 'getisOrdG':
        permuted_raw = values[order]
        return (permuted_raw @ WEIGHTS @ permuted_raw) / (values.sum() ** 2 - values @ values)
    if name == 'bivariateMoran':
        return 6 * (centered @ WEIGHTS @ centered_second[order]) / (WEIGHTS.sum() * cross_scale)
    return 6 * ((WEIGHTS @ permuted) @ (WEIGHTS @ centered_second[order])) / (np.sum(WEIGHTS.sum(axis=1) ** 2) * cross_scale)


def global_result(name, alternative='two-sided', stop_after=None):
    observed = global_statistic(name, list(range(6)))
    simulated = [global_statistic(name, total_order(SEED, permutation))
                 for permutation in range(1, PERMUTATIONS + 1)]
    count, probability, used = tails(observed, simulated, alternative, stop_after)
    simulated = simulated[:used]
    mean, deviation = np.mean(simulated), np.std(simulated, ddof=0)
    standardized = (observed - mean) / deviation
    # Five bins keep every non-extreme member away from an interior bin edge,
    # so float32 normalization cannot legitimately move an exact boundary tie.
    histogram = np.histogram(simulated, bins=5, range=(min(simulated), max(simulated)))[0]
    return {'summary': [observed, probability, count, used, mean, deviation,
                        standardized, ndtr(-abs(standardized)), min(simulated), max(simulated), 1, 6],
            'reference': simulated, 'histogram': histogram.tolist()}


def local_result(name, alternative='greater', stop_after=None, seed=SEED):
    values = VALUES if 'Join' not in name else np.array([1, 1, 0, 1, 1, 1], dtype=float)
    second = SECOND_VALUES if 'Join' not in name else np.array([0, 1, 1, 0, 1, 1], dtype=float)
    centered, centered_second = values - values.mean(), second - second.mean()
    squared = centered @ centered
    cross_scale = math.sqrt(squared * (centered_second @ centered_second))

    def statistic(row, neighbors):
        if name == 'localMoran':
            return centered[row] * sum(centered[neighbors])
        if name == 'localGeary':
            return sum((centered[row] - centered[neighbors]) ** 2)
        if name == 'localG':
            return sum(values[neighbors])
        if name == 'localGStar':
            return values[row] + sum(values[neighbors])
        if name == 'localBivariateMoran':
            return centered[row] * sum(centered_second[neighbors])
        if name == 'localLeeL':
            return sum(centered[neighbors]) * sum(centered_second[neighbors])
        if name == 'localJoinCount':
            return values[row] * sum(values[neighbors])
        if name == 'localBivariateJoinCountBJC':
            return values[row] * (1 - second[row]) * sum((1 - values[neighbors]) * second[neighbors])
        return values[row] * second[row] * sum(values[neighbors] * second[neighbors])

    observed, exceedances, probabilities, counts, replay = [], [], [], [], []
    for row, neighbors in enumerate(NEIGHBORS):
        raw = statistic(row, neighbors)
        simulated = [statistic(row, conditional_order(seed, row, permutation))
                     for permutation in range(PERMUTATIONS)]
        replay.append(simulated)
        scale = {'localMoran': 5 / squared, 'localGeary': 6 / squared,
                 'localG': 1 / (sum(values) - values[row]), 'localGStar': 1 / sum(values),
                 'localBivariateMoran': 5 / cross_scale, 'localLeeL': 6 / cross_scale}.get(name, 1)
        observed.append(raw * scale)
        if 'Join' in name and raw == 0:
            exceedances.append(4294967295)
            probabilities.append(None)
            counts.append(4294967295)
        else:
            count, probability, used = tails(raw, simulated, alternative, stop_after)
            exceedances.append(count)
            probabilities.append(probability)
            counts.append(used)
    return {'observed': observed + [None, None],
            'exceedances': exceedances + [4294967295, 4294967295],
            'probabilities': probabilities + [None, None],
            'counts': counts + [4294967295, 4294967295], 'replayFirstRow': replay[0]}


if __name__ == '__main__':
    from fractions import Fraction
    global_names = ['moran', 'geary', 'getisOrdG', 'bivariateMoran', 'leesL']
    local_names = ['localMoran', 'localGeary', 'localG', 'localGStar',
                   'localBivariateMoran', 'localLeeL', 'localJoinCount',
                   'localBivariateJoinCountBJC', 'localBivariateJoinCountCLC']
    result = {'global': {name: global_result(name) for name in global_names},
              'local': {name: local_result(name) for name in local_names},
              'sequentialGlobal': global_result('leesL', 'two-sided', 3),
              'sequentialLocal': local_result('localMoran', 'greater', 3),
              'globalAlternatives': {alternative: global_result('leesL', alternative)['summary'][1:3]
                                     for alternative in ['greater', 'lesser', 'directed', 'folded']},
              'localAlternatives': {alternative: local_result('localMoran', alternative)
                                    for alternative in ['two-sided', 'lesser', 'directed', 'folded']}}
    for name in global_names:
        order = total_order(SEED, 1)
        rescaled = 2 * VALUES + (0 if name == 'getisOrdG' else 16)
        shifted_second = 4 * SECOND_VALUES + 8
        assert np.isclose(global_statistic(name, order),
                          global_statistic(name, order, rescaled, shifted_second))
    result['cutoffThresholds'] = [[float(np.float32(level)),
        [int(Fraction(float(np.float32(level))) * denominator)
         for denominator in [3, 22, 99, 1048577, 2147483647]]]
        for level in [2**-149, .05, 9/22, 13/22, 1-2**-24]]
    by_boundary = sum((Fraction(1, rank) for rank in range(1, 69)), Fraction(0)) / 128
    result['review2'] = {
        'byExactBoundaryDifference': float(by_boundary - Fraction(.037531705388509126)),
        # Reciprocal edges, x=y=[-2^34,2^34]: SS=2^69, cross=-2^68, lag product=2^68.
        'largeReciprocalBivariateMoran': [-.5, -.5],
        'largeReciprocalLeeL': [1, 1],
        'reciprocalLocalG': [1, 1],
        'singlePositiveGeneralGDenominator': sum(Fraction(first * second)
            for index, first in enumerate([5, 0, 0]) for other, second in enumerate([5, 0, 0]) if index != other)
    }
    result['review2']['singlePositiveGeneralGDenominator'] = int(result['review2']['singlePositiveGeneralGDenominator'])
    # Reciprocal centered pairs have geometric mean 2*|a*b|, independent of any
    # frexp/ldexp shader implementation, including both signs of odd exponent sums.
    result['review2']['geometricMeanCases'] = [[first, second, 2 * first * second]
        for first, second in [(2**34, 2**34), (2**34, 3*2**33), (2**-50, 3*2**-51)]]
    print(json.dumps(result, indent=2))
