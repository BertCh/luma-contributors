# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

"""Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/scan-statistics/scan-definition.oracle.py

Kulldorff conditional Poisson and Bernoulli likelihoods, evaluated as a
multinomial deviance (one half of scipy.stats.power_divergence's G statistic).
Windows are enumerated as sets; circles include every distance tie. Secondary
clusters greedily select disjoint center-best sets, lowest center breaking ties.
No implementation or random-number code is imported.

Eight equally populated zones, one case at zone 3. Every possible relocation of
the single case has the same scan maximum: high=the singleton, low=an empty
four-zone endpoint window. Hence all Monte Carlo maxima equal the observed
maximum and the rank p-value is 1 for every seed, including for secondaries.
For space-time permutation there is one case per zone in two buckets. With a
one-zone spatial cap and one-bucket time cap every possible fixed-margin table
has the same maximum. All eight centers are truncated (the population cap admits
four zones). These facts supply independent, seed-independent replicate values.
"""

import json
import math
import sys
from bisect import bisect_right
from itertools import combinations
from scipy.stats import power_divergence


def likelihood(observed, expected):
    return float(power_divergence(observed, f_exp=expected, lambda_="log-likelihood").statistic / 2)


def scan(model, direction, shape="circle", capacity=8):
    times = 2 if model == "space-time-permutation" else 1
    counts = [[int(zone % 2 == bucket) for bucket in range(2)] if times == 2
              else [int(zone == 3)] for zone in range(8)]
    total = sum(map(sum, counts))
    population = 16 if model == "bernoulli" else 8
    best = []
    for center in range(8):
        ordered = sorted(range(8), key=lambda zone: (abs(zone - center), zone))
        candidates = []
        for size in range(1, min(4, capacity) + 1):
            members = set(ordered[:size])
            radius = max(abs(zone - center) for zone in members)
            if shape == "circle" and members != {zone for zone in range(8) if abs(zone - center) <= radius}:
                continue
            for bucket in range(times):
                observed = sum(counts[zone][bucket] for zone in members)
                expected = total * size / (8 * times)
                if ((direction == "high" and observed <= expected) or
                        (direction == "low" and observed >= expected) or observed == expected):
                    continue
                if model == "bernoulli":
                    inside = 2 * size
                    table = [observed, inside - observed, total - observed,
                             population - inside - total + observed]
                    null = [expected, inside - expected, total - expected,
                            population - inside - total + expected]
                else:
                    table = [observed, total - observed]
                    null = [expected, total - expected]
                score = likelihood(table, null)
                candidates.append((score, size, bucket, observed, expected, radius, members))
        # Stable order: smaller spatial set, then earlier time bucket wins a tie.
        best.append(max(candidates, key=lambda row: row[0]) if candidates else None)
    remaining = {center for center, candidate in enumerate(best) if candidate}
    indices, statistics = [], []
    while remaining:
        center = max(sorted(remaining), key=lambda zone: best[zone][0])
        score, size, bucket, observed, expected, radius, members = best[center]
        indices.extend([center, size, bucket, bucket])
        statistics.extend([score, observed, expected, 1, radius, observed / expected, int(capacity < 4), 0])
        remaining = {zone for zone in remaining if not (best[zone][6] & members)}
    return dict(indices=indices, clusterStatistics=statistics,
                zoneStatistics=[row[0] if row else 0 for row in best],
                maximum=max(row[0] for row in best if row),
                summary=[len(indices) // 4, total, 3, 8 if capacity < 4 else 0])


def philox_words(replicate, cell, seed=17, purpose=0x45424353):
    """Random123 Philox4x32-10, using Python's full-width integer products."""
    block = 0
    while True:
        counter = [block, replicate, cell, purpose]
        key = [seed, 0]
        for _ in range(10):
            first = 0xD2511F53 * counter[0]
            second = 0xCD9E8D57 * counter[2]
            counter = [(second >> 32) ^ counter[1] ^ key[0], second & 0xFFFFFFFF,
                       (first >> 32) ^ counter[3] ^ key[1], first & 0xFFFFFFFF]
            key = [(key[0] + 0x9E3779B9) & 0xFFFFFFFF,
                   (key[1] + 0xBB67AE85) & 0xFFFFFFFF]
        yield from counter
        block += 1


def draw_poisson(mean, replicate, cell):
    # Independent arrival-time construction of a Poisson process, rather than
    # the shader's repeated product and exp threshold. All fixture means < 30.
    if not math.isfinite(mean) or mean <= 0:
        return 0
    elapsed = 0
    for count, word in enumerate(philox_words(replicate, cell)):
        elapsed -= math.log(((word >> 9) + 0.5) / 8388608)
        if elapsed >= mean:
            return count


def expectation_scan(masked=False):
    positions = [0, 1, 2, 3, 10, 11, 12, 13]
    baseline = [[value] * 3 for value in [2, 2, 2, 2, 3, 3, 1, 0]]
    cases = [[2, 2, 2], [2, 5, 8], [2, 4, 6], [2, 1, 1],
             [3, 3, 3], [3, 6, 9], [1, 1, 1], [100, 100, 100]]
    neighbors = 4
    if masked:
        baseline = [[0,2,2], [-2,2,2], [math.nan,2,2], [math.inf,2,2], [0,0,0], [2,2,2], [1,1,1], [2,2,2]]
        cases = [[100,6,8] for _ in range(4)] + [[100,100,100], [2,2,2], [1,1,1], [2,2,5]]
        neighbors = 1
    def valid_cells(zone, length):
        return [bucket for bucket in range(3-length, 3) if math.isfinite(baseline[zone][bucket]) and baseline[zone][bucket] > 0]

    def evaluate(counts):
        best = (0, 0, 0, [], 0, 0)
        scores = []
        for center in range(8):
            neighborhood = sorted(range(8), key=lambda zone: (abs(positions[zone] - positions[center]), zone))[:neighbors]
            center_best = 0
            for length in range(1, 4):
                for size in range(1, neighbors+1):
                    for members in combinations(neighborhood, size):
                        if any(not valid_cells(zone, length) for zone in members):
                            continue
                        observed = sum(sum(counts[zone][bucket] for bucket in valid_cells(zone, length)) for zone in members)
                        expected = sum(sum(baseline[zone][bucket] for bucket in valid_cells(zone, length)) for zone in members)
                        score = observed * math.log(observed / expected) + expected - observed if observed > expected else 0
                        center_best = max(center_best, score)
                        if score > best[0]:
                            # Output membership order is decreasing rate, ties by neighbor slot.
                            ordered = sorted(members, key=lambda zone: (-sum(counts[zone][bucket] for bucket in valid_cells(zone, length)) / sum(baseline[zone][bucket] for bucket in valid_cells(zone, length)), neighborhood.index(zone)))
                            best = (score, center, length, ordered, observed, expected)
            scores.append(center_best)
        return best, scores

    best, scores = evaluate(cases)
    maxima = []
    for replicate in range(3):
        counts = [[draw_poisson(baseline[zone][bucket], replicate, zone * 3 + bucket)
                   for bucket in range(3)] for zone in range(8)]
        maxima.append(evaluate(counts)[0][0])
    score, center, length, members, observed, expected = best
    return dict(clusterZones=members + [0xFFFFFFFF] * (neighbors - len(members)),
                clusterStatistics=[score, observed, expected, (1 + sum(value >= score for value in maxima)) / 4],
                statistics=[score] + maxima + [0, 0], zoneStatistics=scores,
                summary=[1, len(members), length, center, 3])


def cylinder_fixtures():
    """Independent set/time enumeration; sorted-label removal gives fixed margins."""
    observed = [[6,6,0,0], [6,6,0,0]] + [[0,0,2,2] for _ in range(6)]
    result = {}
    for model in ["poisson", "bernoulli", "space-time-permutation"]:
        weights = [[3] * 4] * 2 + [[1] * 4] * 6 if model == "space-time-permutation" else [[10 if model == "bernoulli" else 1] * 4 for _ in range(8)]
        total_weight = sum(map(sum, weights))
        replicate_tables = []
        def below(stream, bound):
            # Lemire's unbiased bounded integer, with arbitrary-width products.
            threshold = (1 << 32) % bound
            while True:
                product = next(stream) * bound
                if product & 0xffffffff >= threshold:
                    return product >> 32
        for replicate in range(3):
            table = [0] * 32
            if model == "poisson":
                # C=48, so each nonempty one-case Philox block owns one draw.
                for draw in range(48):
                    word = next(philox_words(replicate, draw, purpose=0x5ca75747))
                    table[(word * 32) >> 32] += 1
            elif model == "bernoulli":
                stream = philox_words(replicate, 0, purpose=0x5ca75747)
                while sum(table) < 48:
                    cell = below(stream, 320) // 10
                    if below(stream, 10) < 10 - table[cell]:
                        table[cell] += 1
            else:
                stream = philox_words(replicate, 0, purpose=0x5ca75747)
                labels = [bucket for bucket in range(4) for _ in range(12)]
                for zone, counts in enumerate(observed):
                    for _ in range(sum(counts)):
                        bucket = labels.pop(below(stream, len(labels)))
                        table[4 * zone + bucket] += 1
            replicate_tables.append([table[4*zone:4*zone+4] for zone in range(8)])
        for mode in ["retrospective", "prospective"]:
            def evaluate(table):
                best = []
                for center in range(8):
                    candidates = []
                    # Enumerate geometric circles by radius, rather than prefix sums.
                    for radius in sorted({abs(zone-center) for zone in range(8)}):
                        members = {zone for zone in range(8) if abs(zone-center) <= radius}
                        if sum(sum(weights[zone]) for zone in members) > total_weight / 2:
                            continue
                        for first in range(4):
                            for last in range(first, min(first+2, 4)):
                                if mode == "prospective" and last != 3:
                                    continue
                                count = sum(sum(table[zone][first:last+1]) for zone in members)
                                mass = sum(sum(weights[zone][first:last+1]) for zone in members)
                                expected = 48 * mass / total_weight
                                if count <= expected:
                                    continue
                                actual = [count, 48-count]
                                null = [expected, 48-expected]
                                if model == "bernoulli":
                                    actual += [mass-count, total_weight-mass-48+count]
                                    null += [mass-expected, total_weight-mass-48+expected]
                                score = likelihood(actual, null)
                                candidates.append((score, center, len(members), first, last, count, expected, radius, members))
                    best.append(max(candidates, key=lambda row: row[0]) if candidates else None)
                return best
            best = evaluate(observed)
            maxima = [max((row[0] for row in evaluate(table) if row), default=0) for table in replicate_tables]
            indices, statistics = [], []
            remaining = {zone for zone, row in enumerate(best) if row}
            while remaining and len(indices) < 16:
                center = max(sorted(remaining), key=lambda zone: best[zone][0])
                score, center, size, first, last, count, expected, radius, members = best[center]
                indices += [center, size, first, last]
                statistics += [score, count, expected, (1+sum(value >= score for value in maxima))/4, radius, count/expected, 0, 0]
                remaining = {zone for zone in remaining if not best[zone][8] & members}
            result[model + ":" + mode] = dict(indices=indices + [0]*(16-len(indices)), clusterStatistics=statistics + [0]*(32-len(statistics)), statistics=[max((row[0] for row in best if row), default=0)] + maxima + [0,0], zoneStatistics=[row[0] if row else 0 for row in best], summary=[len(indices)//4,48,3,0])
    return result


if __name__ == "__main__" and "--review2" in sys.argv:
    # /usr/bin/python3 -B modules/analysis/test/gpu-spatial-analysis/scan-statistics/scan-definition.oracle.py --review2
    import struct
    from fractions import Fraction
    from decimal import Decimal, getcontext
    getcontext().prec = 70
    f32 = lambda value: struct.unpack("<f", struct.pack("<f", value))[0]
    for values in [[.2, .2, 1], [1, 2**-24, 2**-25]]:
        exact = sum((Fraction(f32(value)) for value in values), Fraction())
        print("R12 exact/rounded", exact, f32(float(exact)))
        words = int(exact * 2**149)
        print("R12 integer words", [(words >> (32*index)) & 0xffffffff for index in range(10)])
    print("R12 winner LLR", 9 * Decimal(7).ln())
    for limit in [1, 2, 3]:
        sizes = [2, 3, 2]  # whole next circles at centers -1,0,1.
        print("R13 frame", limit, "loss", [int(size <= limit) for size in sizes])
    print("R15 packed cap", Fraction(f32(.7)) * 10000, "admitted", int(Fraction(f32(.7)) * 10000))
    print("R15 winner LLR", 2000 * (Decimal(2000)/1600).ln() + 8000 * (Decimal(8000)/8400).ln())
elif __name__ == "__main__" and "--review" in sys.argv:
    # /usr/bin/python3 -B modules/analysis/test/gpu-spatial-analysis/scan-statistics/scan-definition.oracle.py --review
    from decimal import Decimal, getcontext
    getcontext().prec = 70
    count, baseline = Decimal(1000), Decimal("999.99993896484375")
    print("R9 tiny", count * (count / baseline).ln() + baseline - count)
    print("R9 singleton", 2 * Decimal(2).ln() - 1)
    print("R1 conditional", 2 * math.log(2) + 10 * math.log(10 / 11))
    positions, population = [-1, 0, 1], [1, 1, 1]
    for shape in ["circle", "nearest"]:
        lost = []
        for center in range(3):
            order = sorted(range(3), key=lambda zone: (abs(positions[zone]-positions[center]), zone))
            next_members = order[:2] if shape == "nearest" else [zone for zone in range(3) if abs(positions[zone]-positions[center]) <= abs(positions[order[1]]-positions[center])]
            lost.append(int(sum(population[zone] for zone in next_members) <= .75 * sum(population)))
        print("R5", shape, lost, "overflow", sum(lost))
    print("R5 winner", 9 * math.log(3))
elif __name__ == "__main__" and "--masked-baseline" in sys.argv:
    print(json.dumps(expectation_scan(masked=True), indent=2))
elif __name__ == "__main__" and "--cylinders" in sys.argv:
    print(json.dumps(cylinder_fixtures(), indent=2))
elif __name__ == "__main__":
    for model in ["poisson", "bernoulli", "space-time-permutation"]:
        for direction in ["high", "low", "both"]:
            print(model, direction, json.dumps(scan(model, direction, capacity=1 if model == "space-time-permutation" else 8)))
    print("expectation-based", json.dumps(expectation_scan()))
