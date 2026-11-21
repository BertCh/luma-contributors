# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
# Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/space-time-statistics/local-moran-output-oracles.oracle.py
# Independent Random123 integer arithmetic and a conditional single-neighbor null.
# Three analyzed values [1,2,4], directed cycle 0->1->2->0; missing fourth row excluded.
from fractions import Fraction
import json

def philox(counter, key):
    mask = (1 << 32) - 1
    for iteration in range(10):
        first = 0xD2511F53 * counter[0]
        second = 0xCD9E8D57 * counter[2]
        counter = [(second >> 32) ^ counter[1] ^ key[0], second & mask,
                   (first >> 32) ^ counter[3] ^ key[1], first & mask]
        key = [(key[0] + 0x9E3779B9) & mask, (key[1] + 0xBB67AE85) & mask]
    return counter

values = [Fraction(-4, 3), Fraction(-1, 3), Fraction(5, 3)]
neighbors = [1, 2, 0]
# Find a deterministic stream with a nontrivial BY adjustment (neither all zero nor all one).
for seed in range(10000):
    exceedances = []
    draws = []
    for row in range(3):
        others = [candidate for candidate in range(3) if candidate != row]
        # Bound 2 has no rejection: Lemire's exact integer draw is the high word of word*2.
        selected = [others[(philox([0, row, permutation, 0x10ca1], [seed, 0])[0] * 2) >> 32]
                    for permutation in range(7)]
        draws.append(selected)
        observed = values[row] * values[neighbors[row]]
        exceedances.append(sum(values[row] * values[candidate] >= observed for candidate in selected))
    if exceedances[0] == 0:
        probabilities = [Fraction(count + 1, 8) for count in exceedances]
        order = sorted(range(3), key=lambda row: probabilities[row])
        adjusted = [None] * 3
        running = Fraction(1)
        for rank in range(3, 0, -1):
            row = order[rank - 1]
            running = min(running, probabilities[row] * 3 * Fraction(11, 6) / rank)
            adjusted[row] = running
        print(json.dumps({'seed': seed, 'draws': draws, 'exceedances': exceedances,
                          'p': list(map(float, probabilities)), 'BY': list(map(float, adjusted)),
                          'observed': [float(2 * values[row] * values[neighbors[row]] / Fraction(14, 3)) for row in range(3)]}))
        break
