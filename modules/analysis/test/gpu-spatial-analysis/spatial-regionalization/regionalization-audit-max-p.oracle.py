# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
# Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-regionalization/regionalization-audit-max-p.oracle.py
"""Enumerate every contiguous partition of four disjoint pairs, optionally with an island."""
import itertools
import numpy as np

for island in [False, True]:
    values = np.array([0, 2, 10, 14, -4, -4, 20, 26] + ([99] if island else []), dtype=float)
    mass = np.array([1, 2] * 4 + ([1] if island else []))
    candidates = []
    for cuts in itertools.product([False, True], repeat=4):
        groups = []
        for pair, cut in enumerate(cuts):
            first = 2 * pair
            groups += [[first], [first + 1]] if cut else [[first, first + 1]]
        if island:
            groups.append([8])
        infeasible = sum(mass[group].sum() < 3 for group in groups)
        ssd = sum(np.sum((values[group] - values[group].mean()) ** 2) for group in groups)
        candidates.append((infeasible, -len(groups), ssd, groups))
    best = min(candidates)
    print('island', island, 'infeasible, -P, SSD, groups', best,
          'standardized SSD', best[2] / values.var())


def philox_key(row, start, seed):
    # Random123 Philox4x32-10 definition, using Python's exact integer products. No TypeScript
    # RNG or growth code is imported. The registered Max-P key is (seed, purpose=8).
    counter = [row, start, 0, 0]
    key = [seed, 8]
    mask = 2 ** 32 - 1
    for _ in range(10):
        first = 0xD2511F53 * counter[0]
        second = 0xCD9E8D57 * counter[2]
        counter = [(second >> 32) ^ counter[1] ^ key[0], second & mask,
                   (first >> 32) ^ counter[3] ^ key[1], first & mask]
        key = [(key[0] + 0x9E3779B9) & mask, (key[1] + 0xBB67AE85) & mask]
    return counter[0]


# Two cliques of four: three identical core rows with masses 1,2,3 meet threshold six;
# the fourth row is a zero-mass outlier. A core seed grows only core rows with grown SSD zero;
# an outlier seed has positive grown SSD. Lowest all-core start wins. Final each-clique SSD
# is 3/4 * (6^2 + 8^2) = 75 including its enclave, irrespective of the seed row.
for seed in [0, 1, 2, 3]:
    minima = [[min(range(begin, begin + 4), key=lambda row: philox_key(row, start, seed))
               for begin in [0, 4]] for start in range(8)]
    selected = next(start for start, roots in enumerate(minima) if roots[0] != 3 and roots[1] != 7)
    print('two-clique seed', seed, 'seed rows', minima, 'selectedStart', selected, 'final SSD', 150)

# Enclave-budget counterexample, on a 6x6 rook lattice: a published leftover region {1,6}
# contains (0,1) and (1,0), which have Manhattan distance two and no internal rook link.
# Its row values are 12 and 17, so merging these disconnected leftovers contributes 12.5 SSD;
# the conservative singleton fallback contributes zero. Other regions are evaluated using the
# independent pair-distance identity in evaluateAuditMaxPRemainder (no growth/search code).
coordinates = np.array([[0, 1], [1, 0]])
assert np.abs(coordinates[0] - coordinates[1]).sum() == 2
remainder_values = np.array([12., 17.])
print('disconnected remainder SSD', np.sum((remainder_values - remainder_values.mean()) ** 2),
      'singleton SSD', 0)

# Active constrained refinement, independently defined by the existing hand-worked three-region
# partition: moving row 4 changes SSD from 1141/6 to 174 and leaves source mass four >= three.
refinement_values = np.array([0, 9, 2, 12, 5, 15, 1, 10, 4], dtype=float)
refinement_labels = np.array([0, 1, 1, 0, 1, 2, 0, 2, 2])
for exponent in [-80, 64, 120]:
    table = (refinement_values * 2.0 ** exponent).astype(np.float32).astype(float)
    standardized = (table - table.mean()) / table.std()
    within = sum(np.sum((standardized[refinement_labels == region]
                         - standardized[refinement_labels == region].mean()) ** 2)
                 for region in range(3))
    print('active refinement scale', exponent, 'population variance in original units',
          refinement_values.var(), 'standardized SSD', within)
