# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent Random123 full-integer products and recursive hierarchy traversal oracle.

Run from the repository root with /usr/bin/python3 and no -I. The implementation uses Python
arbitrary-width multiplication (not the twin's 16-bit partial products), recursive tree traversal
(not packed Morton codes), and an expanded ticket population (not prefix-rank searches).
"""
import json
from pathlib import Path
import numpy as np


def draw_word(counter, seed):
    key = [seed, 0x53425350]
    mask = 2**32 - 1
    for round_index in range(10):
        first = 0xD2511F53 * counter[0]
        second = 0xCD9E8D57 * counter[2]
        counter = [(second >> 32) ^ counter[1] ^ key[0], second & mask,
                   (first >> 32) ^ counter[3] ^ key[1], first & mask]
        key = [(key[0] + 0x9E3779B9) & mask, (key[1] + 0xBB67AE85) & mask]
    return counter[0]


def create_case(weighted):
    positions = np.array([[0.13, 0.21], [0.87, 0.74], [0.32, 0.62], [0.13, 0.21],
                          [0.91, 0.12], [0.44, 0.31], [0.25, 0.86], [0.77, 0.44],
                          [0.68, 0.93], [0.56, 0.56], [0.03, 0.97], [0.98, 0.03]], dtype=np.float32)
    strata = [2, 9] * 6
    tickets = [1, 2, 2, 1, 3, 1, 1, 2, 2, 1, 3, 1] if weighted else [1] * 12
    allocations = [{'stratumId': 9, 'sampleCount': 2}, {'stratumId': 2, 'sampleCount': 3}]
    seed, replicate = (1234, 17) if weighted else (79, 3)
    cells = np.floor(positions.astype(np.float64) * 65535).astype(int).tolist()

    def traverse(members, stratum, level=0, prefix=0):
        if level == 16:
            return members
        groups = [[] for _ in range(4)]
        rotation = draw_word([prefix, stratum * 16 + level, replicate, 0], seed) % 4
        for member in members:
            child = ((cells[member][0] >> (15-level)) & 1) + 2*((cells[member][1] >> (15-level)) & 1)
            groups[child].append(member)
        result = []
        for rotated in range(4):
            child = (rotated - rotation) % 4
            if groups[child]:
                result += traverse(groups[child], stratum, level+1, prefix*4+child)
        return result

    selected, probabilities = [], []
    for allocation in allocations:
        stratum, count = allocation.values()
        members = traverse([row for row in range(12) if strata[row] == stratum], stratum)
        population = [member for member in members for _ in range(tickets[member])]
        step = len(population) // count
        limit = step if weighted else len(population)
        attempt = 0
        while True:
            word = draw_word([stratum, 0x80000000 + attempt, replicate, 0], seed)
            if word >= (2**32 % limit):
                break
            attempt += 1
        shift = word % limit
        for sample in range(count):
            rank = shift + sample*step if weighted else (shift + sample*len(population)//count) % len(population)
            member = population[rank]
            selected.append(100 + member*7)
            probabilities.append(tickets[member]/step if weighted else count/len(population))
    return dict(positions=positions.ravel().tolist(), strata=strata, tickets=tickets if weighted else None,
                frameIds=[100 + row*7 for row in range(12)], seed=seed, replicate=replicate,
                allocations=allocations, expectedIds=selected, expectedProbabilities=probabilities,
                expectedWeights=[1/value for value in probabilities], expectedStatus=[2, 0, 3, 0])


Path(__file__).with_suffix('.json').write_text(json.dumps([create_case(False), create_case(True)], indent=2) + '\n')
