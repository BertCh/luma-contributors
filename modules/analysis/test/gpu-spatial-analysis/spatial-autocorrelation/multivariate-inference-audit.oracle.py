# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-autocorrelation/multivariate-inference-audit.oracle.py

Independent arbitrary-integer Random123 Philox4x32-10 definition (Salmon et al.,
SC 2011), Lemire multiply/reject bounded draw, and exact rational squared distances.
This checks the existing degree-one CPU twin without importing production helpers.
Population variance of [0,1,3,7,15,31] is 117.25; paired column 2*x+5 has the
same standardized distances. The island is retained as a conditional donor.
Also derive the two-complete-row GPU fixture: standardized vectors [-1,-1],
[1,1], mean distance=4, every draw=4. P=9 gives equal-tail p=1 or GeoDa/esda
strict-min-tail p=.1, legacy Bonferroni .2 (both complete rows), tested-family
Bonferroni .1 (one tested row). These numbers do not depend on random draws.
"""
import json
from fractions import Fraction

MASK32 = 2**32-1

def philox(counter, key):
    words = list(counter)
    for _ in range(10):
        first = 0xd2511f53 * words[0]
        second = 0xcd9e8d57 * words[2]
        words = [((second >> 32) ^ words[1] ^ key[0]) & MASK32, second & MASK32,
                 ((first >> 32) ^ words[3] ^ key[1]) & MASK32, first & MASK32]
        key = ((key[0]+0x9e3779b9) & MASK32, (key[1]+0xbb67ae85) & MASK32)
    return words

def bounded(row, replicate, size):
    threshold = (2**32-size) % size
    for block in range(100):
        for word in philox([block, row, replicate, 0x10ca1], (12, 0x80000001)):
            product = word*size
            if (product & MASK32) >= threshold:
                return product >> 32
    raise RuntimeError('rejection loop')

values = [0, 1, 3, 7, 15, 31]
for mode in ['none', 'bonferroni', 'benjamini-hochberg']:
    statistics, probabilities, directions, simulations = [], [], [], []
    for row in range(5):
        observed = (values[row]-values[(row+1) % 5])**2
        candidates = values[:row]+values[row+1:]
        draws = [(values[row]-candidates[bounded(row, replicate, 5)])**2 for replicate in range(1,20)]
        lower = sum(value <= observed for value in draws)
        upper = sum(value >= observed for value in draws)
        probability = Fraction(min(upper,19-upper)+1,20) if mode == 'bonferroni' else min(1,Fraction(2*(min(lower,upper)+1),20))
        if mode == 'bonferroni': probability = min(1, 6*probability)
        statistics.append(float(Fraction(observed*4,469)))
        probabilities.append(probability)
        directions.append(1 if observed*19 < sum(draws) else 2)
        simulations.extend(float(Fraction(value*4,469)) for value in draws)
    if mode == 'benjamini-hochberg':
        order = sorted(range(5), key=lambda row: probabilities[row])
        previous = Fraction(1)
        for rank in range(5,0,-1):
            row = order[rank-1]
            previous = min(previous, probabilities[row]*5/rank)
            probabilities[row] = previous
    classes = [direction if probability <= Fraction(99,100) else 0 for direction, probability in zip(directions,probabilities)]
    print(mode, json.dumps({'statistic': statistics, 'pValues': list(map(float,probabilities)), 'classes': classes}))
    if mode == 'none': print('simulation squared numerators', [round(value*117.25) for value in simulations])
print('two complete rows: observed=4, simulations=4 (9 draws), equal p=1 class0; min-tail p=.1 class2; legacy Bonferroni p=.2 class0 at alpha=.15; tested Bonferroni p=.1 class2')
