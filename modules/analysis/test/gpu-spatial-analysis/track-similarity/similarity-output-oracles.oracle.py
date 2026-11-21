# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/track-similarity/similarity-output-oracles.oracle.py

Exhaustive paths and alignments, no dynamic-programming recurrence or implementation imports.
"""
from itertools import combinations
from math import dist

first = [(0,0),(2,0),(4,0)]
for second in [[(0,1),(4,1)], [(0,1),(4,1),(8,1)]]:
    def paths(row, column):
        if (row, column) == (len(first)-1, len(second)-1):
            yield [(row,column)]
        for next_row, next_column in [(row+1,column), (row,column+1), (row+1,column+1)]:
            if next_row < len(first) and next_column < len(second):
                for tail in paths(next_row, next_column):
                    yield [(row,column)] + tail

    costs = [[dist(first[row], second[column]) for row,column in path] for path in paths(0,0)]

    def alignments(row, column):
        if row == len(first) and column == len(second):
            yield (0,0)
        if row < len(first):
            for edits, penalty in alignments(row+1,column):
                yield edits+1, penalty+dist(first[row],(0,0))
        if column < len(second):
            for edits, penalty in alignments(row,column+1):
                yield edits+1, penalty+dist(second[column],(0,0))
        if row < len(first) and column < len(second):
            distance = dist(first[row],second[column])
            for edits, penalty in alignments(row+1,column+1):
                yield edits+int(distance>1.1), penalty+distance

    aligned = list(alignments(0,0))
    longest = 0
    for length in range(1,min(len(first),len(second))+1):
        for first_indices in combinations(range(len(first)),length):
            for second_indices in combinations(range(len(second)),length):
                if all(dist(first[a],second[b])<=1.1 for a,b in zip(first_indices,second_indices)):
                    longest = length
    frechet = min(map(max,costs))
    print(dict(frechet=frechet, frechetWithin=int(frechet<=2.5), dynamicTimeWarping=min(map(sum,costs)),
               editDistanceOnRealSequence=min(x[0] for x in aligned), editDistanceWithRealPenalty=min(x[1] for x in aligned),
               longestCommonSubsequence=longest/min(len(first),len(second)), maximumDistance=max(dist(a,b) for a in first for b in second)))
