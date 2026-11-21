# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Run: /usr/bin/python3 modules/analysis/test/gpu-raster/raster-skeleton-status-oracles.oracle.py

Independent set-based Zhang-Suen (1984) deletion conditions: 2..6 neighbors, exactly
one cyclic 0->1 transition, and the phase-specific north/east/south/west products.
No production code imports. Reports a named pending deletion after the two iterations
executed by maximumIterations=1 (the second iteration is the verification pass).
"""

OFFSETS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]


def deletions(pixels, phase):
    removed = set()
    for row, column in pixels:
        neighborhood = [(row + vertical, column + horizontal) in pixels
                        for vertical, horizontal in OFFSETS]
        if not 2 <= sum(neighborhood) <= 6:
            continue
        if sum(not neighborhood[index] and neighborhood[(index + 1) % 8]
               for index in range(8)) != 1:
            continue
        north, east, south, west = (neighborhood[index] for index in [0, 2, 4, 6])
        products = (north and east and south, east and south and west) if phase == 0 else (
            north and east and west, north and south and west)
        if not any(products):
            removed.add((row, column))
    return removed


pixels = {(row, column) for row in range(9) for column in range(9)}
for iteration in range(2):
    for phase in range(2):
        removed = deletions(pixels, phase)
        pixels -= removed
        print('iteration', iteration + 1, 'phase', phase, 'removed', len(removed))
print('retained after two iterations:', sorted(pixels))
print('pending in the next phase:', sorted(deletions(pixels, 0)))
assert (2, 2) in pixels and (2, 2) in deletions(pixels, 0)
assert not deletions({(4, 4)}, 0) and not deletions({(4, 4)}, 1)
