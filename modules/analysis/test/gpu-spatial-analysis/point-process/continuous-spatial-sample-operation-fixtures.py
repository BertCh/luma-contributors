# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Shapely line interpolation/triangle covers oracle with independent big-integer Philox.

Run from the repository root with /usr/bin/python3 (without -I).
"""
import bisect
import itertools
import json
import math
from pathlib import Path
from shapely.geometry import LineString, Point, Polygon


def draw_block(replicate, seed):
    counter = [replicate, 0, 0, 0]
    key = [seed, 0x43535350]
    mask = 2**32 - 1
    for _ in range(10):
        first, second = 0xD2511F53 * counter[0], 0xCD9E8D57 * counter[2]
        counter = [(second >> 32) ^ counter[1] ^ key[0], second & mask,
                   (first >> 32) ^ counter[3] ^ key[1], first & mask]
        key = [(key[0] + 0x9E3779B9) & mask, (key[1] + 0xBB67AE85) & mask]
    return counter


def create_case(area):
    vertices = ([(0, 0), (3, 0), (0, 3), (3, 0), (3, 3), (0, 3),
                 (4, 0), (6, 0), (4, 2)] if area else
                [(0, 0), (2, 0), (2, 0), (2, 2), (0, 2), (0, 0)])
    legacy = [(1, 1), (4.5, 0.5)] if area else [0, 2.5, 7]
    bounds, spacing, seed, replicate, waves, capacity = [0, 0, 6, 3], 0.5, 891, 5, 2, 64
    triangles = [Polygon(vertices[start:start+3]) for start in range(0, len(vertices), 3)] if area else []
    line = None if area else LineString(vertices)
    cumulative = list(itertools.accumulate([math.dist(first, second) for first, second in zip(vertices, vertices[1:])]))
    columns, rows = math.ceil(6/spacing)+1, math.ceil(3/spacing)+1
    output_positions, weights, provenance, statuses = [], [], [], []
    for wave in range(waves):
        current = replicate + wave
        words = draw_block(current, seed)
        phases = [(word >> 8)/2**24 * spacing for word in words[:2]]
        selected = []
        for index, fixed in enumerate(legacy):
            point = fixed if area else line.interpolate(fixed).coords[0]
            selected.append((point, 0, [index, current, index, 2]))
        candidates = range(columns*rows) if area else range(math.ceil(line.length / spacing))
        accepted = 0
        for ordinal in candidates:
            if area:
                point = (phases[0]+ordinal % columns*spacing, phases[1]+ordinal//columns*spacing)
                if point[0] >= bounds[2] or point[1] >= bounds[3]:
                    continue
                containing = [index for index, triangle in enumerate(triangles) if triangle.covers(Point(point))]
                if not containing or any(math.dist(point, fixed) < spacing for fixed in legacy):
                    continue
                source = containing[0]
            else:
                distance_along = phases[0] + ordinal*spacing
                if distance_along >= line.length:
                    break
                if any(abs(distance_along-fixed) < spacing for fixed in legacy):
                    continue
                point = line.interpolate(distance_along).coords[0]
                source = bisect.bisect_left(cumulative, distance_along)
            selected.append((point, spacing**2 if area else spacing, [source, current, ordinal, 0 if wave == 0 else 1]))
            accepted += 1
        required = len(selected)
        statuses += [min(required, capacity), required, accepted, 2 if required > capacity else 0]
        selected = selected[:capacity]
        selected += [((0, 0), 0, [2**32-1, current, 2**32-1, 2**32-1])] * (capacity-len(selected))
        for point, weight, record in selected:
            output_positions += list(point)
            weights.append(weight)
            provenance += record
    return dict(frameType='area' if area else 'line', positions=list(itertools.chain.from_iterable(vertices)),
                legacySites=list(itertools.chain.from_iterable(legacy)) if area else legacy,
                bounds=bounds, spacing=spacing, seed=seed, replicate=replicate, waves=waves, capacity=capacity,
                expectedPositions=output_positions, expectedWeights=weights,
                expectedProvenance=provenance, expectedStatus=statuses)


Path(__file__).with_suffix('.json').write_text(json.dumps([create_case(False), create_case(True)], indent=2) + '\n')
