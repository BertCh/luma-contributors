# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

"""Generate an independent gnomonic/GEOS oracle. Run with /usr/bin/python3."""

import json
import math
import struct
from pathlib import Path
from shapely.geometry import Polygon, box


def packFloat(value):
    return struct.unpack('f', struct.pack('f', value))[0]


def projectSurface(value):
    return (4 * value * value - 1) / 3 if value >= .5 else (1 - 4 * (1 - value) ** 2) / 3


def encodeCell(first, second, level):
    # Independent xy-to-Hilbert-distance rotation algorithm, face zero.
    distance = 0
    size = 2 ** (level - 1) if level else 0
    while size:
        firstBit = int(bool(first & size))
        secondBit = int(bool(second & size))
        distance += size * size * ((3 * firstBit) ^ secondBit)
        if secondBit == 0:
            if firstBit:
                first, second = size - 1 - first, size - 1 - second
            first, second = second, first
        size //= 2
    return (distance * 2 + 1) << (60 - level * 2)


shell = [(-.72, -.61), (.69, -.61), (.69, .73), (-.72, .73)]
hole = [(-.31, -.23), (.34, -.23), (.34, .37), (-.31, .37)]
scenes = []
for withHole in [False, True]:
    rings = [shell, hole] if withHole else [shell]
    positions = []
    projectedRings = []
    for ring in rings:
        projected = []
        for first, second in ring:
            length = math.sqrt(1 + first * first + second * second)
            point = [packFloat(value / length) for value in [1, first, second]]
            positions.extend(point)
            projected.append((point[1] / point[0], point[2] / point[0]))
        projectedRings.append(projected)
    polygon = Polygon(projectedRings[0], projectedRings[1:])
    for mode in ['exterior', 'interior']:
        for minimumLevel in [0, 3]:
            for complement in [False, True]:
                cells = []
                visited = [0]

                def visit(first, second, level):
                    visited[0] += 1
                    scale = 2 ** level
                    rectangle = box(projectSurface(first / scale), projectSurface(second / scale),
                                    projectSurface((first + 1) / scale), projectSurface((second + 1) / scale))
                    # GEOS planar Boolean geometry is exact under the gnomonic chart:
                    # great circles map to straight lines. All scene edges stay on face zero.
                    region = box(-1, -1, 1, 1).difference(polygon) if complement else polygon
                    intersects = region.intersects(rectangle)
                    contained = region.covers(rectangle) and not polygon.boundary.intersects(rectangle)
                    if not intersects:
                        return
                    if level >= minimumLevel and (contained or (mode == 'exterior' and level == 3)):
                        cells.append(encodeCell(first, second, level))
                    elif level < 3:
                        for firstOffset in range(2):
                            for secondOffset in range(2):
                                visit(first * 2 + firstOffset, second * 2 + secondOffset, level + 1)

                visit(0, 0, 0)
                # Other cube faces are disjoint from this polygon; each is visited once.
                visited[0] += 5
                if complement:
                    # With a positive minimum level the other faces refine to that level.
                    visited[0] += 5 * sum(4 ** depth for depth in range(1, minimumLevel + 1))
                    for face in range(1, 6):
                        for position in range(4 ** minimumLevel):
                            cells.append((face << 61) | ((position * 2 + 1) << (60 - 2 * minimumLevel)))
                cells.sort()
                scenes.append(dict(name=f'{withHole}-{mode}-{minimumLevel}-{complement}',
                                   positions=positions, ringOffsets=list(range(0, len(positions) // 3 + 1, 4)),
                                   mode=mode, minimumLevel=minimumLevel, maximumLevel=3, complement=complement,
                                   cells=[format(cell, 'x') for cell in cells], visitedCount=visited[0]))

Path(__file__).with_suffix('.json').write_text(json.dumps(scenes, indent=2) + '\n')
