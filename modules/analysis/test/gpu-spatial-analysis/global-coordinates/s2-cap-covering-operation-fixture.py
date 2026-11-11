# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

"""Independent constrained-optimization oracle. Run with /usr/bin/python3."""

import json
import math
import struct
from pathlib import Path
import numpy
from scipy.optimize import minimize_scalar


def packFloat(value):
    return struct.unpack('f', struct.pack('f', value))[0]


def projectSurface(value):
    return (4 * value * value - 1) / 3 if value >= .5 else (1 - 4 * (1 - value) ** 2) / 3


def getFaceVector(face, first, second):
    return numpy.array([(1, first, second), (-first, 1, second), (-first, -second, 1),
                     (-1, -second, -first), (second, -1, -first), (second, first, -1)][face], dtype=float)


def encodeCell(face, first, second, level):
    if face % 2:
        first, second = second, first
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
    return (face << 61) | ((distance * 2 + 1) << (60 - level * 2))


caps = []
for vector, radius in [([1, 0, 0], 0), ([1, .3, .2], .42), ([-.4, 1, -.5], .75),
                       ([-.2, -.4, -1], 1.05), ([0, 0, 1], 2.3)]:
    vector = numpy.array(vector) / numpy.linalg.norm(vector)
    caps.extend([packFloat(value) for value in vector] + [packFloat(radius)])

scenes = []
for level in [0, 2, 3]:
    for mode in ['exterior', 'interior']:
        selectedCaps = []
        for cap in range(len(caps) // 4):
            center = numpy.array(caps[cap * 4:cap * 4 + 3])
            center /= numpy.linalg.norm(center)
            radius = caps[cap * 4 + 3]
            selected = []
            for face in range(6):
                for first in range(2 ** level):
                    for second in range(2 ** level):
                        lower = [projectSurface(first / 2 ** level), projectSurface(second / 2 ** level)]
                        upper = [projectSurface((first + 1) / 2 ** level), projectSurface((second + 1) / 2 ** level)]
                        corners = [getFaceVector(face, lower[0], lower[1]), getFaceVector(face, upper[0], lower[1]),
                                   getFaceVector(face, upper[0], upper[1]), getFaceVector(face, lower[0], upper[1])]
                        extrema = []
                        for edge in range(4):
                            start = corners[edge]
                            finish = corners[(edge + 1) % 4]

                            def cosine(parameter):
                                point = start + parameter * (finish - start)
                                return numpy.dot(center, point) / numpy.linalg.norm(point)

                            extrema.extend([cosine(0), cosine(1)])
                            # Optimize along the straight unnormalized cube edge. This parameter
                            # covers the complete great-circle edge after radial normalization.
                            for sign in [-1, 1]:
                                optimum = minimize_scalar(lambda parameter: sign * cosine(parameter), bounds=(0, 1), method='bounded', options={'xatol': 1e-14})
                                extrema.append(cosine(optimum.x))
                        # Include stationary interior extrema at the center and antipode.
                        for sign in [-1, 1]:
                            point = sign * center
                            if all(numpy.dot(numpy.cross(corners[edge], corners[(edge + 1) % 4]), point) >= 0 for edge in range(4)):
                                extrema.append(sign)
                        nearest = math.acos(max(-1, min(1, max(extrema))))
                        farthest = math.acos(max(-1, min(1, min(extrema))))
                        keep = farthest <= radius - 1e-5 if mode == 'interior' else nearest <= radius + 1e-5
                        if keep:
                            selected.append(encodeCell(face, first, second, level))
            selectedCaps.append([format(cell, 'x') for cell in sorted(selected)])
        scenes.append(dict(level=level, mode=mode, caps=caps, cells=selectedCaps))

Path(__file__).with_suffix('.json').write_text(json.dumps(scenes, indent=2) + '\n')
