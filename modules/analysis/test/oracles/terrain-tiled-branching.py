# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""R-FLOW oracle: closed-form plane routing and a NumPy transport linear system.

Run /usr/bin/python3 modules/analysis/test/oracles/terrain-tiled-branching.py.
No production code or networkx dependency is used. D-infinity on z = 100 - 2x - y
splits E/SE by atan(1/2)/(pi/4); the bottom/right boundaries drain E/S respectively.
MFD sends positive downhill slopes to all available neighbours. Quinn weights each
slope by its contour length (1/2 cardinal, sqrt(2)/4 diagonal on this unit grid).
Both MFD cases use an explicit exponent of 1.1, packed to float32 like the GPU.
"""

import json
from pathlib import Path

import numpy


WIDTH = 5
HEIGHT = 4
COUNT = WIDTH * HEIGHT
OFFSETS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
ELEVATION = numpy.array([100 - 2 * (cell % WIDTH) - cell // WIDTH for cell in range(COUNT)])
RUNOFF = numpy.array([(cell % 5) * 0.25 for cell in range(COUNT)])
EXPONENT = float(numpy.float32(1.1))


def make_case(routing):
    fractions = numpy.zeros((COUNT, 8), dtype=numpy.float64)
    for cell in range(COUNT):
        column, row = cell % WIDTH, cell // WIDTH
        if routing == 'd-infinity':
            if column == WIDTH - 1:
                if row < HEIGHT - 1:
                    fractions[cell, 2] = 1
            elif row == HEIGHT - 1:
                fractions[cell, 0] = 1
            else:
                split = numpy.arctan(0.5) / (numpy.pi / 4)
                fractions[cell, 0] = 1 - split
                fractions[cell, 1] = split
        else:
            for direction, (column_offset, row_offset) in enumerate(OFFSETS):
                neighbor_column, neighbor_row = column + column_offset, row + row_offset
                if not (0 <= neighbor_column < WIDTH and 0 <= neighbor_row < HEIGHT):
                    continue
                drop = ELEVATION[cell] - ELEVATION[neighbor_row * WIDTH + neighbor_column]
                if drop <= 0:
                    continue
                distance = numpy.hypot(column_offset, row_offset)
                score = drop / distance
                if routing == 'mfd-quinn':
                    score *= 0.5 if direction % 2 == 0 else numpy.sqrt(2) / 4
                fractions[cell, direction] = score ** EXPONENT
            total = fractions[cell].sum()
            if total > 0:
                fractions[cell] /= total

    # Round inputs, then solve in float64; this is independent of the GPU DAG sweeps
    # and the CPU twin's topological recurrence. (I - transport.T) accumulation = runoff.
    fractions = fractions.astype(numpy.float32).astype(numpy.float64)
    transport = numpy.zeros((COUNT, COUNT))
    for cell in range(COUNT):
        for direction, (column_offset, row_offset) in enumerate(OFFSETS):
            if fractions[cell, direction] > 0:
                receiver = (cell // WIDTH + row_offset) * WIDTH + cell % WIDTH + column_offset
                transport[cell, receiver] = fractions[cell, direction]
    accumulation = numpy.linalg.solve(numpy.eye(COUNT) - transport.T, RUNOFF)
    if not numpy.allclose(accumulation - transport.T @ accumulation, RUNOFF, atol=1e-12):
        raise RuntimeError(f'{routing}: transport residual exceeds tolerance')

    reaches = list(range(COUNT))
    for cell in numpy.argsort(-ELEVATION):
        donors = numpy.flatnonzero(transport[:, cell])
        if len(donors) == 1 and numpy.count_nonzero(transport[donors[0]]) == 1:
            reaches[cell] = reaches[donors[0]]

    return {
        'routing': routing,
        'fractions': fractions.ravel().tolist(),
        'accumulation': accumulation.tolist(),
        'reachIds': reaches
    }


result = {
    'provenance': {
        'generator': 'terrain-tiled-branching.py',
        'numpy': numpy.__version__,
        'method': 'closed-form plane routing; float64 solve of (I - transport.T) accumulation = runoff'
    },
    'width': WIDTH,
    'height': HEIGHT,
    'elevation': ELEVATION.tolist(),
    'runoff': RUNOFF.tolist(),
    'flowExponent': EXPONENT,
    'cases': [make_case(routing) for routing in ('d-infinity', 'mfd-freeman', 'mfd-quinn')]
}
Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
