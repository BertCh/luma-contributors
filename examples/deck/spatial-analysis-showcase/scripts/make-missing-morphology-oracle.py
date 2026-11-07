# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent NumPy grayscale morphology using square or Manhattan support.

Updates only morphology and reconstruction references; preserves fill/GDAL evidence.
Run from the repository root with a Python environment containing NumPy.
"""
import json
from pathlib import Path
import numpy as np


def applyExtremum(values, radius, element, maximum):
    height, width = values.shape
    padded = np.pad(values, radius, mode='edge')
    neighbors = []
    for rowOffset in range(-radius, radius + 1):
        for columnOffset in range(-radius, radius + 1):
            if element == 'cross' and abs(rowOffset) + abs(columnOffset) > radius:
                continue
            neighbors.append(padded[
                radius + rowOffset:radius + rowOffset + height,
                radius + columnOffset:radius + columnOffset + width
            ])
    return (np.maximum.reduce if maximum else np.minimum.reduce)(neighbors)


# Radius-two Manhattan support includes [1,1], excludes [2,1], and contains13 cells.
fixture = np.zeros((7, 7), dtype=float)
fixture[4, 4] = 7
fixture[5, 4] = 11
assert applyExtremum(fixture, 2, 'cross', True)[3, 3] == 7
assert sum(abs(rowOffset) + abs(columnOffset) <= 2
           for rowOffset in range(-2, 3) for columnOffset in range(-2, 3)) == 13
assert applyExtremum(fixture, 2, 'square', True)[3, 3] == 11

source = Path('examples/deck/spatial-analysis-showcase/public/data/dixie-spectral/nir_before.bin')
values = np.fromfile(source, dtype='<f4').reshape(250, 250)[50:114, 101:165].astype(float)
assert np.isfinite(values).all(), 'The measured July reference window must be finite.'
morphology = []
reconstruction = []
for element in ['square', 'cross']:
    for radius in [1, 2, 4]:
        dilated = applyExtremum(values, radius, element, True)
        eroded = applyExtremum(values, radius, element, False)
        outputs = {
            'dilate': dilated,
            'erode': eroded,
            'opening': applyExtremum(eroded, radius, element, True),
            'closing': applyExtremum(dilated, radius, element, False)
        }
        morphology.append({'element': element, 'radius': radius,
                           'probe': {operation: float(output[31, 31])
                                     for operation, output in outputs.items()}})
    for mode in ['opening', 'closing']:
        current = applyExtremum(values, 2, element, mode == 'closing')
        changingRounds = 0
        while True:
            candidate = (np.minimum(applyExtremum(current, 1, element, True), values)
                         if mode == 'opening'
                         else np.maximum(applyExtremum(current, 1, element, False), values))
            if np.array_equal(candidate, current):
                break
            current = candidate
            changingRounds += 1
            assert changingRounds <= 500, 'Independent fixed point did not converge.'
        reconstruction.append({'mode': mode, 'connectivity': 4 if element == 'cross' else 8,
                               'changingRounds': changingRounds, 'probe': float(current[31, 31])})

referencePath = Path('dev-docs/research/showcase-overhaul-2026-10-25/earth-source-oracles.json')
reference = json.loads(referencePath.read_text())
reference['missingObservations']['morphology'] = morphology
reference['missingObservations']['reconstruction'] = reconstruction
reference['missingObservations']['morphologyContract'] = {
    'method': 'Independent NumPy local extrema and fixed-point reconstruction on measured finite July values. Cross means Manhattan distance <= radius, not only axis arms. Clamp border. Probe row31,column31 in64² crop rows50:114,columns101:165.',
    'fixture': 'Radius2 center includes diagonal offset[1,1] value7, excludes offset[2,1] value11;13 included offsets. Square includes11.'
}
referencePath.write_text(json.dumps(reference, indent=2) + '\n')
print(json.dumps({'morphology': morphology, 'reconstruction': reconstruction}, indent=2))
