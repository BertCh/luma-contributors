# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Run: /usr/bin/python3 modules/analysis/test/gpu-raster/raster-time-series/raster-phenology-status-oracles.oracle.py

Execute only the independent committed NumPy threshold-crossing oracle function,
avoiding the reference generator's fixture-writing top-level code. No production
TypeScript or WGSL is read. Print the complete two-season metrics and flag words.
"""
import ast
import json
from pathlib import Path

import numpy

reference = Path(__file__).resolve().parents[2] / 'oracles/rts05-compositing-numpy.py'
function = next(
    node for node in ast.parse(reference.read_text()).body
    if isinstance(node, ast.FunctionDef) and node.name == 'phenology'
)
namespace = {'np': numpy}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(reference), 'exec'), namespace)
cases = [
    ('start-at-other-season', [0, 4, 2, 3, 0], 0.25, 0.75),
    ('end-at-other-season', [0, 3, 2, 4, 0], 0.75, 0.25),
]
for name, curve, start, end in cases:
    status, seasons = namespace['phenology'](
        numpy.arange(5, dtype=float), numpy.array(curve, dtype=float),
        start, end, 0.1, 0.75, 2
    )
    print(json.dumps({'name': name, 'status': status, 'seasons': seasons}))
