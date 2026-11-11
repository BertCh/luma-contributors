# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Reference class assignments for GPUClassAssignment `intervalClosure` (math correctness review
2026-11-10, point-process-clustering-03 and fix review fr-models-09).

Writes fixtures/distribution-class-assignment-numpy.json next to this script, consumed by
test/gpu-spatial-analysis/distribution-dynamics/gpu-distribution-dynamics.spec.ts. Run with the
pinned set of requirements.txt (CPython 3.14):
    python modules/analysis/test/oracles/distribution-class-assignment-numpy.py
The fixture was first written on 2026-11-10 with the system CPython 3.9.6 and NumPy 2.0.2, without
mapclassify or giddy. The pinned run of 2026-11-11 reproduces every value and adds the library
checks below.

References. The stored values are NumPy's:
- `'right'`: `numpy.digitize(values, inner, right=True)`, class j holds
  `inner[j - 1] < x <= inner[j]`. This is the rule of mapclassify's `bin1d` (`(x > l) * (x <= r)`
  over upper-bound bins) and therefore of giddy. The script asserts it: mapclassify `bin1d` and
  `UserDefined` on the fixed edges, `Quantiles` on the pooled values (same bins as the NumPy
  percentile edges, same classes), and `giddy.Markov` transitions on the pooled classes all equal
  the `'right'` values, and differ from the `'left'` values where a value is tied at an edge.
- `'left'`: `numpy.digitize(values, inner, right=False)`, class j holds
  `inner[j - 1] <= x < inner[j]` (d3 `bisectRight`, `GPUClassBreaks.classCounts`).
`inner` is the inner edges `e[1..k-1]`, so the result is already clamped to `0..k-1`.

Cases:
- `edges`: fixed edges `[0, 10, 20, 30, 40]` and 400 float32 values: integers and halves in
  `[-5, 45]` (many equal to an edge), both infinities, and values one float32 step either side of
  each inner edge.
- `pooled`: 40 rows over 3 periods of integer "rates" in `0..11`, period-major, four pooled
  quantile classes. Edges are `numpy.percentile(values, [0, 25, 50, 75, 100])` (linear, R-7), the
  rule of `GPUClassBreaks` `'quantile'`; with 120 values the positions are multiples of 0.25, so
  every edge is exact in float32. Classes by both closures, and the transition counts between
  consecutive periods (`counts[from * k + to]`), which is what `giddy.Markov` counts.
"""
import json
import pathlib
import platform

import giddy
import mapclassify
import numpy as np
from giddy.markov import Markov
from mapclassify.classifiers import bin1d

OUTPUT = (
    pathlib.Path(__file__).resolve().parent / 'fixtures' / 'distribution-class-assignment-numpy.json'
)


def encode(value):
    """JSON-safe copy: infinities become strings (decoded by test/oracles/oracle-fixture.ts)."""
    if isinstance(value, dict):
        return {key: encode(entry) for key, entry in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [encode(entry) for entry in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if np.isinf(value):
            return 'Infinity' if value > 0 else '-Infinity'
        return value
    return value


def classify(values, edges):
    inner = np.asarray(edges[1:-1], dtype=np.float64)
    return {
        'right': np.digitize(values, inner, right=True),
        'left': np.digitize(values, inner, right=False),
    }


def count_transitions(classes, rows, periods, class_count):
    counts = np.zeros((class_count, class_count), dtype=np.int64)
    table = np.asarray(classes).reshape(periods, rows)
    for period in range(periods - 1):
        np.add.at(counts, (table[period], table[period + 1]), 1)
    return counts.reshape(-1)


generator = np.random.default_rng(20261110)

# Fixed edges.
fixed_edges = [0.0, 10.0, 20.0, 30.0, 40.0]
steps = []
for edge in fixed_edges[1:-1]:
    steps.extend(
        [np.nextafter(np.float32(edge), np.float32(-np.inf)), np.nextafter(np.float32(edge), np.float32(np.inf))]
    )
fixed_values = np.concatenate(
    [
        (generator.integers(-10, 91, 360) / 2).astype(np.float32),
        np.repeat(np.asarray(fixed_edges[1:-1], dtype=np.float32), 10),
        np.asarray(steps, dtype=np.float32),
        np.asarray([np.inf, -np.inf, 0.0, 40.0], dtype=np.float32),
    ]
).astype(np.float64)
fixed = classify(fixed_values, fixed_edges)
assert int(np.sum(fixed['right'] != fixed['left'])) >= 20, 'the case needs values tied at edges'

# Pooled quantiles.
rows, periods, class_count = 40, 3, 4
pooled_values = generator.integers(0, 12, rows * periods).astype(np.float32).astype(np.float64)
pooled_edges = np.percentile(pooled_values, [0, 25, 50, 75, 100])
assert np.all(pooled_edges == pooled_edges.astype(np.float32)), 'edges must be exact in float32'
pooled = classify(pooled_values, pooled_edges)
tied = int(np.sum(np.isin(pooled_values, pooled_edges[1:-1])))
assert tied >= 3, 'the case needs rows tied at a break'

# The libraries whose rule `'right'` is. They are checks only: the stored values stay NumPy's.
# mapclassify bins are upper bounds; an open last bin keeps `bin1d` inside `0..k-1`.
library_classes, _ = bin1d(fixed_values, np.asarray(fixed_edges[1:-1] + [np.inf]))
assert np.array_equal(library_classes, fixed['right']), 'mapclassify bin1d is the right-closed rule'
assert not np.array_equal(library_classes, fixed['left'])
finite = np.isfinite(fixed_values)
user_defined = mapclassify.UserDefined(
    fixed_values[finite], bins=fixed_edges[1:-1] + [float(fixed_values[finite].max())]
)
assert np.array_equal(user_defined.yb, fixed['right'][finite]), 'mapclassify UserDefined'
quantiles = mapclassify.Quantiles(pooled_values, k=class_count)
assert np.array_equal(quantiles.bins, pooled_edges[1:]), 'mapclassify Quantiles bins are the percentile edges'
assert np.array_equal(quantiles.yb, pooled['right']), 'mapclassify Quantiles classes'
markov = Markov(quantiles.yb.reshape(periods, rows).T, classes=np.arange(class_count), summary=False)
right_counts = count_transitions(pooled['right'], rows, periods, class_count)
assert np.array_equal(np.asarray(markov.transitions).reshape(-1), right_counts), 'giddy.Markov transitions'
assert not np.array_equal(right_counts, count_transitions(pooled['left'], rows, periods, class_count))

result = {
    'generator': 'modules/analysis/test/oracles/distribution-class-assignment-numpy.py',
    'versions': {
        'python': platform.python_version(),
        'numpy': np.__version__,
        'mapclassify': mapclassify.__version__,
        'giddy': giddy.__version__,
    },
    'reference': {
        'edges': {
            'edges': fixed_edges,
            'values': fixed_values,
            'right': fixed['right'],
            'left': fixed['left'],
        },
        'pooled': {
            'rows': rows,
            'periods': periods,
            'classCount': class_count,
            'values': pooled_values,
            'edges': pooled_edges,
            'tiedRowCount': tied,
            'right': {
                'classes': pooled['right'],
                'counts': count_transitions(pooled['right'], rows, periods, class_count),
            },
            'left': {
                'classes': pooled['left'],
                'counts': count_transitions(pooled['left'], rows, periods, class_count),
            },
        },
    },
}
OUTPUT.write_text(json.dumps(encode(result), separators=(',', ':')) + '\n')
print(f'wrote {OUTPUT.name}: {tied} pooled rows tied at a break')
