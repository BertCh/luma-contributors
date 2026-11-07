# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent NumPy reductions and separable Gaussian/categorical support reference."""

import json
from pathlib import Path

import numpy

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'public/data/dixie-spectral'
ORACLE = ROOT / '../../../dev-docs/research/showcase-overhaul-2026-10-25/earth-source-oracles.json'


def smooth(values, radius, border, policy):
    sigma = max(0.5, radius / 2)
    kernel = numpy.exp(-(numpy.arange(-radius, radius + 1) ** 2) / (2 * sigma ** 2))
    result = values.copy()
    for axis in (1, 0):
        center_validity = numpy.isfinite(result)
        padding = [(0, 0), (0, 0)]
        padding[axis] = (radius, radius)
        padded = numpy.pad(
            result, padding,
            mode={'nodata': 'constant', 'clamp': 'edge', 'reflect': 'reflect'}[border],
            **({'constant_values': numpy.nan} if border == 'nodata' else {})
        )
        sums = numpy.zeros(values.shape)
        weights = numpy.zeros(values.shape)
        complete = numpy.ones(values.shape, dtype=bool)
        for offset, weight in enumerate(kernel):
            sample = (
                padded[:, offset:offset + values.shape[1]] if axis == 1
                else padded[offset:offset + values.shape[0], :]
            )
            finite = numpy.isfinite(sample)
            sums += numpy.where(finite, sample, 0) * weight
            weights += finite * weight
            complete &= finite
        result = numpy.divide(
            sums, weights, out=numpy.full_like(sums, numpy.nan), where=weights > 0
        )
        # Smoothing preserves missing centers; it is not an inpainting operator.
        result[~center_validity] = numpy.nan
        if policy == 'propagate':
            result[~complete] = numpy.nan
    return result


# A known constant field distinguishes missing-neighbor renormalization from filling.
fixture = numpy.ones((5, 5))
fixture[2, 2] = numpy.nan
ignored_fixture = smooth(fixture, 1, 'clamp', 'ignore-renormalize')
propagated_fixture = smooth(fixture, 1, 'clamp', 'propagate')
assert numpy.isnan(ignored_fixture[2, 2])
assert numpy.isfinite(ignored_fixture).sum() == 24
assert numpy.allclose(ignored_fixture[numpy.isfinite(ignored_fixture)], 1)
assert numpy.array_equal(numpy.isnan(propagated_fixture), numpy.pad(numpy.ones((3, 3), dtype=bool), 1))

records = []
for date in ('before', 'after'):
    values = numpy.fromfile(DATA / f'nir_{date}.bin', dtype='<f4').reshape(250, 250)[50:178, 60:188].astype(float)
    validity = numpy.fromfile(DATA / f'validity_{date}.bin', dtype='u1').reshape(250, 250)[50:178, 60:188] != 0
    values[~validity] = numpy.nan
    finite = values[numpy.isfinite(values)]
    minimum, maximum = finite.min(), finite.max()
    histograms = []
    for bin_count in (32, 128):
        histogram = numpy.bincount(
            numpy.clip(numpy.floor((finite - minimum) / (maximum - minimum) * bin_count), 0, bin_count - 1).astype(int),
            minlength=bin_count
        )
        cumulative = histogram.cumsum()
        quantiles = {}
        for probability in (0, 0.1, 0.5, 0.9, 1):
            index = int(numpy.searchsorted(cumulative, int(numpy.floor((len(finite) - 1) * probability)), side='right'))
            quantiles[str(probability)] = float(
                minimum if probability == 0 else maximum if probability == 1
                else minimum + (index + 0.5) * (maximum - minimum) / bin_count
            )
        histograms.append({'bins': bin_count, 'counts': histogram.tolist(), 'quantiles': quantiles})
    cases = []
    for border in ('nodata', 'clamp', 'reflect'):
        for policy in ('ignore-renormalize', 'propagate'):
            blurred = smooth(values, 3, border, policy)
            blocks = blurred.reshape(32, 4, 32, 4).transpose(0, 2, 1, 3).reshape(1024, 16)
            block_validity = numpy.isfinite(blocks)
            labels = numpy.where(blocks >= 0.3, 3, numpy.where(blocks >= 0.2, 2, numpy.where(blocks >= 0.1, 1, 0)))
            counts = block_validity.sum(axis=1)
            categorical = []
            for selection in ('nearest', 'mode'):
                if selection == 'nearest':
                    chosen = labels[:, 5]
                    accepted = block_validity[:, 5]
                else:
                    frequencies = numpy.stack([
                        numpy.sum(block_validity & (labels == label), axis=1) for label in range(4)
                    ], axis=1)
                    chosen = numpy.argmax(frequencies, axis=1)
                    accepted = counts > 0
                categorical.append({
                    'policy': selection, 'valid': int(accepted.sum()),
                    'support': int(counts[accepted].sum()),
                    'classes': numpy.bincount(chosen[accepted], minlength=4).tolist()
                })
            cases.append({
                'border': border, 'policy': policy,
                'validBlur': int(numpy.isfinite(blurred).sum()), 'categorical': categorical
            })
    records.append({
        'date': date, 'count': len(finite), 'sum': float(finite.sum()),
        'mean': float(finite.mean()), 'minimum': float(minimum), 'maximum': float(maximum),
        'histograms': histograms, 'cases': cases
    })

document = json.loads(ORACLE.read_text())
document['tiledObservations'] = {
    'method': 'Independent NumPy source reductions, rank-midpoint histogram percentiles and horizontal-then-vertical normalized Gaussian (radius 3, sigma 1.5). Every pass preserves invalid center samples. Categories use 4x4 support, upper-left center nearest and lower-label mode ties. Float64 arithmetic on pinned Float32 input; rows 50:178, columns 60:188, strict source QA.',
    'knownMissingCenterFixture': {
        'shape': [5, 5], 'constantValue': 1, 'missingCell': [2, 2], 'radius': 1,
        'border': 'clamp', 'ignoreFiniteCount': 24, 'propagateFiniteCount': 16,
        'propagateMissingExtent': [1, 1, 3, 3], 'passed': True
    },
    'records': records
}
ORACLE.write_text(json.dumps(document, indent=2) + '\n')
print(json.dumps({
    'fixturePassed': True,
    'validBlur': {record['date']: [case['validBlur'] for case in record['cases']] for record in records}
}, indent=2))
