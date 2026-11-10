# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""RTS-07 reference fixtures: EOF by SVD and lagged correlation in binary64 NumPy.

xeofs (MIT) was not run: it was not installed, to stay within the available disk. The reference is the thin SVD of the weighted anomaly matrix, which is the definition the
method of snapshots reproduces. Sign convention: the largest-magnitude entry of each right singular
vector (first on a tie) is made positive; xeofs' own sign rule is not asserted here.

Output: fixtures/rts07-snapshot-eof-numpy.json. Run with numpy 2.5.4.
"""
import json
import math
import pathlib
import platform

import numpy as np

OUTPUT = pathlib.Path(__file__).resolve().parent / 'fixtures' / 'rts07-snapshot-eof-numpy.json'


def normalized_svd(matrix):
    u, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
    v = vt.T
    for mode in range(v.shape[1]):
        column = v[:, mode]
        largest = int(np.argmax(np.abs(column)))
        if column[largest] < 0:
            v[:, mode] = -column
            u[:, mode] = -u[:, mode]
    return u, singular_values, v


def lagged_correlation(index, field, lag):
    pairs_index, pairs_field = [], []
    for t, value in enumerate(index):
        target = t + lag
        if target < 0 or target >= len(field):
            continue
        if not math.isfinite(value) or not math.isfinite(field[target]):
            continue
        pairs_index.append(value)
        pairs_field.append(field[target])
    if len(pairs_index) < 3:
        return None, len(pairs_index)
    a = np.array(pairs_index) - np.mean(pairs_index)
    b = np.array(pairs_field) - np.mean(pairs_field)
    if a @ a == 0 or b @ b == 0:
        return None, len(pairs_index)
    return float((a @ b) / math.sqrt((a @ a) * (b @ b))), len(pairs_index)


def main():
    generator = np.random.default_rng(707)
    cell_count = 40
    slice_count = 24
    latitudes = np.linspace(-60, 60, cell_count)
    weights = np.cos(np.radians(latitudes))
    weights = weights / np.max(weights)
    time = np.arange(slice_count)
    spatial_one = np.sin(np.linspace(0, math.pi, cell_count))
    spatial_two = np.cos(np.linspace(0, 2 * math.pi, cell_count))
    temporal_one = np.cos(2 * math.pi * time / 12.0)
    temporal_two = np.sin(2 * math.pi * time / 7.0)
    field = (
        1.5 * np.outer(spatial_one, temporal_one)
        + 0.7 * np.outer(spatial_two, temporal_two)
        + 0.15 * generator.standard_normal((cell_count, slice_count))
        + np.linspace(2.0, -1.0, cell_count)[:, None]
    )
    excluded_missing = 7
    excluded_invalid_weight = 13
    field[excluded_missing, 5] = np.nan
    case_weights = weights.copy()
    case_weights[excluded_invalid_weight] = 0.0
    kept = np.array([cell not in (excluded_missing, excluded_invalid_weight) for cell in range(cell_count)])
    weighted = field[kept]
    anomalies = weighted - np.mean(weighted, axis=1, keepdims=True)
    root = np.sqrt(case_weights[kept])
    scaled = anomalies * root[:, None]
    u, singular_values, v = normalized_svd(scaled)
    modes = 3
    weighted_eofs = np.full((cell_count, modes), np.nan)
    value_eofs = np.full((cell_count, modes), np.nan)
    weighted_eofs[kept] = u[:, :modes]
    value_eofs[kept] = u[:, :modes] / root[:, None]
    gram = scaled.T @ scaled

    unweighted_anomalies = anomalies
    unweighted_u, unweighted_singular, unweighted_v = normalized_svd(unweighted_anomalies)
    unweighted_eofs = np.full((cell_count, modes), np.nan)
    unweighted_eofs[kept] = unweighted_u[:, :modes]

    correlation_generator = np.random.default_rng(909)
    cells = 6
    series_length = 36
    index_values = np.cos(2 * math.pi * np.arange(series_length) / 12.0) + 0.2 * correlation_generator.standard_normal(series_length)
    index_values[4] = np.nan
    correlation_field = np.zeros((cells, series_length))
    for cell in range(cells):
        shift = cell - 2
        correlation_field[cell] = (0.5 + 0.1 * cell) * np.roll(np.nan_to_num(index_values), shift) + 0.15 * correlation_generator.standard_normal(series_length)
    correlation_field[1, 10] = np.nan
    correlation_field[3, [20, 21]] = np.nan
    lags = [-3, -1, 0, 2, 5]
    correlations = []
    pair_counts = []
    for cell in range(cells):
        for lag in lags:
            correlation, count = lagged_correlation(index_values, correlation_field[cell], lag)
            correlations.append(correlation)
            pair_counts.append(count)

    def clean(values):
        return [None if not math.isfinite(value) else float(value) for value in values]

    OUTPUT.write_text(
        json.dumps(
            {
                'description': 'Binary64 NumPy SVD references for RTS-07. xeofs was not run; see the generator. Sign rule: largest-magnitude temporal-mode entry positive.',
                'versions': {'python': platform.python_version(), 'numpy': np.__version__},
                'cellCount': cell_count,
                'sliceCount': slice_count,
                'modeCount': modes,
                'values': clean(field.reshape(-1)),
                'weights': case_weights.tolist(),
                'keptCellCount': int(kept.sum()),
                'gram': gram.reshape(-1).tolist(),
                'singularValues': singular_values[:modes].tolist(),
                'explainedVariance': (singular_values[:modes] ** 2 / (slice_count - 1)).tolist(),
                'explainedVarianceRatio': (singular_values[:modes] ** 2 / np.sum(singular_values**2)).tolist(),
                'temporalModes': v[:, :modes].T.reshape(-1).tolist(),
                'weightedEOFs': clean(weighted_eofs.reshape(-1)),
                'valueEOFs': clean(value_eofs.reshape(-1)),
                'unweighted': {
                    'singularValues': unweighted_singular[:modes].tolist(),
                    'temporalModes': unweighted_v[:, :modes].T.reshape(-1).tolist(),
                    'eofs': clean(unweighted_eofs.reshape(-1))
                },
                'laggedCorrelation': {
                    'cellCount': cells,
                    'sliceCount': series_length,
                    'lags': lags,
                    'index': clean(index_values),
                    'values': clean(correlation_field.reshape(-1)),
                    'correlations': correlations,
                    'pairCounts': pair_counts
                }
            }
        )
    )
    print('kept', int(kept.sum()), 'singular', singular_values[:modes], 'ratios', (singular_values[:modes] ** 2 / np.sum(singular_values**2)))


if __name__ == '__main__':
    main()
