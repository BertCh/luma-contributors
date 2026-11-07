# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Regenerate only the independent median-preparation reference from pinned Sentinel data."""
import json
from pathlib import Path

import numpy

repository_root = Path(__file__).resolve().parents[4]
data_directory = Path(__file__).resolve().parents[1] / 'public/data/dixie-spectral'
july = numpy.fromfile(data_directory / 'nir_before.bin', dtype='<f4').reshape(250, 250)[50:114, 101:165]
september = numpy.fromfile(data_directory / 'nir_after.bin', dtype='<f4').reshape(250, 250)[50:114, 101:165]
rows, columns = numpy.indices((64, 64))
masks = {
    'isolated': ((rows * 64 + columns) * 17 + 13) % 31 == 0,
    'block': (rows >= 27) & (rows <= 35) & (columns >= 27) & (columns <= 35),
    'september': ~numpy.isfinite(september),
}
surfaces = {}
for scenario, missing in masks.items():
    source = september if scenario == 'september' else july
    for distance in (2, 8):
        filled_surface = numpy.where(missing, numpy.nan, source).astype('float32')
        for row, column in zip(*numpy.where(missing)):
            minimum_row, maximum_row = max(0, row - distance), min(64, row + distance + 1)
            minimum_column, maximum_column = max(0, column - distance), min(64, column + distance + 1)
            window = numpy.s_[minimum_row:maximum_row, minimum_column:maximum_column]
            squared_distance = (rows[window] - row) ** 2 + (columns[window] - column) ** 2
            support = (~missing[window]) & (squared_distance > 0) & (squared_distance <= distance ** 2)
            if support.any():
                # Row-major enumeration preserves the stated first original-donor tie.
                donor_index = numpy.argmin(squared_distance[support])
                filled_surface[row, column] = source[window][support][donor_index]
        surfaces[(scenario, distance)] = filled_surface
records = []

for scenario in ['isolated', 'block', 'september']:
    for distance in [2, 8]:
        values = surfaces[(scenario, distance)].astype('float32')
        valid = numpy.isfinite(values)
        mask = masks[scenario]
        original = int((~mask).sum())
        filled = int((mask & valid).sum())

        for radius in [1, 2]:
            offsets = range(-radius, radius + 1)
            for border_mode in ['clamp', 'nodata']:
                samples = []
                sample_validity = []
                for row_offset in offsets:
                    for column_offset in offsets:
                        source_rows = numpy.arange(64) + row_offset
                        source_columns = numpy.arange(64) + column_offset
                        inside = (
                            (source_rows[:, None] >= 0)
                            & (source_rows[:, None] < 64)
                            & (source_columns[None, :] >= 0)
                            & (source_columns[None, :] < 64)
                        )
                        if border_mode == 'clamp':
                            source_rows = numpy.clip(source_rows, 0, 63)
                            source_columns = numpy.clip(source_columns, 0, 63)
                            neighbor = values[numpy.ix_(source_rows, source_columns)]
                            neighbor_validity = numpy.isfinite(neighbor)
                        else:
                            safe_rows = numpy.clip(source_rows, 0, 63)
                            safe_columns = numpy.clip(source_columns, 0, 63)
                            neighbor = values[numpy.ix_(safe_rows, safe_columns)]
                            neighbor_validity = numpy.isfinite(neighbor) & inside
                            neighbor = numpy.where(neighbor_validity, neighbor, numpy.nan)
                        samples.append(neighbor)
                        sample_validity.append(neighbor_validity)

                sample_stack = numpy.stack(samples)
                validity_stack = numpy.stack(sample_validity)
                counts = validity_stack.sum(axis=0)
                ordered = numpy.sort(sample_stack, axis=0)
                median = numpy.take_along_axis(
                    ordered, numpy.minimum(counts // 2, len(samples) - 1)[None, :, :], axis=0
                )[0]

                for policy in ['ignore', 'propagate']:
                    accepted = valid & (counts > 0)
                    if policy == 'propagate':
                        accepted &= counts == len(samples)
                    output = numpy.where(accepted, median, numpy.nan)
                    evaluated = accepted & mask
                    errors = output[evaluated].astype('float64') - july[evaluated].astype('float64')
                    records.append(
                        {
                            'scenario': scenario,
                            'distance': distance,
                            'radius': radius,
                            'policy': policy,
                            'borderMode': border_mode,
                            'original': original,
                            'filled': filled,
                            'filtered': int((accepted & (output != values)).sum()),
                            'unresolved': int((~accepted).sum()),
                            'evaluated': int(evaluated.sum()),
                            'rmse': float(numpy.sqrt(numpy.mean(errors**2)))
                            if scenario != 'september' and evaluated.any()
                            else None,
                            'mae': float(numpy.mean(abs(errors)))
                            if scenario != 'september' and evaluated.any()
                            else None,
                            'probe': float(output[31, 31]) if accepted[31, 31] else None,
                        }
                    )

path = repository_root / 'dev-docs/research/showcase-overhaul-2026-10-25/earth-source-oracles.json'
document = json.loads(path.read_text())
document['surfacePreparation'] = {
    'method': (
        'Independent original-donor nearest fill from known oracle followed by NumPy upper-median '
        'order statistic; clamp repeats edge samples, nodata excludes out-of-grid samples; '
        'center validity required; ignore omits invalid samples, propagate invalidates if any '
        'sample is missing. Exact float32 nearest values; 64×64 source.'
    ),
    'records': records,
}
path.write_text(json.dumps(document, indent=2) + '\n')
print(json.dumps({'records': len(records), 'scenarios': 3, 'borderModes': 2}, indent=2))
