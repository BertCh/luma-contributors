# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independently check exported adaptive provider/plan centers against Rasterio."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy

module_path = Path(__file__).with_name('diagnose-grid-alignment-support.py')
module_spec = importlib.util.spec_from_file_location('grid_reference', module_path)
reference = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(reference)

export = json.loads(Path(sys.argv[1]).read_text())
output_path = Path(sys.argv[2])
reference_coordinates, reference_projected = reference.make_destination_source_coordinates(
    export['destinationCenter']
)
origin = numpy.asarray(export['globalNorthwest'])
provider = numpy.asarray([record['providerOffset'] for record in export['centers']]) + origin
plan_offsets = numpy.asarray([record['planOffset'] for record in export['centers']])
plan = plan_offsets + origin

# Simulate the final Float32 storage and the contributor's exact affine expression order.
# This is a CPU prediction; the production GPU comparator remains the acceptance gate.
local_float32 = plan_offsets.astype(numpy.float32)
source_columns = (
    numpy.float32(-60) * local_float32[:, 0]
    - numpy.float32(0) * local_float32[:, 1]
) / numpy.float32(-3600) - numpy.float32(0.5)
source_rows = (
    -numpy.float32(0) * local_float32[:, 0]
    + numpy.float32(60) * local_float32[:, 1]
) / numpy.float32(-3600) - numpy.float32(0.5)
coordinates = numpy.column_stack((source_columns, source_rows))

mask_reports = []
for date in ('before', 'after'):
    mask = reference.load_array(f'validity_{date}.bin', 'u1', (250, 250)) != 0
    source_values = reference.load_array(f'nir_{date}.bin', '<f4', (250, 250))
    mask &= numpy.isfinite(source_values)
    for method in ('nearest', 'bilinear'):
        for policy in ('strict', 'renormalize'):
            reference_mask = numpy.array([
                reference.sample_support(mask, *coordinate, method, policy)
                for coordinate in reference_coordinates
            ])
            simulated_mask = numpy.array([
                reference.sample_support(mask, *coordinate, method, policy)
                for coordinate in coordinates
            ])
            mask_reports.append({
                'case': f'{date}-{method}-{policy}',
                'referenceCount': int(reference_mask.sum()),
                'simulatedCount': int(simulated_mask.sum()),
                'differentCells': int((reference_mask != simulated_mask).sum())
            })

result = {
    'centerCount': len(reference_projected),
    'globalNorthwest': export['globalNorthwest'],
    'sampledFitCertificate': export['plan'],
    'independentProviderMaximumEuclideanErrorMetres': float(
        numpy.linalg.norm(provider - reference_projected, axis=1).max()
    ),
    'independentCpuPlanMaximumEuclideanErrorMetres': float(
        numpy.linalg.norm(plan - reference_projected, axis=1).max()
    ),
    'simulatedLocalFloat32AffineMaximumEuclideanErrorMetres': float(
        numpy.linalg.norm(coordinates - reference_coordinates, axis=1).max() * 60
    ),
    'maskComparisons': mask_reports,
    'cpuOnly': True
}
output_path.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
assert result['centerCount'] == 4096
assert export['globalNorthwest'] == [662800, 4456040]
assert result['independentProviderMaximumEuclideanErrorMetres'] < 0.001
assert result['independentCpuPlanMaximumEuclideanErrorMetres'] < 0.001
assert result['simulatedLocalFloat32AffineMaximumEuclideanErrorMetres'] < 0.01
assert all(not record['differentCells'] for record in mask_reports)
