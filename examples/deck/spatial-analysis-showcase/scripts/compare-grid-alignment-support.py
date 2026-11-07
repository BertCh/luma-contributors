# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Compare actual GPU masks and coordinates with independent Rasterio support."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy

module_path = Path(__file__).with_name('diagnose-grid-alignment-support.py')
module_spec = importlib.util.spec_from_file_location('grid_reference', module_path)
reference = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(reference)

input_path = Path(sys.argv[1])
output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else input_path.with_name('support-comparison.json')
require_accurate_projection = '--require-accurate' in sys.argv[3:]
diagnostic = json.loads(input_path.read_text())
gpu_coordinates = numpy.asarray(diagnostic['sourceCoordinates']).reshape(-1, 2)
conceptual_coordinates, _ = reference.make_destination_source_coordinates()
reference_coordinates, reference_projected = reference.make_destination_source_coordinates(diagnostic.get('destinationCenter'))
projected_offsets = numpy.asarray(diagnostic.get('projectedOffsets', [])).reshape(-1, 2)
projection_origin = numpy.asarray(diagnostic.get('projectionOrigin', [0, 0]))
reports = []
for key, actual_mask_values in diagnostic['validity'].items():
    date, method, policy = key.split('-')
    source_validity = reference.load_array(f'validity_{date}.bin', 'u1', (250, 250)) != 0
    source_values = reference.load_array(f'nir_{date}.bin', '<f4', (250, 250))
    source_mask = source_validity & numpy.isfinite(source_values)
    actual_mask = numpy.asarray(actual_mask_values, dtype=bool)
    reference_mask = numpy.array([
        reference.sample_support(source_mask, column, row, method, policy)
        for column, row in reference_coordinates
    ])
    gpu_coordinate_mask = numpy.array([
        reference.sample_support(source_mask, column, row, method, policy)
        for column, row in gpu_coordinates
    ])
    unexplained_indices = numpy.flatnonzero(actual_mask != gpu_coordinate_mask)
    differing_cells = []
    for cell_index in numpy.flatnonzero(actual_mask != reference_mask):
        differing_cells.append({
            'destinationRow': int(cell_index // 64),
            'destinationColumn': int(cell_index % 64),
            'referenceSourceCoordinate': reference_coordinates[cell_index].tolist(),
            'gpuSourceCoordinate': gpu_coordinates[cell_index].tolist(),
            'referenceProjectedCoordinate': reference_projected[cell_index].tolist(),
            'gpuProjectionBeforeAffineReconstruction': (projected_offsets[cell_index] + projection_origin).tolist() if len(projected_offsets) else None,
            'coordinateDifferenceMetres': ((gpu_coordinates[cell_index] - reference_coordinates[cell_index]) * 60).tolist(),
            'referenceSupport': bool(reference_mask[cell_index]),
            'gpuSupport': bool(actual_mask[cell_index]),
            'independentSupportAtGpuCoordinate': bool(gpu_coordinate_mask[cell_index])
        })
    reports.append({
        'case': key,
        'referenceCount': int(reference_mask.sum()),
        'gpuCount': int(actual_mask.sum()),
        'differentCellCount': len(differing_cells),
        'unexplainedGpuMaskCellIndices': unexplained_indices.tolist(),
        'differingCells': differing_cells
    })
result = {
    'method': 'Independent scalar sampling at Rasterio coordinates and actual GPU coordinates; no support-count tolerance',
    'allGpuSupportExplainedByActualCoordinates': all(not report['unexplainedGpuMaskCellIndices'] for report in reports),
    'maximumDeclaredCenterVersusExactUtmCenterDifferenceMetres': float(numpy.max(numpy.abs(reference_coordinates - conceptual_coordinates)) * 60),
    'maximumCoordinateDifferenceMetres': float(numpy.max(numpy.abs(gpu_coordinates - reference_coordinates)) * 60),
    'maximumEuclideanCoordinateDifferenceMetres': float(numpy.max(numpy.linalg.norm(gpu_coordinates - reference_coordinates, axis=1)) * 60),
    'referenceSupportExactlyMatches': all(not report['differentCellCount'] for report in reports),
    'requiredAccuracyMetres': 0.01 if require_accurate_projection else None,
    'cases': reports
}
result['preciseProjectionPassed'] = (
    result['allGpuSupportExplainedByActualCoordinates']
    and result['referenceSupportExactlyMatches']
    and result['maximumEuclideanCoordinateDifferenceMetres'] <= 0.01
)
output_path.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({key: value for key, value in result.items() if key != 'cases'}, indent=2))
for report in reports:
    print(json.dumps({key: value for key, value in report.items() if key != 'differingCells'}))
if not result['allGpuSupportExplainedByActualCoordinates']:
    raise SystemExit('GPU support differs from independent sampling at the actual GPU coordinate')

if require_accurate_projection and not result['preciseProjectionPassed']:
    raise SystemExit('Precise projection exceeds 0.01 m Euclidean error or differs in source support')
