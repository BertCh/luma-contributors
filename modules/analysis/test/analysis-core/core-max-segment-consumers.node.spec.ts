// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GEOMETRY_CASES} from './kernel-audit/cases-geometry';
import {POINTS_AND_REGIONS_CASES} from './kernel-audit/cases-points-and-regions';
import {RASTER_CASES} from './kernel-audit/cases-raster';
import {CELLS_AND_JOINS_CASES} from './kernel-audit/cases-cells-and-joins';
import {auditKernelCase, getKernelAuditResultFailures} from './kernel-audit/kernel-audit-runner';

const requested = [
  ['GPULineDensity', 'grid at the maximum'],
  ['GPUArealInterpolation', 'cells and pairs at the maximum'],
  [
    'GPUDBSCAN',
    'rows and clusters, index grid at the maximum sorted below the segment-sum dispatch limit'
  ],
  ['GPURasterJoin', 'maximum points and zones, every output'],
  ...['uniform', 'geographic'].flatMap(mode => [
    ['GPURasterZonalStatistics', `maximum grid and zones, sorted sums, ${mode} cells`],
    [
      'GPURasterZonalStatistics',
      `maximum grid and zones without moments, sorted sums, ${mode} cells`
    ]
  ]),
  ['GPUTrajectoryMetrics', 'rows and tracks at the binding limit']
];
const available = [
  ...GEOMETRY_CASES,
  ...POINTS_AND_REGIONS_CASES,
  ...RASTER_CASES,
  ...CELLS_AND_JOINS_CASES
];

it.each(
  requested
)('review B4 accepted maximum %s %s retains an auditable reduction', (exportName, name) => {
  const auditCase = available.find(
    candidate => candidate.exportName === exportName && candidate.name === name
  );
  expect(auditCase, `${exportName}: ${name}`).toBeDefined();
  if (!auditCase) throw new Error('Missing maximum-capacity case');
  const result = auditKernelCase(auditCase);
  expect(result.rejected).toBe(false);
  expect(getKernelAuditResultFailures(result)).toEqual([]);
});
