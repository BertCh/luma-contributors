// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  getCatalogBinding,
  listCatalogEntries,
  runCatalogOperation
} from '../src/analysis/catalog-bindings';

const lines = {
  type: 'FeatureCollection' as const,
  features: [
    {
      id: 'a',
      geometry: {
        type: 'LineString',
        coordinates: [
          [0, 0],
          [10, 0],
          [10, 10]
        ]
      },
      properties: {}
    }
  ]
};

const points = {
  type: 'FeatureCollection' as const,
  features: [
    {id: 'near', geometry: {type: 'Point', coordinates: [5, 1]}, properties: {}},
    {id: 'far', geometry: {type: 'Point', coordinates: [50, 50]}, properties: {}}
  ]
};

test('snap points binds GPUNetworkSnapping and reports edge, fraction and distance', () => {
  const binding = getCatalogBinding('network-snap-points')!;
  expect(binding.gpuContributorNames).toEqual(['GPUNetworkSnapping']);
  expect(binding.engine).toBe('gpu-or-cpu');
  const result = runCatalogOperation(
    binding,
    {coordinateSystem: 'planar', maximumSnapDistance: 3},
    {lines, points}
  );
  const table = result.tables[0];
  expect(table.columns['status']).toEqual(['snapped', 'unsnapped']);
  expect(table.columns['distance'][0]).toBeCloseTo(1, 9);
  // 5 of 20 length units along the edge.
  expect(table.columns['fraction'][0]).toBeCloseTo(0.25, 9);
  expect(table.columns['edge_source_id'][0]).toBe('a');
  expect(table.columns['edge_id'][1]).toBeNull();
  expect(JSON.parse(String(table.columns['geometry'][0])).coordinates).toEqual([5, 0]);
  expect(JSON.parse(String(table.columns['geometry'][1])).coordinates).toEqual([50, 50]);
  expect(result.diagnostics.some(diagnostic => diagnostic.code === 'point-not-snapped')).toBe(true);
});

test('catalog lists GPUNetworkSnapping as runnable', () => {
  const entry = listCatalogEntries({lensScenarios: []}).find(
    candidate => candidate.identity.exportName === 'GPUNetworkSnapping'
  );
  expect(entry?.runnable).toBe('catalog-binding');
  // Drive-time bands dispatch on the GPU, and snap points also supports the CPU.
  expect(entry?.bindingIds).toContain('curated-drive-time-bands');
  expect(entry?.bindingIds).toContain('network-snap-points');
  expect(entry?.engine).toBe('gpu-or-cpu');
});
