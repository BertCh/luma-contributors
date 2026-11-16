// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice} from '../../../analysis/test/utils/require-test-device';
import {RASTER_EMBEDDING_BINDINGS} from '../../src/analysis/catalog-raster-embeddings-bindings';
import {resolveKeplerOperationDescriptor} from '../../src/descriptors/operation-descriptors';
import {resolveKeplerOperationParameters} from '../../src/descriptors/default-parameters';
import {describeKeplerCatalogResult} from '../../src/presentation/result-report';

// Independent publication contracts, not inferred from returned columns or descriptor metadata.
const EXPECTED_TABLES: Readonly<Record<string, {rowCount: number; columns: readonly string[]}>> = {
  'raster-embedding-kmeans': {rowCount: 64, columns: ['index', 'label']},
  'raster-embedding-nearest-prototype': {
    rowCount: 64,
    columns: ['index', 'similarity', 'prototype', 'matches']
  },
  'raster-embedding-nearest-pixels': {
    rowCount: 16,
    columns: ['index', 'pixel_id', 'score', 'query', 'rank']
  },
  'raster-embedding-knn': {rowCount: 64, columns: ['index', 'label', 'confidence']},
  'raster-embedding-linear-probe': {rowCount: 64, columns: ['index', 'label', 'confidence']},
  'raster-embedding-change': {
    rowCount: 64,
    columns: ['index', 'similarity', 'angle', 'magnitude', 'changed']
  },
  'raster-embedding-covariance': {rowCount: 16, columns: ['index', 'covariance']},
  'raster-embedding-pca': {
    rowCount: 64,
    columns: ['index', 'component_1', 'component_2', 'component_3', 'rgb_1', 'rgb_2', 'rgb_3']
  },
  'raster-embedding-projection': {
    rowCount: 64,
    columns: ['index', 'component_1', 'component_2', 'component_3', 'rgb_1', 'rgb_2', 'rgb_3']
  },
  'raster-embedding-dequantize': {
    rowCount: 64,
    columns: ['index', 'usable', 'band_1', 'band_2', 'band_3', 'band_4']
  },
  'raster-embedding-quantize': {rowCount: 64, columns: ['index', 'packed_word']},
  'raster-embedding-pyramid': {
    rowCount: 16,
    columns: [
      'index',
      'usable',
      'valid_count',
      'mean_length',
      'band_1',
      'band_2',
      'band_3',
      'band_4'
    ]
  },
  'raster-embedding-similarity': {
    rowCount: 64,
    columns: ['index', 'similarity', 'prototype', 'matches']
  },
  'raster-embedding-zone-mean': {
    rowCount: 2,
    columns: ['index', 'weight', 'mean_length', 'band_1', 'band_2', 'band_3', 'band_4']
  },
  'raster-embedding-usable-mask': {rowCount: 64, columns: ['index', 'usable']},
  'raster-lagrangian-ftle': {rowCount: 36, columns: ['seed_index', 'final_x', 'final_y', 'ftle']}
};

test.each(
  RASTER_EMBEDDING_BINDINGS
)('$id publishes the declared generated-fixture table and result interpretation', async binding => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {descriptor, source} = resolveKeplerOperationDescriptor(binding);
  const {parameters} = resolveKeplerOperationParameters(binding, descriptor, {});
  const output = await binding.runGpu!({device}, {}, parameters);
  // Include the actual schema and provenance in every mismatch: console output is hidden by
  // the headless runner, so the thrown assertion must carry the diagnostic report itself.
  const evidence = JSON.stringify({
    bindingId: binding.id,
    expected: EXPECTED_TABLES[binding.id],
    tables: output.tables.map(table => ({
      id: table.id,
      rowCount: table.rowCount,
      columns: Object.keys(table.columns)
    })),
    contributors: output.dispatchedContributors,
    summary: output.summary
  });
  expect(source, evidence).toBe('hand-written');
  expect(output.tables, evidence).toHaveLength(1);
  const table = output.tables[0];
  const expected = EXPECTED_TABLES[binding.id];
  expect(table.id, evidence).toBe(descriptor.outputs[0].tableId);
  expect(table.geometryColumn, evidence).toBeUndefined();
  expect(table.rowCount, evidence).toBe(expected.rowCount);
  expect(table.rowCount, evidence).toBeLessThanOrEqual(binding.outputs[0].maximumRows);
  expect(Object.keys(table.columns).sort(), evidence).toEqual([...expected.columns].sort());
  for (const [column, values] of Object.entries(table.columns)) {
    expect(values.length, `${evidence}; column=${column}`).toBe(expected.rowCount);
    expect(
      values.every(
        value => value === null || (typeof value === 'number' && Number.isFinite(value))
      ),
      `${evidence}; column=${column}`
    ).toBe(true);
  }
  for (const contributor of binding.gpuContributorNames) {
    expect(output.dispatchedContributors, evidence).toContain(contributor);
  }
  const identity = binding.outputs[0].identityColumns[0];
  expect(table.columns[identity], evidence).toEqual(
    Array.from({length: expected.rowCount}, (_, index) => index)
  );
  const report = describeKeplerCatalogResult(
    {
      ...output,
      operationId: binding.id,
      engine: 'gpu',
      parameters,
      dispatchedContributors: output.dispatchedContributors ?? []
    },
    binding,
    descriptor
  );
  expect(report.outputs[0].usedTemplate, evidence).toBe(
    descriptor.outputs[0].interpretation!.summary
  );
  expect(report.notes, evidence).toEqual(descriptor.notes);
  if (binding.id === 'raster-embedding-usable-mask') {
    expect(table.columns['usable'], evidence).toEqual([...new Array(63).fill(1), 0]);
  }
  if (binding.id === 'raster-embedding-knn' || binding.id === 'raster-embedding-linear-probe') {
    expect(table.columns['label'][63], evidence).toBeNull();
  }
  if (binding.id === 'raster-embedding-zone-mean') {
    expect(table.columns['weight'], evidence).toEqual([32, 31]);
  }
  if (binding.id === 'raster-embedding-pyramid') {
    expect(table.columns['valid_count'][15], evidence).toBe(3);
  }
});

test.each([
  0.1, -0.1
])('catalog FTLE follows analytic strain for signed time step %s', async timeStep => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const binding = RASTER_EMBEDDING_BINDINGS.find(
    candidate => candidate.id === 'raster-lagrangian-ftle'
  )!;
  const output = await binding.runGpu!({device}, {}, {timeStep});
  const table = output.tables[0];
  const multiplier = (rate: number) => (1 + rate * timeStep + (rate * timeStep) ** 2 / 2) ** 10;
  for (let index = 0; index < 36; index++) {
    const seedX = (index % 6) - 2.5;
    const seedY = Math.floor(index / 6) - 2.5;
    const evidence = JSON.stringify({
      timeStep,
      index,
      finalX: table.columns['final_x'][index],
      finalY: table.columns['final_y'][index],
      ftle: table.columns['ftle'][index]
    });
    expect(table.columns['final_x'][index], evidence).toBeCloseTo(seedX * multiplier(0.02), 5);
    expect(table.columns['final_y'][index], evidence).toBeCloseTo(seedY * multiplier(-0.02), 5);
    expect(table.columns['ftle'][index], evidence).toBeCloseTo(0.02, 5);
  }
  expect(output.summary['integrationTime']).toBe(Math.abs(timeStep * 10));
});
