// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  createKeplerEmbeddingFixture,
  validateKeplerEmbeddingRaster,
  RASTER_EMBEDDING_OPERATIONS,
  runKeplerRasterEmbedding
} from '../src/analysis/raster-embeddings';
import {
  RASTER_EMBEDDING_BINDINGS,
  makeRasterEmbeddingCatalogOutput
} from '../src/analysis/catalog-raster-embeddings-bindings';
import {getCatalogBinding, listCatalogEntries} from '../src/analysis/catalog-bindings';
import {createNumericRasterTile} from '../../../examples/deck/kepler-compute/src/raster-tile-ingest';
import {
  prepareRasterEmbeddingTile,
  createRasterEmbeddingsHost
} from '../../../examples/deck/kepler-compute/src/raster-embeddings-host';
import {
  prepareFlowFieldLagrangian,
  projectFlowPosition,
  unprojectFlowPosition
} from '../../../examples/deck/kepler-compute/src/flow-field-lagrangian';
import type {FlowFieldAnalysisSnapshot} from '../../../examples/deck/kepler-compute/src/flow-field-grid';

function constantGrid(): FlowFieldAnalysisSnapshot {
  const texels = new Float32Array(4 * 4 * 4);
  for (let index = 0; index < 16; index++) texels.set([2, -3, 1, 0], index * 4);
  return {
    mode: 'UV',
    columns: 4,
    rows: 4,
    bounds: [10, 59.9, 10.2, 60.1],
    texels,
    minimumSpeed: 0,
    maximumSpeed: 4,
    sampleCount: 16,
    skippedCount: 0
  };
}

test('every embedding operation is registered as a GPU binding with explicit fixture provenance', () => {
  expect(RASTER_EMBEDDING_BINDINGS).toHaveLength(RASTER_EMBEDDING_OPERATIONS.length + 1);
  for (const binding of RASTER_EMBEDDING_BINDINGS) {
    expect(getCatalogBinding(binding.id)?.id).toBe(binding.id);
    expect(binding.engine).toBe('gpu');
    expect(binding.runGpu).toBeTypeOf('function');
    expect(binding.gpuContributorNames.length).toBeGreaterThan(0);
    expect(binding.title).toContain('generated');
  }
});

test('PCA discovery and returned execution evidence distinguish the CPU eigensolver', () => {
  const entry = listCatalogEntries({lensScenarios: []}).find(
    candidate =>
      candidate.identity.package === '@luma-contributors/analysis/raster-analysis' &&
      candidate.identity.exportName === 'getRasterEmbeddingPrincipalComponents'
  );
  expect(entry?.engine).toBe('cpu-reference');
  for (const operation of ['pca', 'projection']) {
    const binding = getCatalogBinding(`raster-embedding-${operation}`)!;
    expect(binding.gpuContributorNames).toEqual([
      'GPURasterEmbeddingCovariance',
      'GPURasterEmbeddingProjection'
    ]);
  }
  const output = makeRasterEmbeddingCatalogOutput({
    width: 1,
    height: 1,
    rowCount: 1,
    layout: 'pixels',
    columns: {},
    contributors: [
      'GPURasterEmbeddingCovariance',
      'getRasterEmbeddingPrincipalComponents',
      'GPURasterEmbeddingProjection'
    ]
  });
  expect(output.dispatchedContributors).toEqual([
    'GPURasterEmbeddingCovariance',
    'GPURasterEmbeddingProjection'
  ]);
  expect(output.stages?.map(stage => stage.execution)).toEqual(['gpu', 'cpu', 'gpu']);
});

test('fixture is unit length and has one explicit QA exclusion', () => {
  const raster = createKeplerEmbeddingFixture();
  validateKeplerEmbeddingRaster(raster);
  for (let pixel = 0; pixel < 64; pixel++)
    expect(Math.hypot(...raster.vectors.slice(pixel * 4, pixel * 4 + 4))).toBeCloseTo(1, 6);
  expect(Array.from(raster.mask!).filter(value => value === 0)).toHaveLength(1);
});

test('band interleave preserves row order, signed bytes and whole-vector validity', () => {
  const tile = createNumericRasterTile({
    id: 'two-band',
    width: 3,
    height: 1,
    extent: [0, 0, 3, 1],
    bands: [
      {values: new Int8Array([127, -128, 4]), noDataValue: -128},
      {values: new Int8Array([-2, 3, -128]), noDataValue: -128}
    ]
  });
  const raster = prepareRasterEmbeddingTile(tile);
  expect(Array.from(raster.vectors)).toEqual([127, -2, -128, 3, 4, -128]);
  expect(Array.from(raster.mask!)).toEqual([1, 0, 0]);
  expect(raster.quantized!.length).toBe(2);
  expect(Array.from(new Int8Array(raster.quantized!.buffer))).toEqual([
    127, -2, -128, 3, 4, -128, 0, 0
  ]);
});

test('nonfinite and Float32-overflow bands invalidate the entire pixel', () => {
  const raster = prepareRasterEmbeddingTile(
    createNumericRasterTile({
      id: 'overflow',
      width: 2,
      height: 1,
      extent: [0, 0, 2, 1],
      bands: [{values: new Float64Array([1e100, 1])}, {values: new Float32Array([2, NaN])}]
    })
  );
  expect(Array.from(raster.mask!)).toEqual([0, 0]);
});

test('shape admission fails before touching a GPU device', async () => {
  const raster = createKeplerEmbeddingFixture();
  await expect(
    runKeplerRasterEmbedding({} as never, {...raster, dimensions: 129}, {operation: 'kmeans'})
  ).rejects.toThrow('1–128 bands');
  expect(() => validateKeplerEmbeddingRaster({...raster, mask: new Uint32Array(1)})).toThrow(
    'mask'
  );
});

test('native tables preserve uint32 codec words and null invalid labels/nonfinite values', () => {
  const output = makeRasterEmbeddingCatalogOutput({
    width: 2,
    height: 1,
    rowCount: 2,
    layout: 'pixels',
    contributors: [],
    columns: {
      label: new Uint32Array([0, 0xffffffff]),
      packed_word: new Uint32Array([0xffffffff, 0]),
      score: new Float32Array([1, NaN])
    }
  });
  expect(output.tables[0].columns['label']).toEqual([0, null]);
  expect(output.tables[0].columns['packed_word']).toEqual([4294967295, 0]);
  expect(output.tables[0].columns['score']).toEqual([1, null]);
});

test('flow adapter resamples endpoint grid to regular local Mercator centres and converts ground velocities', () => {
  const grid = constantGrid();
  const {field, origin} = prepareFlowFieldLagrangian(grid);
  expect(field.extent.slice(0, 2)).toEqual([0, 0]);
  expect(origin).toEqual(projectFlowPosition(10, 59.9));
  for (let row = 0; row < 4; row++)
    for (let column = 0; column < 4; column++) {
      const latitude = unprojectFlowPosition(
        origin[0],
        origin[1] + (row + 0.5) * field.extent[3]
      )[1];
      const factor = 1 / Math.cos((latitude * Math.PI) / 180);
      const index = row * 4 + column;
      expect(field.velocitiesStart[index * 2]).toBeCloseTo(2 * factor, 5);
      expect(field.velocitiesStart[index * 2 + 1]).toBeCloseTo(-3 * factor, 5);
    }
  expect(unprojectFlowPosition(...origin)[0]).toBeCloseTo(10, 10);
});

test('invalid flow corners propagate as NaNs; wrapping extents are refused', () => {
  const grid = constantGrid();
  grid.texels[2] = 0;
  const {field} = prepareFlowFieldLagrangian(grid);
  expect(Number.isNaN(field.velocitiesStart[0])).toBe(true);
  expect(Number.isFinite(field.velocitiesStart.at(-1))).toBe(true);
  expect(() => prepareFlowFieldLagrangian({...grid, bounds: [170, 0, -170, 1]})).toThrow(
    'non-wrapping'
  );
});

test('host refuses unavailable GPU rather than publishing substitute values', async () => {
  let publications = 0;
  const host = createRasterEmbeddingsHost({
    getDevice: () => undefined,
    getSource: () => undefined,
    publishTable: async () => {
      publications++;
    }
  });
  await expect(
    host.run({sourceId: 'fixture', options: {operation: 'similarity'}, demo: true})
  ).rejects.toThrow('WebGPU');
  expect(publications).toBe(0);
  host.destroy();
});
