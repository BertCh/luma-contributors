// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {runKeplerLagrangianDiagnostics} from './lagrangian-diagnostics';
import type {CatalogOperationBinding, CatalogRunOutput} from './catalog-binding-types';
import {appendCatalogRow, makeCatalogTable} from './catalog-geojson';
import {
  createKeplerEmbeddingFixture,
  runKeplerRasterEmbedding,
  RASTER_EMBEDDING_OPERATIONS,
  type KeplerEmbeddingRaster,
  type RasterEmbeddingOperation,
  type RasterEmbeddingOptions,
  type RasterEmbeddingResult
} from './raster-embeddings';

export const RASTER_EMBEDDING_CONTRIBUTORS: Record<RasterEmbeddingOperation, readonly string[]> = {
  kmeans: ['GPURasterEmbeddingKMeans'],
  'nearest-prototype': ['GPURasterEmbeddingSimilarity'],
  'nearest-pixels': ['GPURasterEmbeddingNearestPixels'],
  knn: ['GPURasterEmbeddingKNearestClassifier'],
  'linear-probe': ['GPURasterEmbeddingLinearProbe'],
  change: ['GPURasterEmbeddingChange'],
  covariance: ['GPURasterEmbeddingCovariance'],
  pca: [
    'GPURasterEmbeddingCovariance',
    'getRasterEmbeddingPrincipalComponents',
    'GPURasterEmbeddingProjection'
  ],
  projection: [
    'GPURasterEmbeddingCovariance',
    'getRasterEmbeddingPrincipalComponents',
    'GPURasterEmbeddingProjection'
  ],
  dequantize: ['GPURasterEmbeddingQuantize', 'GPURasterEmbeddingDequantize'],
  quantize: ['GPURasterEmbeddingQuantize'],
  pyramid: ['GPURasterEmbeddingPyramidLevel'],
  similarity: ['GPURasterEmbeddingSimilarity'],
  'zone-mean': ['GPURasterEmbeddingZoneMean'],
  'usable-mask': ['GPURasterEmbeddingQuantize']
};

/** Explicit demo defaults. Supervised results describe these prototypes, not a trained earth model. */
export function createKeplerEmbeddingDemoOptions(
  raster: KeplerEmbeddingRaster,
  operation: RasterEmbeddingOperation
): RasterEmbeddingOptions {
  const queries = new Float32Array(raster.dimensions * 2);
  queries[0] = 1;
  queries[raster.dimensions] = -1;
  const zoneRaster = Uint32Array.from({length: raster.width * raster.height}, (_, pixel) =>
    pixel % raster.width < raster.width / 2 ? 0 : 1
  );
  const after = {...raster, vectors: raster.vectors.slice()};
  for (let pixel = 0; pixel < raster.width * raster.height; pixel++)
    if (pixel % raster.width >= raster.width / 2) after.vectors[pixel * raster.dimensions] *= -1;
  return {
    operation,
    queries,
    classCount: 2,
    trainingVectors: queries,
    trainingLabels: new Uint32Array([0, 1]),
    weights: queries,
    bias: new Float32Array(2),
    zoneRaster,
    zoneCount: 2,
    after
  };
}

/** Plain arrays for native table publication; nonfinite values and invalid labels become null. */
export function makeRasterEmbeddingCatalogOutput(result: RasterEmbeddingResult): CatalogRunOutput {
  const columnTypes = Object.fromEntries(
    Object.keys(result.columns).map(name => [name, 'float64' as const])
  );
  const table = makeCatalogTable('embedding-result', 'Embedding result', {
    index: 'int32',
    ...columnTypes
  });
  for (let index = 0; index < result.rowCount; index++) {
    const row: Record<string, number | null> = {index};
    for (const [name, values] of Object.entries(result.columns)) {
      const value = values[index];
      row[name] =
        Number.isFinite(value) &&
        !(values instanceof Uint32Array && value === 0xffffffff && name !== 'packed_word')
          ? value
          : null;
    }
    appendCatalogRow(table, row);
  }
  return {
    tables: [table],
    diagnostics: [],
    summary: {
      width: result.width,
      height: result.height,
      layout: result.layout,
      rowCount: result.rowCount
    },
    dispatchedContributors: result.contributors
  };
}

/** Catalog uses a labelled fixture; actual multiband loaded tiles are available in the raster tools. */
const EMBEDDING_BINDINGS: readonly CatalogOperationBinding[] = RASTER_EMBEDDING_OPERATIONS.map(
  operation => ({
    id: `raster-embedding-${operation}`,
    title: `Embedding ${operation} (generated fixture)`,
    domain: 'raster',
    summary:
      'Runs the embedding contributor on an 8 × 8 generated four-band unit-vector raster. Use Raster embedding tools for loaded multiband raster tiles.',
    capabilityIds: [],
    contributorNames: RASTER_EMBEDDING_CONTRIBUTORS[operation],
    gpuContributorNames: RASTER_EMBEDDING_CONTRIBUTORS[operation],
    engine: 'gpu',
    inputs: [],
    parameters: [],
    outputs: [
      {
        tableId: 'embedding-result',
        label: 'Embedding result',
        geometry: 'none',
        identityColumns: ['index'],
        maximumRows: 64
      }
    ],
    cleanup:
      'Every run destroys its graphs and GPU buffers before returning bounded plain arrays. Native Sources owns the published table.',
    declaredBehavior: [
      'The last pixel is masked; no external datasets or model weights are bundled.',
      'Queries and supervised models are illustrative ± first-band prototypes with labels 0 and 1. Change flips the first band on the right half. Zones are left and right halves.',
      'Nearest prototype uses similarity bestQuery; usable mask uses the public quantizer validity output. PCA uses GPU covariance, the library CPU eigensolver, then GPU projection.',
      'Dequantize demonstrates an AlphaEarth quantize/dequantize roundtrip; no implicit decoding is applied to ordinary float bands.'
    ],
    limitation:
      'Generated demonstration, not model training or an earth embedding dataset. Results are native tables; raster tools admit one loaded tile per run with explicit tile selection.',
    runGpu: async ({device}) => {
      const raster = createKeplerEmbeddingFixture();
      let prefix: string[] = [];
      if (operation === 'dequantize') {
        const encoded = await runKeplerRasterEmbedding(device, raster, {operation: 'quantize'});
        raster.quantized = encoded.columns['packed_word'] as Uint32Array;
        prefix = encoded.contributors;
      }
      const result = await runKeplerRasterEmbedding(
        device,
        raster,
        createKeplerEmbeddingDemoOptions(raster, operation)
      );
      result.contributors.unshift(...prefix);
      return makeRasterEmbeddingCatalogOutput(result);
    }
  })
);

const LAGRANGIAN_BINDING: CatalogOperationBinding = {
  id: 'raster-lagrangian-ftle',
  title: 'Flow map and FTLE (generated strain fixture)',
  domain: 'raster',
  summary:
    'Integrates a planar linear strain velocity raster and publishes final positions and FTLE. Active Flow Field grids are available in Raster embedding and flow tools.',
  capabilityIds: [],
  contributorNames: ['GPULagrangianFlowMap', 'GPUFiniteTimeLyapunovExponent'],
  gpuContributorNames: ['GPULagrangianFlowMap', 'GPUFiniteTimeLyapunovExponent'],
  engine: 'gpu',
  inputs: [],
  parameters: [
    {
      name: 'timeStep',
      label: 'Signed time step (seconds)',
      kind: 'number',
      minimum: -1,
      maximum: 1,
      defaultValue: 0.1
    }
  ],
  outputs: [
    {
      tableId: 'lagrangian-result',
      label: 'Flow map and FTLE',
      geometry: 'none',
      identityColumns: ['seed_index'],
      maximumRows: 36
    }
  ],
  cleanup: 'Graphs and GPU buffers are destroyed before the native table returns.',
  declaredBehavior: [
    'Generated 8 × 8 cell-centered planar velocity grid, u = 0.02 x and v = -0.02 y metres/second. Ten RK2 steps and a regular inset seed grid.',
    'Negative time steps integrate backward; zero duration is refused. Lost seeds and FTLE stencils are null.'
  ],
  limitation:
    'Analytic generated fixture. Use the flow tools for a physical velocity snapshot of a native Flow Field layer, optionally with an aligned end slice.',
  runGpu: async ({device}, _inputs, parameters) => {
    const velocitiesStart = new Float32Array(8 * 8 * 2);
    for (let row = 0; row < 8; row++)
      for (let column = 0; column < 8; column++)
        velocitiesStart.set([0.02 * (column - 3.5), -0.02 * (row - 3.5)], (row * 8 + column) * 2);
    const result = await runKeplerLagrangianDiagnostics(
      device,
      {width: 8, height: 8, extent: [-4, -4, 1, 1], velocitiesStart},
      {timeStep: parameters['timeStep'] as number, stepCount: 10}
    );
    const table = makeCatalogTable('lagrangian-result', 'Flow map and FTLE', {
      seed_index: 'int32',
      final_x: 'float64',
      final_y: 'float64',
      ftle: 'float64'
    });
    for (let index = 0; index < result.ftle.length; index++)
      appendCatalogRow(table, {
        seed_index: index,
        final_x: result.flowValidity[index] ? result.positions[index * 2] : null,
        final_y: result.flowValidity[index] ? result.positions[index * 2 + 1] : null,
        ftle: result.validity[index] ? result.ftle[index] : null
      });
    return {
      tables: [table],
      diagnostics: [],
      summary: {
        rowCount: table.rowCount,
        integrationTime: Math.abs((parameters['timeStep'] as number) * 10)
      },
      dispatchedContributors: result.contributors
    };
  }
};

export const RASTER_EMBEDDING_BINDINGS: readonly CatalogOperationBinding[] = [
  ...EMBEDDING_BINDINGS,
  LAGRANGIAN_BINDING
];

export * from './raster-embeddings';
export * from './lagrangian-diagnostics';
