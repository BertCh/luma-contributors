// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {getGPUPermutationParameterValues} from '../../../src/gpu-spatial-analysis/permutation-inference/index';
import {GPUSpatialEmpiricalBayesRates} from '../../../src/gpu-spatial-analysis/rate-smoothing/index';
import {GPUMultivariateLocalGearyPermutationTest} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/index';
import {
  addSpatialRateRecipe,
  GPUQuantileLISA
} from '../../../src/gpu-spatial-analysis/recipes/index';
import type {
  GPUChunkedSpatialWeights,
  GPUSpatialWeights
} from '../../../src/gpu-spatial-analysis/spatial-weights/index';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';

const ROWS = 60;
const CHUNK_ROWS = [25, 25, 10];

function createWhole(graph: GPUCommandGraph, name: string, rows: number): GPUSpatialWeights {
  return {
    offsets: createTransientView(graph, `${name}-offsets`, 'uint32', rows + 1),
    neighbors: createTransientView(graph, `${name}-neighbors`, 'uint32', rows * 4),
    weights: createTransientView(graph, `${name}-weights`, 'float32', rows * 4),
    information: {rowCount: rows, transform: 'binary', symmetric: 'unknown'}
  };
}

function createChunked(graph: GPUCommandGraph): GPUChunkedSpatialWeights {
  let rowStart = 0;
  return {
    rowCount: ROWS,
    information: {rowCount: ROWS, transform: 'binary', symmetric: 'unknown'},
    chunks: CHUNK_ROWS.map((rowCount, index) => {
      const chunk = {
        rowStart,
        rowCount,
        weights: createWhole(graph, `chunk${index}`, rowCount)
      };
      rowStart += rowCount;
      return chunk;
    })
  };
}

function createGraph() {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'chunked-rates-geary'});
  const view = <Format extends 'float32' | 'uint32'>(name: string, format: Format, length = ROWS) =>
    createTransientView(graph, name, format, length);
  return {graph, view};
}

it('generates one kernel per chunk for the rate, Geary and quantile LISA consumers', () => {
  const {graph, view} = createGraph();
  const weights = createChunked(graph);
  const nodes = [
    ...new GPUSpatialEmpiricalBayesRates({
      id: 'eb',
      weights,
      events: view('events', 'float32'),
      populations: view('populations', 'float32'),
      mask: view('mask', 'uint32'),
      output: {
        spatialRates: view('r', 'float32'),
        smoothedRates: view('s', 'float32'),
        priorMeans: view('m', 'float32'),
        priorVariances: view('v', 'float32')
      }
    }).getCommandNodes(graph),
    ...new GPUMultivariateLocalGearyPermutationTest({
      id: 'exact',
      weights,
      values: view('exact-values', 'float32', ROWS * 2),
      variableCount: 2,
      strategy: 'exact',
      maximumPermutations: 9,
      output: {
        statistic: view('es', 'float32'),
        pValues: view('ep', 'float32'),
        classes: view('ec', 'uint32')
      }
    }).getCommandNodes(graph),
    ...new GPUMultivariateLocalGearyPermutationTest({
      id: 'tiled',
      weights,
      values: view('tiled-values', 'float32', ROWS * 2),
      variableCount: 2,
      strategy: 'tiled',
      rowTileSize: 10,
      maximumPermutations: 9,
      output: {
        statistic: view('ts', 'float32'),
        pValues: view('tp', 'float32'),
        classes: view('tc', 'uint32')
      }
    }).getCommandNodes(graph),
    ...new GPUQuantileLISA({
      id: 'quantile',
      weights,
      values: view('quantile-values', 'float32'),
      mask: view('quantile-mask', 'uint32'),
      quantileParameters: view('quantile-parameters', 'uint32', 2),
      parameters: view(
        'quantile-permutation-parameters',
        'uint32',
        getGPUPermutationParameterValues({seed: 1, permutationCount: 9}).length
      ),
      maximumPermutations: 9,
      maximumNeighbors: 8,
      output: {
        membership: view('membership', 'uint32'),
        boundaries: view('boundaries', 'float32', 2),
        validCount: view('valid-count', 'uint32', 1),
        classes: view('classes', 'uint32'),
        invalidQuantile: view('invalid', 'uint32', 1),
        exceedances: view('exceedances', 'uint32'),
        pseudoPValues: view('pseudo', 'float32'),
        status: {overflow: view('overflow', 'uint32', 1)}
      }
    }).getCommandNodes(graph)
  ];
  addSpatialRateRecipe(graph, {
    id: 'rate',
    weights,
    events: view('rate-events', 'float32'),
    populations: view('rate-populations', 'float32'),
    output: view('rate-output', 'float32')
  });
  const kernels = getGPUCommandNodeKernelSources(nodes);
  const ids = kernels.map(kernel => kernel.id);
  for (const family of ['eb-smooth', 'exact-draws', 'quantile-classes']) {
    expect(
      ids.filter(id => id.includes(family) && /chunk\d/.test(id)).length,
      family
    ).toBeGreaterThanOrEqual(CHUNK_ROWS.length);
  }
  // Tiled tiles cover each chunk with global row bases: chunk 1 starts at row 25.
  // Tiles of 10 rows: 25 + 25 + 10 rows give 3 + 3 + 1 tiles at global row bases.
  expect(ids.filter(id => id.startsWith('tiled-observed-')).sort()).toEqual(
    [0, 10, 20, 25, 35, 45, 50].map(base => `tiled-observed-${base}`).sort()
  );
  for (const {id, source} of kernels) {
    if (/^(eb-smooth|exact-draws|tiled-(observed|simulate)|quantile-classes)/.test(id)) {
      expect(source, id).toContain('WEIGHTS_ROW_START');
    }
    expect(getStorageBindingCount(source), id).toBeLessThanOrEqual(8);
  }
  for (const {id, source} of kernels) {
    if (/^(eb-smooth|quantile-classes)/.test(id) && /chunk\d/.test(id)) {
      expect(() => new WgslReflect(source), id).not.toThrow();
    }
  }
  graph.device.destroy();
});

it('shares one runtime-offset rate kernel between whole CSR and a single CSR chunk', () => {
  const {graph, view} = createGraph();
  const nodes = new GPUSpatialEmpiricalBayesRates({
    id: 'eb',
    weights: createWhole(graph, 'whole', ROWS),
    events: view('events', 'float32'),
    populations: view('populations', 'float32'),
    output: {spatialRates: view('r', 'float32')}
  }).getCommandNodes(graph);
  const kernels = getGPUCommandNodeKernelSources(nodes).filter(kernel =>
    kernel.id.includes('smooth')
  );
  expect(kernels).toHaveLength(1);
  expect(kernels[0].id).toBe('eb-smooth');
  expect(kernels[0].source).toContain('var<private> WEIGHTS_ROW_START: u32;');
  expect(kernels[0].source).not.toMatch(/(?:const|override) WEIGHTS_ROW_START/);
  const chunked = new GPUSpatialEmpiricalBayesRates({
    id: 'eb-chunk',
    weights: {
      rowCount: ROWS,
      chunks: [{rowStart: 0, rowCount: ROWS, weights: createWhole(graph, 'chunk', ROWS)}]
    },
    events: view('chunk-events', 'float32'),
    populations: view('chunk-populations', 'float32'),
    output: {spatialRates: view('chunk-rates', 'float32')}
  }).getCommandNodes(graph);
  const chunkKernels = getGPUCommandNodeKernelSources(chunked).filter(kernel =>
    kernel.id.includes('smooth')
  );
  expect(chunkKernels).toHaveLength(1);
  expect(chunkKernels[0].source).toBe(kernels[0].source);
  expect(chunkKernels[0].constants).toEqual(kernels[0].constants);
  graph.device.destroy();
});
