// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {expect, it} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {GPUGlobalPermutationTest} from '../../../src/gpu-spatial-analysis/permutation-inference';
import {
  GPUHotSpotAnalysis,
  GPULocalGeary,
  GPULocalMoran
} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation';
import {
  GPUSpatialLag,
  GPUSpatialWeightsSummary,
  getGPUSpatialWeightsChunkDispatches,
  type GPUChunkedSpatialWeights
} from '../../../src/gpu-spatial-analysis/spatial-weights';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';

const ROWS = 60;
const CHUNK_ROWS = [25, 25, 10];

function createChunked(graph: GPUCommandGraph): GPUChunkedSpatialWeights {
  let rowStart = 0;
  return {
    rowCount: ROWS,
    information: {rowCount: ROWS, transform: 'binary', symmetric: 'unknown'},
    chunks: CHUNK_ROWS.map((rowCount, index) => {
      const slots = rowCount * 4;
      const chunk = {
        rowStart,
        rowCount,
        weights: {
          offsets: createTransientView(graph, `offsets-${index}`, 'uint32', rowCount + 1),
          neighbors: createTransientView(graph, `neighbors-${index}`, 'uint32', slots),
          weights: createTransientView(graph, `weights-${index}`, 'float32', slots)
        }
      };
      rowStart += rowCount;
      return chunk;
    })
  };
}

function createGraph() {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'chunked-consumers'});
  const weights = createChunked(graph);
  const view = <Format extends 'float32' | 'uint32'>(name: string, format: Format, length = ROWS) =>
    createTransientView(graph, name, format, length);
  return {graph, weights, view};
}

it('exposes whole weights as one dispatch without a row offset', () => {
  const {graph, weights} = createGraph();
  const whole = getGPUSpatialWeightsChunkDispatches(weights.chunks[0].weights);
  expect(whole).toHaveLength(1);
  expect(whole[0].constantsWGSL).toBe('');
  expect(whole[0].rowStartWGSL).toBeUndefined();
  const chunked = getGPUSpatialWeightsChunkDispatches(weights);
  expect(chunked.map(dispatch => dispatch.rowStart)).toEqual([0, 25, 50]);
  expect(chunked[1].constantsWGSL).toContain('WEIGHTS_ROW_START: u32 = 25u');
  graph.device.destroy();
});

it('generates one kernel per chunk for each chunk-aware consumer and keeps core binding limits', () => {
  const {graph, weights, view} = createGraph();
  const parameters = view('parameters', 'float32', 8);
  const values = view('values', 'float32');
  const nodes = [
    ...new GPULocalMoran({
      weights,
      values,
      parameters,
      output: {
        zScores: view('z', 'float32'),
        localI: view('i', 'float32'),
        spatialLag: view('lag', 'float32'),
        quadrants: view('q', 'uint32'),
        pValues: view('p', 'float32'),
        neighborCounts: view('n', 'uint32')
      }
    }).getCommandNodes(graph),
    ...new GPUHotSpotAnalysis({
      id: 'gi',
      weights,
      values,
      parameters,
      output: {zScores: view('gz', 'float32'), neighborCounts: view('gn', 'uint32')}
    }).getCommandNodes(graph),
    ...new GPULocalGeary({
      id: 'geary',
      weights,
      values,
      parameters,
      output: {statistic: view('gs', 'float32')}
    }).getCommandNodes(graph),
    ...new GPUSpatialLag({
      id: 'lag',
      weights,
      values,
      normalize: true,
      output: view('lag-out', 'float32')
    }).getCommandNodes(graph),
    ...new GPUSpatialWeightsSummary({
      id: 'summary',
      weights,
      output: {
        statistics: view('s', 'float32', 3),
        counts: view('c', 'uint32', 6),
        cardinality: view('k', 'uint32')
      }
    }).getCommandNodes(graph),
    ...new GPUGlobalPermutationTest({
      id: 'global',
      weights,
      values,
      parameters: view('permutation-parameters', 'uint32', 16),
      statistic: 'moran',
      maximumPermutations: 99,
      output: {summary: view('global-summary', 'float32', 16)}
    }).getCommandNodes(graph)
  ];
  const kernels = getGPUCommandNodeKernelSources(nodes);
  const ids = kernels.map(kernel => kernel.id);
  // Parse the chunk kernels generated here (shared reduction kernels are unchanged).
  for (const {id, source} of kernels) {
    // wgsl_reflect cannot parse the existing `workgroup` local of the pair kernel.
    if (/chunk\d|reverse/.test(id) && !id.includes('pairs')) {
      expect(() => new WgslReflect(source), id).not.toThrow();
    }
  }
  for (const family of [
    'local-moran-neighbors',
    'gi-neighbors',
    'geary-statistic',
    'lag-chunk',
    'summary-rows',
    'global-pairs'
  ]) {
    expect(
      ids.filter(id => id.includes(family) && /chunk\d/.test(id)).length,
      family
    ).toBeGreaterThanOrEqual(CHUNK_ROWS.length);
  }
  for (const {id, source} of kernels) {
    expect(getStorageBindingCount(source), id).toBeLessThanOrEqual(8);
  }
  graph.device.destroy();
});

it('accepts sequential stopping with chunked weights (one gated dispatch per chunk)', () => {
  const {graph, weights, view} = createGraph();
  const createTest = (stopAfterExceedances?: number) =>
    new GPUGlobalPermutationTest({
      id: stopAfterExceedances === undefined ? 'unstopped' : 'stopped',
      weights,
      values: view(`values-${stopAfterExceedances}`, 'float32'),
      parameters: view(`parameters-${stopAfterExceedances}`, 'uint32', 16),
      statistic: 'moran',
      maximumPermutations: 99,
      stopAfterExceedances,
      output: {summary: view(`summary-${stopAfterExceedances}`, 'float32', 16)}
    });
  const countPairKernels = (test: GPUGlobalPermutationTest) =>
    getGPUCommandNodeKernelSources(test.getCommandNodes(graph)).filter(
      ({id}) => /pairs/.test(id) && /chunk\d/.test(id)
    );
  const stopped = countPairKernels(createTest(10));
  // Sequential stopping gates a pair-sum dispatch for every chunk, not just the first.
  for (let chunk = 0; chunk < CHUNK_ROWS.length; chunk++) {
    expect(
      stopped.some(({id}) => id.includes(`chunk${chunk}`)),
      `chunk ${chunk}`
    ).toBe(true);
  }
  expect(stopped.length).toBeGreaterThanOrEqual(countPairKernels(createTest()).length);
  for (const {id, source} of stopped) {
    expect(getStorageBindingCount(source), id).toBeLessThanOrEqual(8);
  }
  graph.device.destroy();
});
