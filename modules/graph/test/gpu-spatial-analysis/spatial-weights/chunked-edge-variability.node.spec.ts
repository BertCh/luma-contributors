// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUGammaStatistic,
  GPULocalSpatialHeteroscedasticity,
  GPULOSHResidualPermutationTest,
  GPUMultivariateLocalGeary,
  GPUSpatialPearson
} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation';
import type {GPUChunkedSpatialWeights} from '../../../src/gpu-spatial-analysis/spatial-weights';
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

it('builds one kernel per chunk for LOSH, multivariate Geary, Pearson and Gamma', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'weights6-chunked'});
  const weights = createChunked(graph);
  const view = <Format extends 'float32' | 'uint32'>(name: string, format: Format, length = ROWS) =>
    createTransientView(graph, name, format, length);
  const nodes = [
    ...new GPULocalSpatialHeteroscedasticity({
      id: 'losh',
      weights,
      values: view('values', 'float32'),
      output: {statistic: view('losh', 'float32'), status: view('losh-status', 'uint32')}
    }).getCommandNodes(graph),
    ...new GPUMultivariateLocalGeary({
      id: 'mgeary',
      weights,
      values: view('values2', 'float32', ROWS * 2),
      variableCount: 2,
      output: {statistic: view('mg', 'float32'), status: view('mg-status', 'uint32')}
    }).getCommandNodes(graph),
    ...new GPUSpatialPearson({
      id: 'pearson',
      weights,
      values: view('x', 'float32'),
      secondValues: view('y', 'float32'),
      output: {statistic: view('pearson-out', 'float32', 1)}
    }).getCommandNodes(graph),
    ...new GPUGammaStatistic({
      id: 'gamma',
      weights,
      values: view('gx', 'float32'),
      output: {statistic: view('gamma-out', 'float32', 1)}
    }).getCommandNodes(graph)
  ];
  const kernels = getGPUCommandNodeKernelSources(nodes);
  const ids = kernels.map(kernel => kernel.id);
  for (const family of [
    'losh-residuals',
    'losh-finish',
    'mgeary-statistic',
    'pearson-moments',
    'pearson-centered',
    'gamma-moments',
    'gamma-centered'
  ]) {
    expect(ids.filter(id => id.includes(family) && /chunk\d/.test(id)).length, family).toBe(
      CHUNK_ROWS.length
    );
  }
  // The final reduction and the shared moments stay single nodes.
  expect(ids.filter(id => id === 'pearson-statistic')).toHaveLength(1);
  expect(ids.filter(id => id.startsWith('losh-mean'))).toHaveLength(1);
  for (const {id, source} of kernels) {
    expect(getStorageBindingCount(source), id).toBeLessThanOrEqual(8);
  }
  graph.device.destroy();
});

it('accepts chunked weights in the LOSH residual permutation test', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'weights6-losh-test'});
  const weights = createChunked(graph);
  const view = <Format extends 'float32' | 'uint32'>(name: string, format: Format, length = ROWS) =>
    createTransientView(graph, name, format, length);
  expect(
    () =>
      new GPULOSHResidualPermutationTest({
        weights,
        values: view('values', 'float32'),
        parameters: view('parameters', 'uint32', 16),
        maximumPermutations: 99,
        alternative: 'greater',
        output: {
          statistic: view('statistic', 'float32'),
          status: view('status', 'uint32'),
          inference: {
            exceedances: view('exceedances', 'uint32'),
            pseudoPValues: view('pseudo', 'float32'),
            status: {overflow: view('overflow', 'uint32', 1)}
          } as never
        }
      })
  ).not.toThrow();
  graph.device.destroy();
});

it('keeps whole weights as a single unsuffixed dispatch', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'weights6-whole'});
  const chunked = createChunked(graph);
  const view = <Format extends 'float32' | 'uint32'>(name: string, format: Format, length = ROWS) =>
    createTransientView(graph, name, format, length);
  const nodes = new GPUSpatialPearson({
    id: 'whole',
    weights: {...chunked.chunks[0].weights, offsets: view('o', 'uint32', ROWS + 1)},
    values: view('x', 'float32'),
    secondValues: view('y', 'float32'),
    output: {statistic: view('out', 'float32', 1)}
  }).getCommandNodes(graph);
  const ids = getGPUCommandNodeKernelSources(nodes).map(kernel => kernel.id);
  expect(ids.filter(id => id.includes('chunk'))).toHaveLength(0);
  graph.device.destroy();
});
