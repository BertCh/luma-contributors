// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUChunkedSpatialWeightsAlgebra,
  GPUChunkedSpatialWeightsTranspose,
  GPUChunkedSpatialWeightsTransform,
  GPUChunkedSpatialWeightsValidation,
  GPUSpatialWeightsAlgebra,
  type GPUChunkedSpatialWeights,
  type GPUSpatialWeightsTransformOperation
} from '../../../src/gpu-spatial-analysis/spatial-weights/index';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getParsedKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';

const ROWS = 64;
const CHUNK_COUNTS = [1, 2, 4, 8, 16];

function createChunked(graph: GPUCommandGraph, chunkCount: number, name = 'w') {
  const rowsPerChunk = ROWS / chunkCount;
  const create = (suffix: string): GPUChunkedSpatialWeights => ({
    rowCount: ROWS,
    information: {rowCount: ROWS, transform: 'raw', symmetric: 'unknown'},
    chunks: Array.from({length: chunkCount}, (_, index) => {
      const slots = rowsPerChunk * 4;
      return {
        rowStart: index * rowsPerChunk,
        rowCount: rowsPerChunk,
        weights: {
          offsets: createTransientView(
            graph,
            `${name}${suffix}-o${index}`,
            'uint32',
            rowsPerChunk + 1
          ),
          neighbors: createTransientView(graph, `${name}${suffix}-n${index}`, 'uint32', slots),
          weights: createTransientView(graph, `${name}${suffix}-w${index}`, 'float32', slots),
          distances: createTransientView(graph, `${name}${suffix}-d${index}`, 'float32', slots)
        }
      };
    })
  });
  return {left: create('l'), right: create('r'), rowsPerChunk};
}

function view<Format extends 'float32' | 'uint32'>(
  graph: GPUCommandGraph,
  name: string,
  format: Format,
  length: number
) {
  return createTransientView(graph, name, format, length);
}

function outputChunks(graph: GPUCommandGraph, chunked: GPUChunkedSpatialWeights, extra: number) {
  return chunked.chunks.map((chunk, index) => ({
    rowStart: chunk.rowStart,
    rowCount: chunk.rowCount,
    weights: {
      offsets: view(graph, `out-o${index}`, 'uint32', chunk.rowCount + 1),
      neighbors: view(graph, `out-n${index}`, 'uint32', chunk.weights.neighbors.length + extra),
      weights: view(graph, `out-w${index}`, 'float32', chunk.weights.neighbors.length + extra),
      status: {overflow: view(graph, `out-f${index}`, 'uint32', 1)}
    }
  }));
}

type Builder = (graph: GPUCommandGraph, chunkCount: number) => readonly unknown[];

const BUILDERS: Record<string, Builder> = {
  validation: (graph, chunkCount) => {
    const {left} = createChunked(graph, chunkCount);
    return new GPUChunkedSpatialWeightsValidation({
      weights: left,
      output: {status: {invalidCount: view(graph, 'invalid', 'uint32', 1)}}
    }).getCommandNodes(graph);
  },
  transpose: (graph, chunkCount) => {
    const {left} = createChunked(graph, chunkCount);
    const capacity = left.chunks.reduce((sum, chunk) => sum + chunk.weights.neighbors.length, 0);
    return new GPUChunkedSpatialWeightsTranspose({
      weights: left,
      output: {
        offsets: view(graph, 't-o', 'uint32', ROWS + 1),
        neighbors: view(graph, 't-n', 'uint32', capacity),
        weights: view(graph, 't-w', 'float32', capacity),
        distances: view(graph, 't-d', 'float32', capacity),
        asymmetricSlots: view(graph, 't-a', 'uint32', 1)
      }
    }).getCommandNodes(graph);
  },
  union: (graph, chunkCount) => {
    const {left, right} = createChunked(graph, chunkCount);
    return new GPUChunkedSpatialWeightsAlgebra({
      operation: 'union',
      left,
      right,
      output: {chunks: outputChunks(graph, left, left.chunks[0].weights.neighbors.length)}
    }).getCommandNodes(graph);
  },
  subgraph: (graph, chunkCount) => {
    const {left} = createChunked(graph, chunkCount);
    return new GPUChunkedSpatialWeightsAlgebra({
      operation: 'subgraph',
      weights: left,
      mask: view(graph, 'mask', 'uint32', ROWS),
      output: {chunks: outputChunks(graph, left, 0)}
    }).getCommandNodes(graph);
  }
};
for (const operation of [
  'row',
  'binary',
  'kernel',
  'double',
  'variance',
  'symmetrize'
] as GPUSpatialWeightsTransformOperation[]) {
  BUILDERS[`transform-${operation}`] = (graph, chunkCount) => {
    const {left} = createChunked(graph, chunkCount);
    return new GPUChunkedSpatialWeightsTransform({
      operation,
      weights: left,
      bandwidth: 10,
      output:
        operation === 'symmetrize'
          ? left.chunks.map((chunk, index) =>
              view(graph, `tout${index}`, 'float32', chunk.weights.neighbors.length)
            )
          : undefined
    }).getCommandNodes(graph);
  };
}

for (const [name, build] of Object.entries(BUILDERS)) {
  it(`${name}: dispatch count is linear in the chunk count and kernels stay within bindings`, () => {
    const counts = CHUNK_COUNTS.map(chunkCount => {
      const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: `weights7-${name}`});
      const nodes = build(graph, chunkCount);
      const kernels = getGPUCommandNodeKernelSources(nodes);
      // The generated WGSL must parse.
      expect(getParsedKernelSources(nodes).size).toBeGreaterThan(0);
      for (const {id, source} of kernels) {
        expect(getStorageBindingCount(source), id).toBeLessThanOrEqual(8);
      }
      graph.device.destroy();
      return {chunkCount, nodes: nodes.length, kernels: kernels.length};
    });
    // nodes(N) = base + perChunk * N: constant first difference per added chunk.
    const base = counts[0];
    const slopes = counts
      .slice(1)
      .map(entry => (entry.nodes - base.nodes) / (entry.chunkCount - 1));
    for (const slope of slopes) {
      expect(slope, JSON.stringify(counts)).toBeCloseTo(slopes[0], 6);
    }
    // Quadratic growth would make the last slope exceed the first by far.
    expect(slopes[slopes.length - 1]).toBeLessThanOrEqual(slopes[0] + 1e-9);
  });
}

it('keeps whole-CSR algebra free of chunk constants', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'weights7-whole'});
  const csr = (name: string) => ({
    offsets: view(graph, `${name}-o`, 'uint32', ROWS + 1),
    neighbors: view(graph, `${name}-n`, 'uint32', 256),
    weights: view(graph, `${name}-w`, 'float32', 256)
  });
  const nodes = new GPUSpatialWeightsAlgebra({
    operation: 'union',
    left: csr('l'),
    right: csr('r'),
    output: {...csr('out'), status: {overflow: view(graph, 'f', 'uint32', 1)}}
  }).getCommandNodes(graph);
  const kernels = getGPUCommandNodeKernelSources(nodes);
  expect(kernels.length).toBeGreaterThan(0);
  for (const {source} of kernels) {
    expect(source).not.toContain('WEIGHTS_ROW_START');
    expect(source).not.toContain('CHUNK_ROWS');
  }
  graph.device.destroy();
});

it('rejects chunked algebra output that does not match the operand chunks', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'weights7-mismatch'});
  const {left, right} = createChunked(graph, 4);
  const output = outputChunks(graph, left, 4);
  expect(
    () =>
      new GPUChunkedSpatialWeightsAlgebra({
        operation: 'union',
        left,
        right,
        output: {chunks: output.slice(0, 3)}
      })
  ).toThrow();
  graph.device.destroy();
});
