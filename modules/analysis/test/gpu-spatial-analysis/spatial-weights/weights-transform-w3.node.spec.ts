// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUSpatialWeightsTransform,
  getGPUSpatialWeightsKernelValue
} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-transform';
import {GPUChunkedSpatialWeightsTransform} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-chunked-spatial-weights-transform';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {computeGlobalSpatialStatisticsOracle} from '../global-spatial-statistics/global-spatial-statistics-oracle';
import {computeSpatialWeightsValidationOnCPU} from '../../../src/gpu-spatial-analysis/spatial-weights/cpu-spatial-weights-validation-operation';
import {GPUChunkedSpatialWeightsTranspose} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-chunked-spatial-weights-transpose';
import {readFileSync} from 'node:fs';
import {transpileModule, ModuleKind} from 'typescript';

function createSetup() {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `transform-${serial++}`, format, length);
  const weights = {
    rowCount: 6,
    chunks: [0, 2, 4].map(rowStart => ({
      rowStart,
      rowCount: 2,
      weights: {
        offsets: view('uint32', 3),
        neighbors: view('uint32', 2),
        weights: view('float32', 2),
        distances: view('float32', 2)
      }
    }))
  };
  return {graph, view, weights};
}

for (const chunked of [false, true]) {
  for (const unusedPort of ['neighbor', 'status'] as const) {
    it(`B2 retains ownership validation for an unused ${unusedPort} view, chunked=${chunked}`, () => {
      const {graph, view} = createSetup();
      const foreign = createSetup();
      try {
        const weights = {
          offsets: view('uint32', 3),
          neighbors: unusedPort === 'neighbor' ? foreign.view('uint32', 2) : view('uint32', 2),
          weights: view('float32', 2),
          status: unusedPort === 'status' ? {requiredCount: foreign.view('uint32', 1)} : undefined
        };
        const transform = chunked
          ? new GPUChunkedSpatialWeightsTransform({
              operation: 'row',
              weights: {rowCount: 2, chunks: [{rowStart: 0, rowCount: 2, weights}]}
            })
          : new GPUSpatialWeightsTransform({operation: 'row', weights});
        expect(() => transform.getCommandNodes(graph)).toThrow(/target graph/);
      } finally {
        graph.device.destroy();
        foreign.graph.device.destroy();
      }
    });
  }
  for (const operation of [
    'row',
    'binary',
    'kernel',
    'double',
    'variance',
    'average-reciprocal'
  ] as const) {
    it(`B2 ${operation} preflights only bound views, chunked=${chunked}`, () => {
      const {graph, view} = createSetup();
      Object.assign(graph.device.limits, {maxStorageBufferBindingSize: 16});
      const backing = view('uint32', 5);
      const weights = {
        offsets: view('uint32', 3),
        neighbors: graph.createDataView(backing.buffer, {
          format: 'uint32',
          length: 4,
          byteOffset: 4
        }),
        weights: view('float32', 4),
        distances: view('float32', 4)
      };
      const output = view('float32', 4);
      try {
        const transform = chunked
          ? new GPUChunkedSpatialWeightsTransform({
              operation,
              bandwidth: 1,
              weights: {rowCount: 2, chunks: [{rowStart: 0, rowCount: 2, weights}]},
              output: [output]
            })
          : new GPUSpatialWeightsTransform({operation, bandwidth: 1, weights, output});
        if (operation === 'average-reciprocal') {
          expect(() => transform.getCommandNodes(graph)).toThrow(/20.*16/);
        } else {
          expect(() => transform.getCommandNodes(graph)).not.toThrow();
        }
      } finally {
        graph.device.destroy();
      }
    });
  }
}

it('B4 documented transpose admits independently sized output chunks above the total binding limit', () => {
  const {graph, view} = createSetup();
  Object.assign(graph.device.limits, {maxStorageBufferBindingSize: 32});
  const input = {
    rowCount: 4,
    chunks: [0, 2].map(rowStart => ({
      rowStart,
      rowCount: 2,
      weights: {
        offsets: view('uint32', 3),
        neighbors: view('uint32', 8),
        weights: view('float32', 8)
      }
    }))
  };
  const page = readFileSync(
    new URL(
      '../../../../../docs/api/gpu-spatial-analysis/chunked-spatial-weights-transpose.md',
      import.meta.url
    ),
    'utf8'
  );
  const snippet = page.split('```ts\n')[1].split('```')[0];
  const javascript = transpileModule(snippet, {
    compilerOptions: {module: ModuleKind.CommonJS}
  }).outputText;
  const addTranspose = new Function('require', 'exports', `${javascript}\nreturn addTranspose;`)(
    (name: string) => {
      if (name === '@luma.gl/graph/gpu-core') return {createTransientView};
      if (name === '@luma-contributors/analysis/gpu-spatial-analysis')
        return {GPUChunkedSpatialWeightsTranspose};
      throw new Error(`unexpected snippet import ${name}`);
    },
    {}
  );
  try {
    const {output} = addTranspose(graph, input, 4, [8, 8]);
    expect(
      output.chunks.map(
        (chunk: {weights: {neighbors: {length: number}}}) => chunk.weights.neighbors.length
      )
    ).toEqual([8, 8]);
  } finally {
    graph.device.destroy();
  }
});

it('B6 packed status refusal names omitted cardinalities and the retained-count offset', () => {
  const {graph, weights, view} = createSetup();
  try {
    expect(
      () =>
        new GPUChunkedSpatialWeightsTranspose({
          weights,
          output: {
            offsets: view('uint32', 7),
            neighbors: view('uint32', 6),
            weights: view('float32', 6),
            status: {overflow: view('uint32', 1)}
          }
        })
    ).toThrow(/except count and requiredCount.*offsets\[columnCount\]/);
  } finally {
    graph.device.destroy();
  }
});

it('S2 UST-13 row order changes Moran despite identical valid CSR dimensions', () => {
  const weights = {
    offsets: Uint32Array.of(0, 1, 3, 5, 6),
    neighbors: Uint32Array.of(1, 0, 2, 1, 3, 2),
    weights: Float32Array.of(1, 0.5, 0.5, 0.5, 0.5, 1)
  };
  // Exact rational calculation: centered values have sum of squares 115/4.
  // Ordered-pair cross-products are 67/8 and -101/8; S0=n=4, giving 67/230 and -101/230.
  for (const [values, expected] of [
    [Float32Array.of(1, 2, 4, 8), 0.29130434782608694],
    [Float32Array.of(1, 4, 2, 8), -0.4391304347826087]
  ] as const) {
    const actual = computeGlobalSpatialStatisticsOracle({weights, values, standardized: true}).moran
      .statistic;
    expect(Number.isFinite(actual)).toBe(true);
    expect(actual).toBeCloseTo(expected, 12);
  }
});

it('UST-02 masked Moran distinguishes the original and transformed descriptions of shared bytes', () => {
  const weights = {
    offsets: Uint32Array.of(0, 1, 3, 5, 6),
    neighbors: Uint32Array.of(1, 0, 2, 1, 3, 2),
    weights: Float32Array.of(1, 0.5, 0.5, 0.5, 0.5, 1)
  };
  const values = Float32Array.of(1, 2, 4, 8);
  const mask = Uint32Array.of(1, 1, 1, 0);
  // Included deviations (-4,-1,5)/3 give sum(c_i^2)=14/3.
  // As given: S0=5/2 and weighted cross-products=1/9 => 1/35.
  // Re-standardized after masking: S0=3 and cross-products=-1/6 => -1/28.
  for (const [standardized, expected] of [
    [false, 1 / 35],
    [true, -1 / 28]
  ] as const) {
    const actual = computeGlobalSpatialStatisticsOracle({weights, values, mask, standardized}).moran
      .statistic;
    expect(Number.isFinite(actual)).toBe(true);
    expect(actual).toBeCloseTo(expected, 12);
  }
});

it('ING-15 parallel roads need an explicit aggregation and influence policy before CSR validation', () => {
  const parallelRoads = computeSpatialWeightsValidationOnCPU(
    Uint32Array.of(0, 2, 2),
    Uint32Array.of(1, 1),
    Float32Array.of(2, 3)
  );
  expect(Array.from(parallelRoads.flags)).toEqual([4, 0]);
  expect(Array.from(parallelRoads.invalidCount)).toEqual([1]);
  // Choose the minimum travel cost (2) per directed pair, then binary adjacency influence (1).
  // The road costs stay in the network: they are not statistical influence weights.
  const aggregated = computeSpatialWeightsValidationOnCPU(
    Uint32Array.of(0, 1, 1),
    Uint32Array.of(1),
    Float32Array.of(1)
  );
  expect(Array.from(aggregated.flags)).toEqual([0, 0]);
  expect(Array.from(aggregated.invalidCount)).toEqual([0]);
});

for (const operation of ['row', 'binary', 'kernel'] as const) {
  it(`LUMA-104 ${operation} admits fitting chunks when the global row array exceeds the binding limit`, () => {
    const {graph, weights} = createSetup();
    Object.assign(graph.device.limits, {maxStorageBufferBindingSize: 16});
    try {
      expect((weights.rowCount + 1) * 4).toBeGreaterThan(16);
      const transform = new GPUChunkedSpatialWeightsTransform({operation, weights, bandwidth: 1});
      expect(transform.getCommandNodes(graph)).toHaveLength(operation === 'kernel' ? 6 : 3);
    } finally {
      graph.device.destroy();
    }
  });
}

it('LUMA-104 average-reciprocal still refuses global scratch above the binding limit before allocation', () => {
  const {graph, view, weights} = createSetup();
  Object.assign(graph.device.limits, {maxStorageBufferBindingSize: 16});
  try {
    const transform = new GPUChunkedSpatialWeightsTransform({
      operation: 'average-reciprocal',
      weights,
      output: weights.chunks.map(() => view('float32', 2))
    });
    expect(() => transform.getCommandNodes(graph)).toThrow(/row scratch.*28.*16/);
    expect(() =>
      createTransientView(graph, `${transform.id}-reverse-offsets`, 'uint32', 7)
    ).not.toThrow();
  } finally {
    graph.device.destroy();
  }
});

it('NAM-16 quartic and bisquare retain spatial-weights normalization in both constructors and the host evaluator', () => {
  const {graph, weights} = createSetup();
  try {
    for (const kernel of ['quartic', 'bisquare'] as const) {
      // Hand calculation: (15 / 16) * (1 - (1 / 2)^2)^2 = 135 / 256.
      expect(getGPUSpatialWeightsKernelValue(kernel, 0.5)).toBe(0.52734375);
      expect(
        () =>
          new GPUSpatialWeightsTransform({
            weights: weights.chunks[0].weights,
            operation: 'kernel',
            bandwidth: 1,
            kernel
          })
      ).not.toThrow();
      expect(
        () =>
          new GPUChunkedSpatialWeightsTransform({
            weights,
            operation: 'kernel',
            bandwidth: 1,
            kernel
          })
      ).not.toThrow();
    }
    const unknown = 'unknown' as 'quartic';
    for (const z of [0.5, NaN]) {
      expect(() => getGPUSpatialWeightsKernelValue(unknown, z)).toThrow(
        /kernel.*expected one of.*quartic.*bisquare/
      );
    }
    expect(
      () =>
        new GPUSpatialWeightsTransform({
          weights: weights.chunks[0].weights,
          operation: 'kernel',
          bandwidth: 1,
          kernel: unknown
        })
    ).toThrow(/kernel.*expected one of/);
    expect(
      () =>
        new GPUChunkedSpatialWeightsTransform({
          weights,
          operation: 'kernel',
          bandwidth: 1,
          kernel: unknown
        })
    ).toThrow(/kernel.*expected one of/);
  } finally {
    graph.device.destroy();
  }
});
