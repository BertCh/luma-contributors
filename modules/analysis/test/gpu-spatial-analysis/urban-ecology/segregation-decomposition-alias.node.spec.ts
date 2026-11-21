// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {addSegregationDecompositionRecipe} from '../../../src/gpu-spatial-analysis/landscape-ecology/segregation-decomposition';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it.each([
  0, 16, 28
])('rejects population/output overlap through distinct handles at byte %i before mutation and permits retry', byteOffset => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const physical = device.createBuffer({byteLength: 1024, usage: Buffer.STORAGE});
  try {
    const first = graph.importBuffer(
      {id: 'population', byteLength: 1024, usage: Buffer.STORAGE},
      physical
    );
    const second = graph.importBuffer(
      {id: 'alias', byteLength: 1024, usage: Buffer.STORAGE},
      physical
    );
    const population = graph.createDataView(first, {format: 'float32x2', length: 4});
    const indices = graph.createDataView(second, {format: 'float32', length: 4, byteOffset});
    const props = {
      id: 'retry',
      populations: [population, population, population, population] as const,
      supportId: 'common',
      counterfactualMethod: 'supplied'
    };
    const before = [...(graph as unknown as {buffers: Map<string, unknown>}).buffers.keys()];
    expect(() => addSegregationDecompositionRecipe(graph, {...props, output: {indices}})).toThrow(
      /logical byte ranges must not overlap/
    );
    expect((graph as unknown as {nodes: unknown[]}).nodes).toHaveLength(0);
    expect([...(graph as unknown as {buffers: Map<string, unknown>}).buffers.keys()]).toEqual(
      before
    );
    const result = addSegregationDecompositionRecipe(graph, {
      ...props,
      output: {indices: createTransientView(graph, 'safe', 'float32', 4)}
    });
    expect(result.outputs.indices.length).toBe(4);
    expect((graph as unknown as {nodes: unknown[]}).nodes.length).toBeGreaterThan(0);
  } finally {
    physical.destroy();
    device.destroy();
  }
});

it('accepts adjacent population and indices byte ranges on one handle across separate dispatches', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const storage = createTransientView(graph, 'storage', 'uint32', 256);
    const population = graph.createDataView(storage.buffer, {format: 'float32x2', length: 4});
    // [0,32) and [32,48) are adjacent, never co-bound: units read population;
    // each index publication reads only the reductions' separate transient buffers.
    const indices = graph.createDataView(storage.buffer, {
      format: 'float32',
      length: 4,
      byteOffset: 32
    });
    const result = addSegregationDecompositionRecipe(graph, {
      populations: [population, population, population, population],
      supportId: 'common',
      counterfactualMethod: 'supplied',
      output: {indices}
    });
    expect(result.outputs.indices).toBe(indices);
  } finally {
    device.destroy();
  }
});

it('rejects output/output and randomization read/write overlap before adding scenarios', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const population = createTransientView(graph, 'population', 'float32x2', 4);
    const storage = createTransientView(graph, 'storage', 'uint32', 256);
    const props = {
      populations: [population, population, population, population] as const,
      supportId: 'common',
      counterfactualMethod: 'supplied'
    };
    const indices = graph.createDataView(storage.buffer, {format: 'float32', length: 4});
    const validity = graph.createDataView(storage.buffer, {
      format: 'uint32',
      length: 4,
      byteOffset: 12
    });
    expect(() =>
      addSegregationDecompositionRecipe(graph, {...props, output: {indices, validity}})
    ).toThrow(/logical byte ranges must not overlap/);
    const swaps = graph.createDataView(storage.buffer, {format: 'uint32', length: 16});
    expect(() =>
      addSegregationDecompositionRecipe(graph, {
        ...props,
        randomization: {replicates: 4, seed: 1, swaps},
        output: {distribution: indices}
      })
    ).toThrow(/logical byte ranges must not overlap/);
    expect((graph as unknown as {nodes: unknown[]}).nodes).toHaveLength(0);
  } finally {
    device.destroy();
  }
});

it.each([
  ['indices', 'float32', 4],
  ['validity', 'uint32', 4],
  ['decomposition', 'float32x4', 1],
  ['distribution', 'float32', 4],
  ['inference', 'float32x4', 1]
] as const)('preflights population overlap for output.%s before any scenario', (name, format, length) => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const population = createTransientView(graph, 'population', 'float32x2', 4);
    const output = graph.createDataView(population.buffer, {format, length, byteOffset: 16});
    expect(() =>
      addSegregationDecompositionRecipe(graph, {
        populations: [population, population, population, population],
        supportId: 'common',
        counterfactualMethod: 'supplied',
        randomization: {replicates: 4, seed: 1},
        output: {[name]: output}
      })
    ).toThrow(/logical byte ranges must not overlap/);
    expect((graph as unknown as {nodes: unknown[]}).nodes).toHaveLength(0);
  } finally {
    device.destroy();
  }
});

it('rejects disjoint co-bound randomization input/output ranges before scenarios', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const storage = createTransientView(graph, 'storage', 'uint32', 256);
    const population = graph.createDataView(storage.buffer, {format: 'float32x2', length: 4});
    const distribution = graph.createDataView(storage.buffer, {
      format: 'float32',
      length: 4,
      byteOffset: 256
    });
    expect(() =>
      addSegregationDecompositionRecipe(graph, {
        populations: [population, population, population, population],
        supportId: 'common',
        counterfactualMethod: 'supplied',
        randomization: {replicates: 4, seed: 1},
        output: {distribution}
      })
    ).toThrow(/outputs must not share buffers with inputs/);
    expect((graph as unknown as {nodes: unknown[]}).nodes).toHaveLength(0);
  } finally {
    device.destroy();
  }
});
