// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, GraphVectorView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUVariableStandardization,
  type GPUVariableStandardizationMethod
} from '../../../src/dataframe-analysis/variable-standardization/index';
import {
  createNullWebGPUDevice,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {standardizeReference} from './standardization-oracle';

const methods: GPUVariableStandardizationMethod[] = [
  'deviation-from-mean',
  'z-score',
  'mad',
  'range-adjust',
  'range-standardize'
];

it('GeoDa recipes distinguish sample, population, mean absolute deviation and range adjustment', () => {
  expect(standardizeReference([1, 2, 3], 'z-score')).toEqual([-1, 0, 1]);
  expect(standardizeReference([1, 2, 3], 'z-score', undefined, 'population')[2]).toBeCloseTo(
    Math.sqrt(1.5)
  );
  // Arithmetic mean = 4, mean absolute deviation = 16 / 3; median absolute deviation = 0.
  expect(standardizeReference([0, 0, 12], 'mad')).toEqual([-0.75, -0.75, 1.5]);
  expect(standardizeReference([-6, -3, -1], 'range-adjust')).toEqual([-1.2, -0.6, -0.2]);
  expect(standardizeReference([10, 5, 0], 'range-standardize')).toEqual([1, 0.5, 0]);
  expect(standardizeReference([7, 7], 'mad')).toEqual([7, 7]);
  expect(standardizeReference([7, 7], 'z-score')).toEqual([0, 0]);
  expect(Number.isNaN(standardizeReference([7], 'z-score')[0])).toBe(true);
});

it.each(
  methods
)('GPUVariableStandardization %s composes shared reductions and parseable elementwise WGSL', method => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const chunks = [2, 3].map((length, index) =>
    createTransientView(graph, `values-${index}`, 'float32', length)
  );
  const values = new GraphVectorView({
    id: 'values',
    name: 'values',
    format: 'float32',
    length: 5,
    valueLength: 5,
    stride: 1,
    byteStride: 4,
    rowByteLength: 4,
    data: chunks
  });
  const output = createTransientView(graph, 'output', 'float32', 5);
  const mask = createTransientView(graph, 'mask', 'uint32', 5);
  const contributor = new GPUVariableStandardization({values, output, mask, method});
  const nodes = contributor.getCommandNodes(graph);
  const sources = getParsedKernelSources(nodes);
  expect(sources.size).toBeGreaterThan(4);
  expect([...sources.keys()].some(name => name.includes('-count'))).toBe(true);
  expect(output.length).toBe(5);
  device.destroy();
});

it('GPUVariableStandardization validates topology, format and ownership', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const values = createTransientView(graph, 'values', 'float32', 3);
  const output = createTransientView(graph, 'output', 'float32', 3);
  expect(() => new GPUVariableStandardization({values, output: values, method: 'mad'})).toThrow(
    /share buffers/
  );
  expect(
    () =>
      new GPUVariableStandardization({
        values,
        output: createTransientView(graph, 'short', 'float32', 2),
        method: 'mad'
      })
  ).toThrow(/length/);
  const overlapping = new GraphVectorView({
    id: 'overlap',
    name: 'overlap',
    format: 'float32',
    length: 3,
    valueLength: 3,
    stride: 1,
    byteStride: 4,
    rowByteLength: 4,
    data: [
      graph.createDataView(output.buffer, {format: 'float32', length: 2}),
      graph.createDataView(output.buffer, {format: 'float32', length: 1})
    ]
  });
  expect(
    () => new GPUVariableStandardization({values, output: overlapping, method: 'mad'})
  ).toThrow(/overlap/);
  const otherGraph = new GPUCommandGraph(device);
  const foreign = createTransientView(otherGraph, 'foreign', 'float32', 3);
  expect(() =>
    new GPUVariableStandardization({values: foreign, output, method: 'mad'}).getCommandNodes(graph)
  ).toThrow(/target graph/);
  device.destroy();
});

it('GPUVariableStandardization accepts an empty logical column', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const values = createTransientView(graph, 'empty-values', 'float32', 0);
  const output = createTransientView(graph, 'empty-output', 'float32', 0);
  const nodes = new GPUVariableStandardization({values, output, method: 'mad'}).getCommandNodes(
    graph
  );
  expect(getParsedKernelSources(nodes).size).toBeGreaterThan(0);
  device.destroy();
});

it('GPUVariableStandardization emits an unmasked recipe without a mask binding', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const values = createTransientView(graph, 'values', 'float32', 3);
  const output = createTransientView(graph, 'output', 'float32', 3);
  const sources = getParsedKernelSources(
    new GPUVariableStandardization({values, output, method: 'z-score'}).getCommandNodes(graph)
  );
  expect(
    [...sources.values()]
      .filter(source => source.includes('fn isFiniteBits'))
      .every(source => !source.includes('maskOffset'))
  ).toBe(true);
  device.destroy();
});
