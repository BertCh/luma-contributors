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
  getParsedKernelSources,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys
} from '../../utils/gpu-contributor-test-utils';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
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
  const values = createTransientView(graph, 'values', 'float32', 5);
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

it('GPUVariableStandardization rejects mismatched chunk topology', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const output = createTransientView(graph, 'output', 'float32', 3);
  const makeTwoChunkVector = <Format extends 'float32' | 'uint32'>(format: Format, name: string) =>
    new GraphVectorView<Format>({
      id: name,
      name,
      format,
      length: 3,
      valueLength: 3,
      stride: 1,
      byteStride: 4,
      rowByteLength: 4,
      data: [
        createTransientView(graph, `${name}-a`, format, 2),
        createTransientView(graph, `${name}-b`, format, 1)
      ]
    });
  expect(() =>
    new GPUVariableStandardization({
      values: makeTwoChunkVector('float32', 'chunked-values'),
      output,
      method: 'mad'
    }).getCommandNodes(graph)
  ).toThrow(/topology/);
  expect(() =>
    new GPUVariableStandardization({
      values: createTransientView(graph, 'values', 'float32', 3),
      mask: makeTwoChunkVector('uint32', 'chunked-mask'),
      output,
      method: 'mad'
    }).getCommandNodes(graph)
  ).toThrow(/topology/);
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

it('standardization keeps uint32 count reduction and fixed pipelines across uneven chunk counts', () => {
  const build = (lengths: number[]) => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const length = lengths.reduce((total, value) => total + value, 0);
    let serial = 0;
    const vector = (name: string) =>
      new GraphVectorView({
        id: name,
        name,
        format: 'float32',
        length,
        valueLength: length,
        stride: 1,
        byteStride: 4,
        rowByteLength: 4,
        data: lengths.map(rows =>
          createTransientView(graph, `${name}-${serial++}`, 'float32', rows)
        )
      });
    try {
      const contributor = new GPUVariableStandardization({
        values: vector('values'),
        output: vector('output'),
        method: 'z-score'
      });
      const nodes = contributor.getCommandNodes(graph);
      expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
      const sources = getGPUCommandNodeKernelSources(nodes);
      for (const kernel of sources)
        expect(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
      for (const kernel of sources.filter(kernel => kernel.source.includes('fn getRatio'))) {
        expect(kernel.source).toContain('return getScaledQuotient(numerator, denominator, 1.0, 0)');
        expect(kernel.source).toContain('frexp(value)');
        expect(kernel.source).not.toContain('numerator * 1e-19');
      }
      const reduction = sources.find(
        kernel => kernel.id.includes('-count') && kernel.source.includes('inputValues')
      )!;
      expect(reduction.source).toMatch(/inputValues:\s*array<u32>/);
      expect(reduction.source).toMatch(/outputValues:\s*array<u32>/);
      const dispatches = nodes.filter(node => node.id.includes('-prepare-'));
      expect(dispatches.map(node => node.workload?.maximumInvocationCount)).toEqual(
        lengths.filter(rows => rows > 0)
      );
      expect(dispatches.map(node => node.workload?.maximumWorkgroupCount)).toEqual(
        lengths.filter(rows => rows > 0).map(rows => Math.ceil(rows / 256))
      );
      return getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true});
    } finally {
      device.destroy();
    }
  };
  expect(build([1, 0, 256, 768])).toEqual(build([1025]));
});
