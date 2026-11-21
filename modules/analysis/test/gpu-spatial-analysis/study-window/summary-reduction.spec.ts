// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {GPUStudyWindowPreparation} from '../../../src/gpu-spatial-analysis/study-window';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {getSummaryReductionPrefixCount} from './summary-reduction-prefix';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

it('CORE-118 and W2-05 publish complete multi-block summaries and four cell words after a discarded reduction prefix', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const ringCount = 513;
  const coordinates = new Float32Array(8 * ringCount);
  coordinates.set([0, 0, 4096, 0, 4096, 4096, 0, 4096]);
  for (let ring = 1; ring < ringCount; ring++) {
    const left = 5000 + 2 * (ring - 1);
    coordinates.set([left, 0, left + 1, 0, left + 1, 1, left, 1], 8 * ring);
  }
  const positionBuffer = track(createInputBuffer(device, coordinates));
  const input = (name: string, values: Uint32Array) =>
    importGraphBuffer(
      graph,
      name,
      track(createInputBuffer(device, values)),
      'uint32',
      values.length
    );
  const summaryBuffer = track(createOutputBuffer(device, 8));
  const parameterBuffer = track(createOutputBuffer(device, 4));
  const summary = importGraphBuffer(graph, 'output-summary', summaryBuffer, 'float32', 8);
  const parameterOutput = importGraphBuffer(graph, 'output-cell', parameterBuffer, 'float32', 4);
  const preparation = new GPUStudyWindowPreparation({
    positions: importGraphBuffer(
      graph,
      'positions',
      positionBuffer,
      'float32x2',
      coordinates.length / 2
    ),
    featureOffsets: input('features', Uint32Array.of(0, ringCount)),
    polygonOffsets: input(
      'polygons',
      Uint32Array.from({length: ringCount + 1}, (_, index) => index)
    ),
    ringOffsets: input(
      'rings',
      Uint32Array.from({length: ringCount + 1}, (_, index) => 4 * index)
    ),
    samplingGrid: [2, 2],
    output: {summary}
  });
  const createView = vi.spyOn(graph, 'createDataView');
  const nodes = preparation.getCommandNodes(graph);
  const parameters = createView.mock.results.find(
    result =>
      result.type === 'return' && result.value.buffer.id === 'study-window-sample-parameters'
  )!.value as GraphDataView<'float32'>;
  createView.mockRestore();
  for (const node of nodes) graph.add(node);
  graph.add(
    createWGSLKernelNode(graph, {
      id: 'read-cell',
      operation: 'test',
      invocationCount: 4,
      bindings: [
        {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
        {name: 'output', view: parameterOutput, type: 'f32', access: 'read_write'}
      ],
      body: 'output[outputOffset + index] = parameters[parametersOffset + index];'
    })
  );
  const internalSummaryBuffer = preparation.getViews(graph).summary.buffer.defaultBuffer as Buffer;
  expect(internalSummaryBuffer).toBeDefined();
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    // Closed forms: one 4096^2 square and 512 disjoint unit squares. Each has four edges.
    // Bounds maxX = 5000 + 2*511 + 1; perimeter = 4*4096 + 4*512.
    const expected = [0, 0, 6023, 4096, 16777728, 18432, 2052, 0];
    expect(await readFloat32(summaryBuffer, 8)).toEqual(expected);
    expect(await readFloat32(parameterBuffer, 4)).toEqual([0, 0, 3011.5, 2048]);
    const shifted = Float32Array.from(
      coordinates,
      (value, index) => value + (index % 2 === 0 ? 10 : 20)
    );
    positionBuffer.write(shifted);
    preparation.invalidate();
    const encoder = device.createCommandEncoder();
    try {
      // Submit initialization and a few merges, leaving publication and generation commit pending.
      const prefixCount = getSummaryReductionPrefixCount(compiled.stats.nodeOrder);
      expect(compiled.stats.nodeOrder[prefixCount - 1]).toBe('study-window-summary-reduce-4');
      expect(compiled.stats.nodeOrder.slice(0, prefixCount)).toContain(
        'study-window-summary-partials'
      );
      expect(compiled.stats.nodeOrder.slice(0, prefixCount)).not.toContain(
        'study-window-summarize'
      );
      expect(compiled.stats.nodeOrder.slice(0, prefixCount)).not.toContain(
        'study-window-snapshot-commit'
      );
      const execution = compiled.createExecution({
        maximumInvocationCount: 1_000_000,
        maximumNodeCount: prefixCount
      });
      expect(execution.encodeNext(encoder, {parameters: undefined}).nodeCount).toBe(prefixCount);
      const commandBuffer = encoder.finish();
      device.submit(commandBuffer);
      commandBuffer.destroy();
    } finally {
      encoder.destroy();
    }
    expect(await readFloat32(summaryBuffer, 8)).toEqual(expected);
    expect(await readFloat32(internalSummaryBuffer, 8)).toEqual(expected);
    // The requested generation is still pending. Change the source again WITHOUT invalidation;
    // an early commit would keep the earlier +10/+20 snapshot and fail the new +30/+40 bounds.
    positionBuffer.write(
      Float32Array.from(coordinates, (value, index) => value + (index % 2 === 0 ? 30 : 40))
    );
    submitGraph(device, compiled, undefined);
    expect(await readFloat32(summaryBuffer, 8)).toEqual([
      30, 40, 6053, 4136, 16777728, 18432, 2052, 0
    ]);
    expect(await readFloat32(parameterBuffer, 4)).toEqual([30, 40, 3011.5, 2048]);
    expect(await readFloat32(internalSummaryBuffer, 8)).toEqual([
      30, 40, 6053, 4136, 16777728, 18432, 2052, 0
    ]);
    // A clean replay consumes the completed generation, without reusing discarded partials.
    submitGraph(device, compiled, undefined);
    expect(await readFloat32(summaryBuffer, 8)).toEqual([
      30, 40, 6053, 4136, 16777728, 18432, 2052, 0
    ]);
  } finally {
    compiled.destroy();
    preparation.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});
